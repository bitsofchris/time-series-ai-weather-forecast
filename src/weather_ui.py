"""Helpers for the weather-page-style UI: emoji mapping, headline forecast
formatting, current-conditions hero block, and the combined Plotly figure."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .forecast import TotoForecast


# Order matters — match more specific terms first.
_EMOJI_RULES: list[tuple[str, str]] = [
    ("Thunder", "⛈"),
    ("Tornado", "🌪"),
    ("Hurricane", "🌀"),
    ("Tropical", "🌀"),
    ("Snow", "❄️"),
    ("Sleet", "🌨"),
    ("Freezing", "🌨"),
    ("Hail", "🌨"),
    ("Rain", "🌧"),
    ("Showers", "🌧"),
    ("Drizzle", "🌦"),
    ("Fog", "🌫"),
    ("Haze", "🌫"),
    ("Smoke", "🌫"),
    ("Mostly Cloudy", "☁️"),
    ("Partly Cloudy", "⛅"),
    ("Mostly Sunny", "🌤"),
    ("Partly Sunny", "⛅"),
    ("Cloud", "☁️"),
    ("Sunny", "☀️"),
    ("Clear", "🌙"),
    ("Windy", "💨"),
    ("Breezy", "💨"),
]


def emoji_for(short_forecast: str | None) -> str:
    if not short_forecast:
        return "🌡"
    for needle, glyph in _EMOJI_RULES:
        if needle.lower() in short_forecast.lower():
            return glyph
    return "🌡"


def _fmt_hour_local(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(tz).strftime("%-I %p")


def hi_lo(series: pd.Series, tz: str) -> tuple[float, str, float, str]:
    """Return (high, hour-of-high, low, hour-of-low) over the series."""
    s = series.dropna()
    hi_t = s.idxmax()
    lo_t = s.idxmin()
    return float(s.max()), _fmt_hour_local(hi_t, tz), float(s.min()), _fmt_hour_local(lo_t, tz)


def hero_markdown(
    place: str,
    history: pd.DataFrame,
    nws_first: pd.Series | None,
    tz: str,
) -> str:
    """A 'now' tile: current temp/RH/P with hour-over-hour delta and weather words."""
    if history.empty:
        return "_(no current readings yet)_"
    last = history.dropna(how="all").index.max()
    cur = history.loc[last]
    prev_idx = history.index[history.index < last]
    prev = history.loc[prev_idx[-1]] if len(prev_idx) else None

    def delta(col: str, unit: str, fmt: str = "+.1f") -> str:
        if prev is None or pd.isna(prev.get(col)) or pd.isna(cur.get(col)):
            return ""
        d = cur[col] - prev[col]
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "·")
        return f" <span style='opacity:0.6'>{arrow} {d:{fmt}} {unit}/h</span>"

    short = ""
    glyph = "🌡"
    if nws_first is not None and not nws_first.empty:
        first_row = nws_first.iloc[0] if isinstance(nws_first, pd.DataFrame) else None
        if isinstance(first_row, pd.Series) and "short_forecast" in first_row:
            short = str(first_row["short_forecast"])
            glyph = emoji_for(short)

    when = last.tz_convert(tz).strftime("%-I:%M %p %Z")
    lines = [
        f"### {glyph} {place} · {cur['temp_f']:.1f}°F",
        f"<span style='font-size:1.1em'>{cur['humidity']:.0f}% RH · {cur['pressure_inhg']:.2f} inHg{delta('temp_f','°F')}{delta('humidity','%','+.0f')}{delta('pressure_inhg','inHg','+.3f')}</span>",
        f"<span style='opacity:0.6'>Last reading {when} · NWS now: {short or '—'}</span>",
    ]
    return "\n\n".join(lines)


def headline_forecast_markdown(
    toto: TotoForecast,
    nws_temp: pd.Series | None,
    tz: str,
) -> str:
    """Side-by-side hi/lo plus Toto confidence chip at +24h."""
    t_hi, t_hi_t, t_lo, t_lo_t = hi_lo(toto.median, tz)
    width24 = float(toto.p90.iloc[-1] - toto.p10.iloc[-1])

    toto_block = (
        f"**🤖 Toto says (next 24h)**\n\n"
        f"High **{t_hi:.0f}°F** at {t_hi_t} · Low **{t_lo:.0f}°F** at {t_lo_t}\n\n"
        f"<span style='opacity:0.6'>80% interval at +24h: ±{width24/2:.1f}°F</span>"
    )

    if nws_temp is None or nws_temp.empty:
        nws_block = "**🌎 NWS** _(unavailable)_"
    else:
        n_hi, n_hi_t, n_lo, n_lo_t = hi_lo(nws_temp, tz)
        nws_block = (
            f"**🌎 NWS says (next 24h)**\n\n"
            f"High **{n_hi:.0f}°F** at {n_hi_t} · Low **{n_lo:.0f}°F** at {n_lo_t}\n\n"
            f"<span style='opacity:0.6'>Point forecast (no interval)</span>"
        )

    # Two columns via Markdown table with bare cells.
    return f"| {toto_block} | {nws_block} |\n|---|---|"


def emoji_strip_markdown(nws_df: pd.DataFrame, tz: str, n: int = 12) -> str:
    """Compact horizontal strip: hour | emoji | temp for the next n NWS hours."""
    if nws_df is None or nws_df.empty:
        return ""
    df = nws_df.head(n)
    hours = " | ".join(_fmt_hour_local(t, tz) for t in df.index)
    glyphs = " | ".join(emoji_for(s) for s in df.get("short_forecast", pd.Series([None]*len(df))))
    temps = " | ".join(f"{t:.0f}°" for t in df["temp_f"])
    sep = "|---" * len(df) + "|"
    return f"| {hours} |\n{sep}\n| {glyphs} |\n| {temps} |"


def combined_figure(
    history: pd.DataFrame,
    totos: dict[str, TotoForecast],
    nws_df: pd.DataFrame | None,
    metrics: list[dict],
    now: pd.Timestamp | None = None,
) -> go.Figure:
    """Three stacked subplots sharing the x-axis."""
    fig = make_subplots(
        rows=len(metrics), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[m["title"] for m in metrics],
    )
    showlegend = True
    for i, m in enumerate(metrics, start=1):
        col = m["col"]
        if col not in history.columns:
            continue
        hist = history[col].dropna()
        toto = totos.get(col)

        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist.values,
                name="Ecowitt (past)", mode="lines",
                line=dict(color="#222", width=2),
                showlegend=showlegend, legendgroup="hist",
            ),
            row=i, col=1,
        )
        if toto is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(toto.p90.index) + list(toto.p10.index[::-1]),
                    y=list(toto.p90.values) + list(toto.p10.values[::-1]),
                    fill="toself", fillcolor="rgba(31,119,180,0.18)",
                    line=dict(width=0), hoverinfo="skip",
                    name="Toto 10–90% interval",
                    showlegend=showlegend, legendgroup="toto-band",
                ),
                row=i, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=toto.median.index, y=toto.median.values,
                    name="Toto median", mode="lines",
                    line=dict(color="#1f77b4", width=2, dash="dash"),
                    showlegend=showlegend, legendgroup="toto-med",
                ),
                row=i, col=1,
            )
        if nws_df is not None and m.get("nws_col") and m["nws_col"] in nws_df.columns:
            ns = nws_df[m["nws_col"]].dropna()
            if not ns.empty:
                fig.add_trace(
                    go.Scatter(
                        x=ns.index, y=ns.values,
                        name="NWS forecast", mode="lines",
                        line=dict(color="#d62728", width=2, dash="dot"),
                        showlegend=showlegend, legendgroup="nws",
                    ),
                    row=i, col=1,
                )
        if now is not None:
            fig.add_vline(x=now, line=dict(color="#888", dash="dot", width=1), row=i, col=1)
        fig.update_yaxes(title_text=m["y"], row=i, col=1)
        showlegend = False  # only first subplot shows legend entries

    fig.update_layout(
        height=720,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    return fig
