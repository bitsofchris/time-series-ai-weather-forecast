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
    """A 'now' tile that explicitly distinguishes Ecowitt's measured value
    from NWS's forecast for the same hour, so the viewer can see the model
    error live."""
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
        return f" <span style='opacity:0.55'>({arrow} {d:{fmt}} {unit}/h)</span>"

    when = last.tz_convert(tz).strftime("%-I:%M %p %Z")

    # Pull NWS's forecast for the same wall-clock hour (its first period).
    nws_temp_str = "—"
    nws_short = "—"
    glyph = "🌡"
    if nws_first is not None and not nws_first.empty:
        row = nws_first.iloc[0] if isinstance(nws_first, pd.DataFrame) else nws_first
        if isinstance(row, pd.Series):
            if "temp_f" in row and pd.notna(row["temp_f"]):
                nws_temp_str = f"{row['temp_f']:.0f}°F"
            if "short_forecast" in row:
                nws_short = str(row["short_forecast"])
                glyph = emoji_for(nws_short)

    # Highlight the gap between actual and NWS prediction for this hour.
    gap_str = ""
    if nws_first is not None and not nws_first.empty:
        row = nws_first.iloc[0] if isinstance(nws_first, pd.DataFrame) else nws_first
        if isinstance(row, pd.Series) and "temp_f" in row and pd.notna(row["temp_f"]):
            gap = float(cur["temp_f"]) - float(row["temp_f"])
            sign = "+" if gap >= 0 else ""
            gap_str = f"<span style='opacity:0.55'>(NWS off by {sign}{gap:.1f}°F)</span>"

    lines = [
        f"### {glyph} {place}",
        (
            "| | Temperature | Humidity | Pressure | Conditions |\n"
            "|---|---|---|---|---|\n"
            f"| **📡 Ecowitt now** | **{cur['temp_f']:.1f}°F**{delta('temp_f','°F')}"
            f" | **{cur['humidity']:.0f}%**{delta('humidity','%','+.0f')}"
            f" | **{cur['pressure_inhg']:.2f} inHg**{delta('pressure_inhg','inHg','+.3f')}"
            f" | _(measured)_ |\n"
            f"| **🌎 NWS this hour** | **{nws_temp_str}** {gap_str} | — | — | {nws_short} |"
        ),
        f"<span style='opacity:0.55'>Last Ecowitt reading: {when}</span>",
    ]
    return "\n\n".join(lines)


def aligned_comparison_markdown(
    toto: TotoForecast,
    nws_temp: pd.Series | None,
    tz: str,
    offsets_hours: list[int] = (6, 12, 18, 24),
) -> str:
    """Apples-to-apples table: at the same future hour, show Toto and NWS.

    For each requested offset h, find the forecast point in each series
    closest to t0 + h hours and report both numbers in the same row.
    """
    if toto is None or toto.median.empty:
        return ""
    base = toto.median.index[0] - (toto.median.index[1] - toto.median.index[0]) if len(toto.median) > 1 else toto.median.index[0]

    def _nearest(series: pd.Series, target: pd.Timestamp):
        if series is None or series.empty:
            return None, None
        idx = series.index.get_indexer([target], method="nearest")[0]
        if idx < 0 or idx >= len(series):
            return None, None
        return series.index[idx], float(series.iloc[idx])

    rows = ["| When | 🤖 Toto | 🌎 NWS | Δ |", "|---|---|---|---|"]
    for h in offsets_hours:
        target = base + pd.Timedelta(hours=h)
        t_idx, t_val = _nearest(toto.median, target)
        n_idx, n_val = _nearest(nws_temp, target) if nws_temp is not None else (None, None)
        if t_val is None and n_val is None:
            continue
        when_label = (t_idx or n_idx).tz_convert(tz).strftime("%-I %p %a")
        toto_str = f"**{t_val:.0f}°F**" if t_val is not None else "—"
        nws_str = f"**{n_val:.0f}°F**" if n_val is not None else "—"
        if t_val is not None and n_val is not None:
            d = t_val - n_val
            sign = "+" if d >= 0 else ""
            delta_str = f"{sign}{d:.1f}°F"
        else:
            delta_str = "—"
        rows.append(f"| +{h}h · {when_label} | {toto_str} | {nws_str} | {delta_str} |")
    return "\n".join(rows)


def headline_forecast_blocks(
    toto: TotoForecast,
    nws_temp: pd.Series | None,
    tz: str,
) -> tuple[str, str]:
    """Side-by-side high/low summaries — same 24h window for both, so
    different peak/trough times become a real comparison."""
    t_hi, t_hi_t, t_lo, t_lo_t = hi_lo(toto.median, tz)
    width24 = float(toto.p90.iloc[-1] - toto.p10.iloc[-1])
    toto_md = (
        "### 🤖 Toto's 24h forecast\n\n"
        f"High **{t_hi:.0f}°F** at {t_hi_t}\n\n"
        f"Low **{t_lo:.0f}°F** at {t_lo_t}\n\n"
        f"<span style='opacity:0.55'>80% interval at +24h: ±{width24/2:.1f}°F</span>"
    )
    if nws_temp is None or nws_temp.empty:
        nws_md = "### 🌎 NWS 24h forecast\n\n_(unavailable)_"
    else:
        n_hi, n_hi_t, n_lo, n_lo_t = hi_lo(nws_temp, tz)
        nws_md = (
            "### 🌎 NWS 24h forecast\n\n"
            f"High **{n_hi:.0f}°F** at {n_hi_t}\n\n"
            f"Low **{n_lo:.0f}°F** at {n_lo_t}\n\n"
            "<span style='opacity:0.55'>Point forecast (no interval)</span>"
        )
    return toto_md, nws_md


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
                name="📡 Ecowitt (measured)", mode="lines",
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
                    mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="🤖 Toto 80% interval",
                    showlegend=showlegend, legendgroup="toto-band",
                ),
                row=i, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=toto.median.index, y=toto.median.values,
                    name="🤖 Toto median", mode="lines",
                    line=dict(color="#1f77b4", width=2.5),
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
                        name="🌎 NWS forecast", mode="lines",
                        line=dict(color="#d62728", width=2.5, dash="dash"),
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
