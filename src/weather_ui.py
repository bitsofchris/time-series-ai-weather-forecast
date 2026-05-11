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
    realtime: dict | None = None,
) -> str:
    """Two-row 'now' table: temperature only, with Ecowitt vs NWS this hour.

    Prefers `realtime` (from /device/real_time) for the live reading because
    the hourly bucket lags by up to 30 min on the GW3000B.
    """
    cur_temp: float | None = None
    when_ts: pd.Timestamp | None = None

    if realtime and realtime.get("temp_f") is not None and realtime.get("last_ts") is not None:
        cur_temp = float(realtime["temp_f"])
        when_ts = realtime["last_ts"]
    elif not history.empty:
        last = history.dropna(how="all").index.max()
        cur_temp = float(history.loc[last, "temp_f"])
        when_ts = last

    if cur_temp is None or when_ts is None:
        return "_(no current readings yet)_"

    when = when_ts.tz_convert(tz).strftime("%-I:%M %p %Z on %a %b %-d")

    nws_temp_str = "—"
    nws_short = ""
    glyph = "🌡"
    gap_str = ""
    nws_hour_label = "this hour"
    if nws_first is not None and not nws_first.empty:
        idx0 = nws_first.index[0] if isinstance(nws_first, pd.DataFrame) else nws_first.name
        if isinstance(idx0, pd.Timestamp):
            nws_hour_label = idx0.tz_convert(tz).strftime("%-I %p %Z")
        row = nws_first.iloc[0] if isinstance(nws_first, pd.DataFrame) else nws_first
        if isinstance(row, pd.Series):
            if "temp_f" in row and pd.notna(row["temp_f"]):
                nws_temp_str = f"**{row['temp_f']:.0f}°F**"
                gap = cur_temp - float(row["temp_f"])
                sign = "+" if gap >= 0 else ""
                gap_str = f" <span style='opacity:0.55'>(NWS off by {sign}{gap:.1f}°F)</span>"
            if "short_forecast" in row:
                nws_short = str(row["short_forecast"])
                glyph = emoji_for(nws_short)

    table = (
        "| Source | Temperature |\n"
        "|---|---|\n"
        f"| 📡 Ecowitt (measured) | **{cur_temp:.1f}°F** |\n"
        f"| 🌎 NWS forecast for {nws_hour_label} | {nws_temp_str}{gap_str} |"
    )
    return (
        f"### {glyph} {place}\n\n"
        f"{table}\n\n"
        f"<span style='opacity:0.55'>Last Ecowitt reading at {when}</span>"
    )


def aligned_comparison_markdown(
    toto: TotoForecast,
    nws_temp: pd.Series | None,
    tz: str,
    step_hours: int = 3,
    max_offset_hours: int = 24,
) -> str:
    """Future forecast table — same wall-clock hour for both models.

    Starts at the first forecast hour (i.e. the next hour after the most
    recent Ecowitt reading) and steps forward in `step_hours` increments.
    """
    if toto is None or toto.median.empty:
        return ""
    base = toto.median.index[0]  # first forecast hour
    base_day = base.tz_convert(tz).strftime("%a")

    def _nearest(series: pd.Series | None, target: pd.Timestamp):
        if series is None or series.empty:
            return None, None
        idx = series.index.get_indexer([target], method="nearest")[0]
        if idx < 0 or idx >= len(series):
            return None, None
        return series.index[idx], float(series.iloc[idx])

    rows = ["| When | 🤖 Toto | 🌎 NWS | Δ |", "|---|---|---|---|"]
    for h in range(0, max_offset_hours + 1, step_hours):
        target = base + pd.Timedelta(hours=h)
        t_idx, t_val = _nearest(toto.median, target)
        n_idx, n_val = _nearest(nws_temp, target)
        if t_val is None and n_val is None:
            continue
        local = (t_idx or n_idx).tz_convert(tz)
        if local.strftime("%a") == base_day:
            label = local.strftime("%-I %p")
        else:
            label = local.strftime("%a %-I %p")
        toto_str = f"**{t_val:.0f}°F**" if t_val is not None else "—"
        nws_str = f"**{n_val:.0f}°F**" if n_val is not None else "—"
        if t_val is not None and n_val is not None:
            d = t_val - n_val
            sign = "+" if d >= 0 else ""
            delta_str = f"{sign}{d:.1f}°F"
        else:
            delta_str = "—"
        rows.append(f"| {label} | {toto_str} | {nws_str} | {delta_str} |")
    return "\n".join(rows)


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
    past_toto: dict[str, pd.DataFrame] | None = None,
    past_nws: dict[str, pd.DataFrame] | None = None,
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
        # Past Toto forecasts vs the same hours' actuals.
        if past_toto and col in past_toto:
            pt = past_toto[col]
            fig.add_trace(
                go.Scatter(
                    x=pt.index, y=pt["p50"].values,
                    name="🤖 Toto (past forecasts)", mode="lines",
                    line=dict(color="rgba(31,119,180,0.55)", width=1.5),
                    showlegend=showlegend, legendgroup="toto-past",
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
        height=900,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    return fig
