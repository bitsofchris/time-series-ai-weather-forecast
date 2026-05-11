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
    toto_temp: TotoForecast | None = None,
    nws_temp: pd.Series | None = None,
    horizon_hours: int = 1,
) -> str:
    """Three-row 'now / N h-ahead' table: measured Ecowitt + each model's
    prediction for the same wall-clock hour `horizon_hours` from now."""
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

    eco_when = when_ts.tz_convert(tz).strftime("%-I:%M %p %Z, %a %b %-d")

    glyph = "🌡"
    if nws_first is not None and not nws_first.empty:
        row = nws_first.iloc[0] if isinstance(nws_first, pd.DataFrame) else nws_first
        if isinstance(row, pd.Series) and "short_forecast" in row:
            glyph = emoji_for(str(row["short_forecast"]))

    # 'Next round hour' — if it's 3:55, target = 4 PM. If it's 4:01,
    # target = 5 PM. Matches what people intuit by 'in the next hour'.
    target = pd.Timestamp.now(tz="UTC").ceil("h")

    def _nearest(series, target_ts):
        if series is None or series.empty:
            return None, None
        idx = series.index.get_indexer([target_ts], method="nearest")[0]
        if idx < 0 or idx >= len(series):
            return None, None
        return series.index[idx], float(series.iloc[idx])

    toto_idx, toto_val = _nearest(toto_temp.median if toto_temp is not None else None, target)
    nws_idx, nws_val = _nearest(nws_temp, target)

    def _row(label: str, val: float | None, ts):
        when = ts.tz_convert(tz).strftime("%-I %p %Z, %a %b %-d") if ts is not None else "—"
        cell = f"**{val:.0f}°F**" if val is not None else "—"
        return f"| {label} | {cell} | {when} |"

    table = (
        "| Source | Temperature | When |\n"
        "|---|---|---|\n"
        f"| 📡 Ecowitt (now) | **{cur_temp:.1f}°F** | {eco_when} |\n"
        f"{_row('🤖 Toto (next hour)', toto_val, toto_idx)}\n"
        f"{_row('🌎 NWS (next hour)', nws_val, nws_idx)}"
    )
    return f"### {glyph} {place}\n\n{table}"


def aligned_comparison_markdown(
    toto: TotoForecast,
    nws_temp: pd.Series | None,
    tz: str,
    offsets_hours: tuple[int, ...] = (1, 3, 12),
) -> str:
    """Future forecast table — same wall-clock hour for both models, at
    the same lookaheads we score on the scoreboard (1h / 3h / 12h)."""
    if toto is None or toto.median.empty:
        return ""
    now_utc = pd.Timestamp.now(tz="UTC")
    base_day = now_utc.tz_convert(tz).strftime("%a")

    def _nearest(series: pd.Series | None, target: pd.Timestamp):
        if series is None or series.empty:
            return None, None
        idx = series.index.get_indexer([target], method="nearest")[0]
        if idx < 0 or idx >= len(series):
            return None, None
        return series.index[idx], float(series.iloc[idx])

    rows = ["| Lookahead | When | 🤖 Toto | 🌎 NWS | Δ |", "|---|---|---|---|---|"]
    for h in offsets_hours:
        target = now_utc + pd.Timedelta(hours=h)
        t_idx, t_val = _nearest(toto.median, target)
        n_idx, n_val = _nearest(nws_temp, target)
        if t_val is None and n_val is None:
            continue
        local = (t_idx or n_idx).tz_convert(tz)
        if local.strftime("%a") == base_day:
            when_label = local.strftime("%-I %p")
        else:
            when_label = local.strftime("%a %-I %p")
        toto_str = f"**{t_val:.0f}°F**" if t_val is not None else "—"
        nws_str = f"**{n_val:.0f}°F**" if n_val is not None else "—"
        if t_val is not None and n_val is not None:
            d = t_val - n_val
            sign = "+" if d >= 0 else ""
            delta_str = f"{sign}{d:.1f}°F"
        else:
            delta_str = "—"
        rows.append(f"| **{h} h** | {when_label} | {toto_str} | {nws_str} | {delta_str} |")
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


def hero_gauges(
    cur_temp: float,
    toto_next: float | None,
    nws_next: float | None,
    temp_range: tuple[float, float] = (20.0, 100.0),
) -> go.Figure:
    """Three side-by-side gauges: current Ecowitt temperature, plus each
    model's prediction for the next round hour. Each forecast gauge also
    shows its delta vs the current reading."""
    cool_to_warm = [
        {"range": [20, 40], "color": "rgba(31,119,180,0.18)"},
        {"range": [40, 60], "color": "rgba(31,119,180,0.08)"},
        {"range": [60, 80], "color": "rgba(214,39,40,0.08)"},
        {"range": [80, 100], "color": "rgba(214,39,40,0.18)"},
    ]
    specs = [[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    fig = make_subplots(rows=1, cols=3, specs=specs)

    def _ind(value, title, bar_color, with_delta: bool):
        ind = go.Indicator(
            mode="gauge+number+delta" if with_delta else "gauge+number",
            value=value if value is not None else float("nan"),
            title={"text": title, "font": {"size": 14}},
            number={"suffix": " °F", "font": {"size": 30}},
            gauge=dict(
                axis=dict(range=list(temp_range), tickwidth=1, tickcolor="#888"),
                bar=dict(color=bar_color, thickness=0.25),
                bgcolor="white",
                borderwidth=1,
                bordercolor="#e0e0e0",
                steps=cool_to_warm,
                threshold=dict(line=dict(color=bar_color, width=4), value=value or 0),
            ),
            delta=(
                dict(reference=cur_temp, suffix=" °F", increasing={"color": "#d62728"}, decreasing={"color": "#1f77b4"})
                if with_delta else None
            ),
        )
        return ind

    fig.add_trace(_ind(cur_temp, "📡 Ecowitt (now)", "#222", False), row=1, col=1)
    fig.add_trace(_ind(toto_next, "🤖 Toto (next hour)", "#1f77b4", True), row=1, col=2)
    fig.add_trace(_ind(nws_next, "🌎 NWS (next hour)", "#d62728", True), row=1, col=3)
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def residual_figure(
    df: pd.DataFrame,
    title: str = "Forecast residual — 1 h-ahead prediction minus Ecowitt actual, last 48 h (°F)",
) -> go.Figure:
    """Plot signed residuals over time for Toto and NWS. Zero is perfect."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["toto_residual"],
            name="🤖 Toto residual", mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["nws_residual"],
            name="🌎 NWS residual", mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=5),
        )
    )
    fig.update_layout(
        title=title,
        height=320,
        hovermode="x unified",
        yaxis_title="°F (signed error)",
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    fig.update_xaxes(tickformat="%b %-d\n%-I %p", showgrid=True)
    return fig


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
        height=900,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    # Explicit date + hour on the x-axis so the reader doesn't have to guess
    # what day a tick refers to.
    fig.update_xaxes(
        tickformat="%b %-d\n%-I %p",
        ticklabelmode="instant",
        showgrid=True,
    )
    return fig
