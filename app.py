"""Toto weather forecasting demo — Gradio app for HuggingFace Spaces.

Pulls live data from an Ecowitt GW3000B home weather station, runs Datadog's
Toto 2.0 (smallest, 4M params) to forecast the next 24h of temperature,
humidity, and pressure, and shows it next to the National Weather Service
forecast for the same window.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import gradio as gr
import pandas as pd

from src import ecowitt, forecast_log, nws
from src.forecast import forecast_series
from src.plotting import metric_figure

CACHE_TTL_SECONDS = 60 * 60  # 1 hour
HISTORY_DAYS = 7
HORIZON_HOURS = 24
AUTO_REFRESH_SECONDS = 60 * 60  # log a fresh forecast snapshot every hour

# Three metrics to forecast. Maps Ecowitt history column → plot config.
METRICS = [
    {"col": "temp_f",        "title": "Outdoor temperature",  "y": "°F",  "nws_col": "temp_f"},
    {"col": "humidity",      "title": "Outdoor humidity",      "y": "%",   "nws_col": "humidity"},
    {"col": "pressure_inhg", "title": "Barometric pressure",   "y": "inHg", "nws_col": None},
]


# --- tiny TTL cache (Gradio has no @st.cache_data equivalent) -------------
_cache: dict[tuple, tuple[float, object]] = {}


def cached(ttl: int):
    def deco(fn):
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            hit = _cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
            out = fn(*args, **kwargs)
            _cache[key] = (now, out)
            return out
        return wrapper
    return deco


# --- data fetchers ---------------------------------------------------------
@cached(CACHE_TTL_SECONDS)
def fetch_history() -> pd.DataFrame:
    cfg = ecowitt.EcowittConfig.from_env()
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=HISTORY_DAYS)
    raw = ecowitt.fetch_history(cfg, start, end, cycle_type="30min", call_back="outdoor,pressure")
    return ecowitt.history_to_dataframe(raw, resample="1h")


@cached(CACHE_TTL_SECONDS)
def fetch_nws() -> pd.DataFrame:
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    return nws.hourly_forecast_df(lat, lon, hours=HORIZON_HOURS)


# --- main refresh ---------------------------------------------------------
def refresh():
    history = fetch_history()
    nws_df = fetch_nws()
    now = pd.Timestamp.now(tz="UTC").floor("h")

    # NWS's first period and Ecowitt's last bucket describe the same wall-clock
    # hour; drop the overlap so all forecasts begin one hour after the last
    # observed actual.
    last_actual = history.dropna(how="all").index.max()
    nws_future = nws_df[nws_df.index > last_actual] if last_actual is not None else nws_df

    log_conn = forecast_log.connect()
    forecast_log.record_actuals(log_conn, history)

    figs = []
    for m in METRICS:
        series = history[m["col"]].dropna()
        toto = forecast_series(series, horizon=HORIZON_HOURS)
        forecast_log.record_toto(log_conn, m["col"], toto)

        nws_series = None
        if m["nws_col"] and m["nws_col"] in nws_future.columns:
            nws_series = nws_future[m["nws_col"]].dropna()
            forecast_log.record_nws(log_conn, m["col"], nws_series)

        figs.append(
            metric_figure(
                history=series.tail(HISTORY_DAYS * 24),
                toto=toto,
                nws=nws_series,
                title=m["title"],
                y_label=m["y"],
                now=now,
            )
        )
    scoreboard_md = render_scoreboard(log_conn)
    return figs[0], figs[1], figs[2], scoreboard_md


# --- scoreboard ----------------------------------------------------------
def render_scoreboard(conn) -> str:
    lines = ["### Forecast scoreboard (rolling 48h MAE, lower = better)"]
    any_data = False
    for metric, label, unit in [
        ("temp_f", "Temperature", "°F"),
        ("humidity", "Humidity", "%"),
        ("pressure_inhg", "Pressure", "inHg"),
    ]:
        summ = forecast_log.scoreboard_summary(conn, metric=metric, window_hours=48)
        if summ.empty:
            continue
        any_data = True
        by = {row["source"]: row for _, row in summ.iterrows()}
        toto = by.get("toto")
        nws = by.get("nws")
        parts = [f"**{label}**"]
        if toto is not None:
            parts.append(f"Toto {toto['mae']:.2f} {unit} (n={int(toto['n'])})")
        if nws is not None:
            parts.append(f"NWS {nws['mae']:.2f} {unit} (n={int(nws['n'])})")
        if toto is not None and nws is not None:
            diff = toto["mae"] - nws["mae"]
            winner = "Toto" if diff < 0 else "NWS"
            parts.append(f"→ **{winner}** by {abs(diff):.2f} {unit}")
        lines.append(" · ".join(parts))
    if not any_data:
        lines.append("_No scored forecasts yet — the scoreboard fills in once forecasts have target hours in the past with matching Ecowitt actuals (typically after the first hour of running)._")
    return "\n\n".join(lines)


# --- auto-refresh background thread --------------------------------------
def _autorefresh_loop():
    """Call refresh() on a schedule so we accumulate forecast snapshots even
    when nobody is loading the page. Errors are logged and swallowed so a
    transient API failure doesn't kill the thread."""
    while True:
        try:
            refresh()
        except Exception:  # noqa: BLE001
            print("[autorefresh] error during refresh:")
            traceback.print_exc()
        time.sleep(AUTO_REFRESH_SECONDS)


def _start_autorefresh():
    t = threading.Thread(target=_autorefresh_loop, daemon=True, name="autorefresh")
    t.start()
    print(f"[autorefresh] started, interval={AUTO_REFRESH_SECONDS}s")


# --- UI -------------------------------------------------------------------
HOOK = (
    "**Language models predict the next token. "
    "What if you could predict the future with the same technology?**"
)
SUBTITLE = (
    "Live readings from my Ecowitt GW3000B + a 24h forecast from "
    "[Datadog's Toto 2.0 (4M)](https://huggingface.co/Datadog/Toto-2.0-4m), "
    "compared against the [NWS hourly forecast](https://www.weather.gov/documentation/services-web-api)."
)

with gr.Blocks(title="Toto Weather Forecast") as demo:
    gr.Markdown("# Toto on my home weather station")
    gr.Markdown(HOOK)
    gr.Markdown(SUBTITLE)
    refresh_btn = gr.Button("Refresh forecast", variant="primary")
    scoreboard_md = gr.Markdown()
    temp_plot = gr.Plot(label="Temperature")
    humidity_plot = gr.Plot(label="Humidity")
    pressure_plot = gr.Plot(label="Pressure")

    outputs = [temp_plot, humidity_plot, pressure_plot, scoreboard_md]
    demo.load(refresh, outputs=outputs)
    refresh_btn.click(refresh, outputs=outputs)


if __name__ == "__main__":
    _start_autorefresh()
    demo.launch()
