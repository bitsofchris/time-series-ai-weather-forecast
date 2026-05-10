"""Toto weather forecasting demo — Gradio app for HuggingFace Spaces.

Pulls live data from an Ecowitt GW3000B home weather station, runs Datadog's
Toto 2.0 (smallest, 4M params) to forecast the next 24h of temperature,
humidity, and pressure, and shows it next to the National Weather Service
forecast for the same window.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import gradio as gr
import pandas as pd

from src import ecowitt, nws
from src.forecast import forecast_series
from src.plotting import metric_figure

CACHE_TTL_SECONDS = 60 * 60  # 1 hour
HISTORY_DAYS = 7
HORIZON_HOURS = 24

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

    figs = []
    for m in METRICS:
        series = history[m["col"]].dropna()
        toto = forecast_series(series, horizon=HORIZON_HOURS)
        nws_series = (
            nws_df[m["nws_col"]] if (m["nws_col"] and m["nws_col"] in nws_df.columns) else None
        )
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
    return figs[0], figs[1], figs[2]


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
    temp_plot = gr.Plot(label="Temperature")
    humidity_plot = gr.Plot(label="Humidity")
    pressure_plot = gr.Plot(label="Pressure")

    outputs = [temp_plot, humidity_plot, pressure_plot]
    demo.load(refresh, outputs=outputs)
    refresh_btn.click(refresh, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
