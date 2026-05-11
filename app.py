"""Toto weather forecasting demo — Gradio app for HuggingFace Spaces.

Pulls live data from an Ecowitt GW3000B home weather station, runs Datadog's
Toto 2.0 (4M) for the next N hours, and shows it next to the National Weather
Service hourly forecast — plus a rolling Toto-vs-NWS scoreboard once enough
forecasts have aged into actuals.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import gradio as gr
import pandas as pd

from src import ecowitt, forecast_log, nws, persist, storage, sync
from src.forecast import forecast_series
from src.weather_ui import (
    aligned_comparison_markdown,
    combined_figure,
    hero_markdown,
)

AUTO_REFRESH_SECONDS = 15 * 60          # background tick + archive sync
CACHE_TTL_SECONDS = AUTO_REFRESH_SECONDS - 60  # so autorefresh always refetches
DISPLAY_TZ = os.environ.get("DISPLAY_TZ", "America/New_York")
PLACE_NAME = os.environ.get("PLACE_NAME", "Yaphank, NY")

# Display cadence options. Each maps to (Ecowitt cycle_type, pandas resample,
# history_days). The GW3000B uploads every 30 min, so '5min' cycle resamples
# to 30min anyway — included for completeness.
CYCLE_CONFIG: dict[str, tuple[str, str, int]] = {
    "Hourly":   ("30min", "1h",    7),
    "30-min":   ("30min", "30min", 7),
    "4-hour":   ("4hour", "4h",    30),
}
HORIZON_CONFIG: dict[str, int] = {"24 h": 24, "48 h": 48, "72 h": 72}

METRICS = [
    {"col": "temp_f",        "title": "Outdoor temperature", "y": "°F",     "nws_col": "temp_f"},
    {"col": "rain_in_hr",    "title": "Rainfall rate",       "y": "in/hr",  "nws_col": None},
    {"col": "humidity",      "title": "Outdoor humidity",    "y": "%",      "nws_col": "humidity"},
    {"col": "pressure_inhg", "title": "Barometric pressure", "y": "inHg",   "nws_col": None},
]


# --- TTL cache (multi-arg keys) ------------------------------------------
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


# --- data fetchers --------------------------------------------------------
@cached(CACHE_TTL_SECONDS)
def fetch_history(cycle_type: str, resample: str, days: int) -> pd.DataFrame:
    cfg = ecowitt.EcowittConfig.from_env()
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    raw = ecowitt.fetch_history(
        cfg, start, end, cycle_type=cycle_type,
        call_back="outdoor,pressure,rainfall_piezo",
    )
    return ecowitt.history_to_dataframe(raw, resample=resample)


@cached(60)  # short TTL — real-time is the freshness path
def fetch_realtime_snapshot() -> dict:
    """Return the most recent reading from /device/real_time as a flat dict.

    The hourly history bucket only fills once Ecowitt has at least one
    reading inside that hour, which can lag real time by up to 30 min on
    the GW3000B. /device/real_time returns the device's last reading with
    its own minute-resolution timestamp, so we use it for the live hero.
    """
    cfg = ecowitt.EcowittConfig.from_env()
    body = ecowitt.fetch_real_time(cfg, call_back="outdoor,pressure,rainfall_piezo")
    data = body.get("data") or {}
    out = {}

    def _val(node):
        if isinstance(node, dict) and "value" in node:
            try:
                return float(node["value"])
            except (TypeError, ValueError):
                return None
        return None

    def _ts(node):
        if isinstance(node, dict) and "time" in node:
            try:
                return pd.to_datetime(int(node["time"]), unit="s", utc=True)
            except (TypeError, ValueError):
                return None
        return None

    out["temp_f"] = _val(data.get("outdoor", {}).get("temperature"))
    out["humidity"] = _val(data.get("outdoor", {}).get("humidity"))
    out["pressure_inhg"] = _val(data.get("pressure", {}).get("relative"))
    out["rain_in_hr"] = _val(data.get("rainfall_piezo", {}).get("rain_rate"))
    out["last_ts"] = _ts(data.get("outdoor", {}).get("temperature"))
    return out


@cached(CACHE_TTL_SECONDS)
def fetch_nws(horizon_hours: int) -> pd.DataFrame:
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])
    return nws.hourly_forecast_df(lat, lon, hours=horizon_hours)


def _resample_hours(resample: str) -> float:
    return pd.to_timedelta(resample).total_seconds() / 3600.0


def _resample_nws_to(nws_df: pd.DataFrame, resample: str) -> pd.DataFrame:
    """NWS gives hourly periods. For coarser cadences, average. For finer
    (e.g. 30-min), forward-fill. Either way return on a regular index."""
    if nws_df.empty:
        return nws_df
    target_h = _resample_hours(resample)
    if target_h >= 1:
        return nws_df.resample(resample).mean(numeric_only=True)
    # Sub-hourly: upsample with forward-fill on the numeric columns.
    return nws_df.select_dtypes("number").resample(resample).ffill()


# --- main refresh ---------------------------------------------------------
def refresh(cycle_label: str = "Hourly", horizon_label: str = "24 h"):
    cycle_type, resample, hist_days = CYCLE_CONFIG[cycle_label]
    horizon_hours = HORIZON_CONFIG[horizon_label]
    step_hours = _resample_hours(resample)
    horizon_steps = max(1, int(round(horizon_hours / step_hours)))

    history = fetch_history(cycle_type, resample, hist_days)
    realtime = fetch_realtime_snapshot()
    nws_df_raw = fetch_nws(horizon_hours)
    nws_df = _resample_nws_to(nws_df_raw, resample)

    last_actual = history.dropna(how="all").index.max()
    nws_future = nws_df[nws_df.index > last_actual] if last_actual is not None else nws_df

    # The NWS response often starts with a period that began in the past
    # (forecasts are issued every ~hour but each period is 1h long). For
    # the hero, find the period that *contains* "now".
    now_utc = pd.Timestamp.now(tz="UTC")
    if not nws_df_raw.empty:
        covering = nws_df_raw[nws_df_raw.index <= now_utc]
        nws_first = covering.tail(1) if not covering.empty else nws_df_raw.head(1)
    else:
        nws_first = None

    # Log to SQLite (always at the chosen cadence)
    log_conn = forecast_log.connect()
    forecast_log.record_actuals(log_conn, history)

    totos: dict[str, object] = {}
    nws_aligned: dict[str, pd.Series] = {}
    for m in METRICS:
        series = history[m["col"]].dropna()
        if series.empty:
            continue
        toto = forecast_series(series, horizon=horizon_steps)
        totos[m["col"]] = toto
        forecast_log.record_toto(log_conn, m["col"], toto)
        if m["nws_col"] and m["nws_col"] in nws_future.columns:
            ns = nws_future[m["nws_col"]].dropna()
            nws_aligned[m["col"]] = ns
            forecast_log.record_nws(log_conn, m["col"], ns)

    now = pd.Timestamp.now(tz="UTC").floor("h")

    # Look back: for each metric, build a series of past predictions issued
    # for actual hours that have already passed. Lets us overlay what each
    # model thought vs what happened.
    visible_history = history.tail(int(hist_days * 24 / step_hours))
    since_unix = (
        int(visible_history.index.min().timestamp()) if not visible_history.empty else None
    )
    past_toto: dict[str, pd.DataFrame] = {}
    past_nws: dict[str, pd.DataFrame] = {}
    for m in METRICS:
        col = m["col"]
        pt = forecast_log.historical_predictions(log_conn, "toto", col, since_unix=since_unix)
        if not pt.empty:
            past_toto[col] = pt
        pn = forecast_log.historical_predictions(log_conn, "nws", col, since_unix=since_unix)
        if not pn.empty:
            past_nws[col] = pn

    fig = combined_figure(
        history=visible_history,
        totos=totos,
        nws_df=nws_future,
        metrics=METRICS,
        now=now,
        past_toto=past_toto,
        past_nws=past_nws,
    )

    hero = hero_markdown(PLACE_NAME, history, nws_first, DISPLAY_TZ, realtime=realtime)
    if "temp_f" in totos:
        comparison_md = "### 🆚 24-hour temperature forecast — same hour, side-by-side\n\n" + aligned_comparison_markdown(
            toto=totos["temp_f"],
            nws_temp=nws_aligned.get("temp_f"),
            tz=DISPLAY_TZ,
        )
    else:
        comparison_md = ""
    scoreboard = render_scoreboard(log_conn)

    # Backup forecast log to HF Dataset (non-blocking).
    persist.push_db_async()
    # The full archive sync + push happens in the autorefresh thread.

    return hero, comparison_md, fig, scoreboard


# --- scoreboard ----------------------------------------------------------
def render_scoreboard(conn) -> str:
    lines = ["### 📊 Forecast scoreboard (rolling 48h MAE — lower is better)"]
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
        nws_row = by.get("nws")
        parts = [f"**{label}**"]
        if toto is not None:
            parts.append(f"Toto **{toto['mae']:.2f} {unit}** _(n={int(toto['n'])})_")
        if nws_row is not None:
            parts.append(f"NWS **{nws_row['mae']:.2f} {unit}** _(n={int(nws_row['n'])})_")
        if toto is not None and nws_row is not None:
            diff = toto["mae"] - nws_row["mae"]
            winner = "🤖 Toto" if diff < 0 else "🌎 NWS"
            parts.append(f"→ **{winner}** wins by {abs(diff):.2f} {unit}")
        lines.append(" · ".join(parts))
    if not any_data:
        lines.append(
            "_No scored forecasts yet. The scoreboard fills in once forecasts have target hours that have already passed and matching Ecowitt actuals — typically within an hour or two of running._"
        )
    return "\n\n".join(lines)


# --- auto-refresh background thread --------------------------------------
ECOWITT_ARCHIVE_DB = "data/ecowitt.db"


def _sync_archive_all_cycles() -> None:
    """Refresh the SQLite archive (data/ecowitt.db) for every cycle_type
    so the local mirror of Ecowitt's storage stays current."""
    try:
        cfg = ecowitt.EcowittConfig.from_env()
    except RuntimeError:
        return
    conn = storage.connect(ECOWITT_ARCHIVE_DB)
    try:
        for cycle in sync.CYCLES:
            try:
                sync.sync_cycle(cfg, conn, cycle, verbose=False)
            except ecowitt.EcowittRateLimitError as err:
                print(f"[autorefresh] rate-limited on {cycle.name}: {err} — skipping rest")
                break
            except Exception:  # noqa: BLE001
                print(f"[autorefresh] sync error on {cycle.name}:")
                traceback.print_exc()
    finally:
        conn.close()


def _autorefresh_loop():
    while True:
        try:
            refresh()                  # live forecast + forecasts.db log
            _sync_archive_all_cycles() # 5min/30min/4hour raw archive
            persist.push_all_async()   # back up both DBs to HF Dataset
        except Exception:  # noqa: BLE001
            print("[autorefresh] error during refresh:")
            traceback.print_exc()
        time.sleep(AUTO_REFRESH_SECONDS)


def _start_autorefresh():
    threading.Thread(target=_autorefresh_loop, daemon=True, name="autorefresh").start()
    print(f"[autorefresh] started, interval={AUTO_REFRESH_SECONDS}s")


# --- UI -------------------------------------------------------------------
HOOK = (
    "**Language models predict the next token. "
    "What if you could predict the future with the same technology?**"
)
SUBTITLE = (
    "Live readings from my Ecowitt GW3000B + a probabilistic forecast from "
    "[Datadog's Toto 2.0 (4M)](https://huggingface.co/Datadog/Toto-2.0-4m), "
    "compared against the [NWS hourly forecast](https://www.weather.gov/documentation/services-web-api). "
    "The scoreboard tracks who's been more accurate over the past 48 hours."
)

with gr.Blocks(title="Toto Weather Forecast", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Toto on my home weather station")
    gr.Markdown(HOOK)
    gr.Markdown(SUBTITLE)

    hero_md = gr.Markdown()
    comparison_md = gr.Markdown()
    gr.HTML(
        # KOKX is the NWS radar site at Upton, NY — covers Long Island incl.
        # Westhampton Beach. The looped GIF refreshes itself on the browser
        # so the map stays live without us doing anything.
        '<div style="text-align:center;margin:0.5em 0;">'
        '<a href="https://radar.weather.gov/station/kokx/standard" target="_blank" rel="noopener">'
        '<img src="https://radar.weather.gov/ridge/standard/KOKX_loop.gif" '
        'alt="NWS radar loop, Long Island (KOKX)" '
        'style="max-width:560px;width:100%;border-radius:6px;border:1px solid #e0e0e0;" />'
        '</a>'
        '<div style="opacity:0.55;font-size:0.85em;margin-top:0.3em;">'
        'NWS radar loop · station KOKX (Upton, NY) · refreshes ~6 min'
        '</div>'
        '</div>'
    )

    with gr.Row():
        cycle_dd = gr.Dropdown(
            choices=list(CYCLE_CONFIG.keys()), value="Hourly",
            label="Display cadence", scale=1,
        )
        horizon_dd = gr.Dropdown(
            choices=list(HORIZON_CONFIG.keys()), value="24 h",
            label="Forecast horizon", scale=1,
        )
    gr.Markdown(
        "<span style='opacity:0.55'>🔄 Live data + forecast auto-refresh every 15 minutes.</span>"
    )

    scoreboard_md = gr.Markdown()
    plot = gr.Plot(label="Forecast")

    with gr.Accordion("How the forecast is made", open=False):
        gr.Markdown(
            "**Model.** [Datadog/Toto-2.0-4m](https://huggingface.co/Datadog/Toto-2.0-4m) "
            "(~4 M params, CPU). Smallest variant of Toto 2.0; the larger ones would tighten the band.\n\n"
            "**Input.** For each metric we feed Toto a univariate window of the most recent "
            "Ecowitt history at the chosen display cadence (default 1 h spacing). "
            "Toto requires the context length to be a multiple of its `patch_size=32`, so we "
            "truncate the oldest points to the largest multiple of 32 we have — or, if we have "
            "fewer than 32, left-pad to one patch and set `target_mask=False` on the padded "
            "steps so the model ignores them.\n\n"
            "**Output.** `model.forecast(...)` returns 9 analytical quantiles "
            "(`[0.1, 0.2, …, 0.9]`) for each future step — no Monte-Carlo sampling. "
            "We plot the p10–p90 band and the p50 median. "
            "**Horizon.** `horizon_steps = round(horizon_hours / step_hours)`; defaults give 24 hourly steps.\n\n"
            "**Cadence.** A daemon thread inside the Space re-runs the whole pipeline every "
            "15 minutes (cache TTL is 14 min, so each tick re-hits Ecowitt and NWS). Every "
            "snapshot is persisted to SQLite and backed up to a private HF Dataset, which is "
            "also what powers the side-by-side scoreboard and the past-forecast overlays "
            "above.\n\n"
            "Full spec: [`docs/toto-inference.md`](https://huggingface.co/spaces/bitsofchris/time-series-ai-weather-forecast/blob/main/docs/toto-inference.md)."
        )

    outputs = [hero_md, comparison_md, plot, scoreboard_md]
    inputs = [cycle_dd, horizon_dd]
    demo.load(refresh, inputs=inputs, outputs=outputs)
    cycle_dd.change(refresh, inputs=inputs, outputs=outputs)
    horizon_dd.change(refresh, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    persist.pull_all()  # bootstrap forecast log + archive from the HF Dataset
    _start_autorefresh()
    demo.launch()
