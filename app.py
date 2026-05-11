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
    residual_figure,
)

AUTO_REFRESH_SECONDS = 15 * 60          # background tick + archive sync
CACHE_TTL_SECONDS = AUTO_REFRESH_SECONDS - 60  # so autorefresh always refetches
DISPLAY_TZ = os.environ.get("DISPLAY_TZ", "America/New_York")
PLACE_NAME = os.environ.get("PLACE_NAME", "Yaphank, NY")

# Single canonical view. History is read from the local SQLite archive
# (data/ecowitt.db) instead of hitting the Ecowitt API on every page load.
# The archive is kept current by the autorefresh thread + the per-Space-tick
# sync, so over time we accumulate true 5-min granularity beyond Ecowitt's
# own 24-30h 5-min retention window.
#
# context_days = how much past data we feed Toto. display_days = how much
# we show on the chart. Feeding the model a longer context than we display
# keeps the forecast informed by the full week without making the chart
# noisy.
VIEW_WEEK = {
    "label": "Past 5 days · 48 h forecast (5-min display, Toto fed hourly)",
    "cycle_type": "5min",
    "resample": "5min",            # display cadence on the chart
    "forecast_resample": "1h",     # cadence Toto actually consumes
    "display_days": 5,
    "context_days": 7,
    "horizon_hours": 48,
}

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
ECOWITT_ARCHIVE_DB_PATH = "data/ecowitt.db"


def fetch_history(cycle_type: str, resample: str, hours: float) -> pd.DataFrame:
    """Read history from the local SQLite archive. If the archive is empty
    (cold start before the first sync), fall back to a one-shot API pull so
    the page still renders something."""
    now_unix = int(time.time())
    since_unix = now_unix - int(hours * 3600)

    conn = storage.connect(ECOWITT_ARCHIVE_DB_PATH)
    try:
        df = storage.read_history_dataframe(
            conn, since_unix=since_unix, until_unix=now_unix,
            cycle_type=cycle_type, resample=resample,
        )
    finally:
        conn.close()

    if not df.empty:
        return df

    # Cold-start fallback: pull a small slice directly from the API so the
    # page isn't blank on the very first visit before sync has run.
    cfg = ecowitt.EcowittConfig.from_env()
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(hours=hours)
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
def _build_view(view: dict, log_conn, log_to_scoreboard: bool) -> dict:
    """Fetch + forecast for one view config. Returns intermediate pieces so
    the caller can stitch the page together."""
    cycle_type = view["cycle_type"]
    display_resample = view["resample"]
    forecast_resample = view.get("forecast_resample", display_resample)

    display_step_hours = _resample_hours(display_resample)
    forecast_step_hours = _resample_hours(forecast_resample)
    horizon_hours = view["horizon_hours"]
    horizon_steps = max(1, int(round(horizon_hours / forecast_step_hours)))

    # context_hours = what we feed Toto; display_hours = what we show on
    # the chart. Fall back to old keys for backward compatibility.
    context_hours = view.get(
        "context_hours",
        view.get("context_days", view.get("history_days", 0)) * 24
        or view.get("history_hours", 0),
    )
    display_hours = view.get(
        "display_hours",
        view.get("display_days", view.get("history_days", 0)) * 24
        or view.get("history_hours", 0),
    )

    history = fetch_history(cycle_type, display_resample, context_hours)
    # Coarser series for Toto inference: keeps the input length and
    # forecast horizon short enough for the 4M model to predict cleanly,
    # while the chart still shows the full 5-min granularity.
    if forecast_resample != display_resample:
        history_for_toto = history.resample(forecast_resample).mean()
    else:
        history_for_toto = history
    nws_df_raw = fetch_nws(horizon_hours)
    nws_df = _resample_nws_to(nws_df_raw, display_resample)
    last_actual = history.dropna(how="all").index.max()
    nws_future = nws_df[nws_df.index > last_actual] if last_actual is not None else nws_df

    if log_to_scoreboard:
        forecast_log.record_actuals(log_conn, history)

    totos: dict[str, object] = {}
    nws_aligned: dict[str, pd.Series] = {}
    for m in METRICS:
        series = history_for_toto[m["col"]].dropna()
        if series.empty:
            continue
        toto = forecast_series(series, horizon=horizon_steps)
        totos[m["col"]] = toto
        if log_to_scoreboard:
            forecast_log.record_toto(log_conn, m["col"], toto)
        if m["nws_col"] and m["nws_col"] in nws_future.columns:
            ns = nws_future[m["nws_col"]].dropna()
            nws_aligned[m["col"]] = ns
            if log_to_scoreboard:
                forecast_log.record_nws(log_conn, m["col"], ns)

    now = pd.Timestamp.now(tz="UTC").floor(display_resample)
    visible_steps = int(round(display_hours / display_step_hours))
    visible_history = history.tail(visible_steps)

    fig = combined_figure(
        history=visible_history,
        totos=totos,
        nws_df=nws_future,
        metrics=METRICS,
        now=now,
    )
    return {
        "fig": fig,
        "history": history,
        "totos": totos,
        "nws_aligned": nws_aligned,
        "nws_df_raw": nws_df_raw,
    }


def refresh():
    realtime = fetch_realtime_snapshot()
    log_conn = forecast_log.connect()

    week = _build_view(VIEW_WEEK, log_conn, log_to_scoreboard=True)

    # Hero uses the weekly history + the NWS period containing "now".
    nws_df_raw = week["nws_df_raw"]
    now_utc = pd.Timestamp.now(tz="UTC")
    if not nws_df_raw.empty:
        covering = nws_df_raw[nws_df_raw.index <= now_utc]
        nws_first = covering.tail(1) if not covering.empty else nws_df_raw.head(1)
    else:
        nws_first = None

    hero = hero_markdown(
        PLACE_NAME, week["history"], nws_first, DISPLAY_TZ,
        realtime=realtime,
        toto_temp=week["totos"].get("temp_f"),
        nws_temp=week["nws_aligned"].get("temp_f"),
        horizon_hours=1,
    )
    if "temp_f" in week["totos"]:
        comparison_md = (
            "### 🆚 Toto vs NWS — same hour, side-by-side\n\n"
            + aligned_comparison_markdown(
                toto=week["totos"]["temp_f"],
                nws_temp=week["nws_aligned"].get("temp_f"),
                tz=DISPLAY_TZ,
            )
        )
    else:
        comparison_md = ""
    scoreboard = render_scoreboard(log_conn)

    # Residual chart — 1 h-ahead, matching the first row of the scoreboard
    # table so the picture corresponds to the easiest headline number.
    resid_df = forecast_log.residuals(log_conn, metric="temp_f", window_hours=48, lag_hours=1.0)
    resid_fig = residual_figure(resid_df) if not resid_df.empty else None

    persist.push_db_async()
    return hero, comparison_md, week["fig"], scoreboard, resid_fig


# --- scoreboard ----------------------------------------------------------
SCOREBOARD_HORIZONS_H = [1, 3, 12]
# Pressure has no NWS counterpart, so the head-to-head scoreboard skips it.
SCOREBOARD_METRICS = [
    ("temp_f", "Temperature", "°F"),
    ("humidity", "Humidity", "%"),
]


def render_scoreboard(conn) -> str:
    """Per-metric, per-lookahead MAE table.

    Rows = forecast lookahead (1h / 3h / 12h). Cols = Toto MAE, NWS MAE,
    delta. Pressure has no NWS forecast so its column is dashed."""
    started_row = conn.execute(
        "SELECT MIN(forecast_made_at) FROM forecast_snapshots"
    ).fetchone()
    started_unix = started_row[0] if started_row and started_row[0] else None
    started_str = (
        datetime.fromtimestamp(started_unix, tz=timezone.utc)
        .astimezone(__import__("zoneinfo").ZoneInfo(DISPLAY_TZ))
        .strftime("%b %-d, %Y %-I:%M %p %Z")
        if started_unix else "—"
    )

    lines = [
        "### 📊 Forecast scoreboard (rolling 48 h MAE — lower is better)",
        (
            f"<span style='opacity:0.6'>**n** = number of past hours scored in the rolling 48 h window. "
            f"Scoreboard started {started_str}.</span>"
        ),
    ]
    any_data = False
    for metric, label, unit in SCOREBOARD_METRICS:
        rows: list[str] = []
        for lag_h in SCOREBOARD_HORIZONS_H:
            df = forecast_log.scoreboard_at_lag(
                conn, metric=metric, lag_hours=lag_h, window_hours=48,
            )
            if df.empty:
                continue
            any_data = True
            by = {r["source"]: r for _, r in df.iterrows()}
            toto = by.get("toto")
            nws_row = by.get("nws")
            t_cell = f"**{toto['mae']:.2f} {unit}** _(n={int(toto['n'])})_" if toto is not None else "—"
            n_cell = f"**{nws_row['mae']:.2f} {unit}** _(n={int(nws_row['n'])})_" if nws_row is not None else "—"
            if toto is not None and nws_row is not None:
                diff = toto["mae"] - nws_row["mae"]
                winner = "🤖 Toto" if diff < 0 else "🌎 NWS"
                d_cell = f"**{winner}** by {abs(diff):.2f} {unit}"
            elif toto is not None:
                d_cell = "—"
            else:
                d_cell = "—"
            rows.append(f"| **{lag_h} h-ahead** | {t_cell} | {n_cell} | {d_cell} |")
        if rows:
            # Build the whole table as one chunk — Markdown breaks the
            # table if there's a blank line between the header and rows,
            # and "\n\n".join below would insert exactly that.
            table = "\n".join(
                [
                    f"**{label}**",
                    "",
                    "| Lookahead | 🤖 Toto MAE | 🌎 NWS MAE | Δ |",
                    "|---|---|---|---|",
                    *rows,
                ]
            )
            lines.append(table)
    if not any_data:
        lines.append(
            "_No scored forecasts yet. The scoreboard fills in once forecasts have target hours that have already passed and matching Ecowitt actuals — typically within an hour or two of running._"
        )
    return "\n\n".join(lines)


# --- auto-refresh background thread --------------------------------------
ECOWITT_ARCHIVE_DB = ECOWITT_ARCHIVE_DB_PATH  # alias


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


def _sync_5min_only() -> None:
    """Quick sync of just the 5-min cycle (the one the display reads from).
    Called on startup so the first visitor sees fresh data."""
    try:
        cfg = ecowitt.EcowittConfig.from_env()
    except RuntimeError:
        return
    conn = storage.connect(ECOWITT_ARCHIVE_DB)
    try:
        cycle = next(c for c in sync.CYCLES if c.name == "5min")
        sync.sync_cycle(cfg, conn, cycle, verbose=False)
    except Exception:  # noqa: BLE001
        print("[startup] 5-min sync error:")
        traceback.print_exc()
    finally:
        conn.close()


def _autorefresh_loop():
    while True:
        try:
            _sync_archive_all_cycles()  # bring the local archive up to date first
            refresh()                    # read fresh data from DB, write forecast log
            persist.push_all_async()     # ship both DBs to the HF Dataset
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
    "[Datadog's Toto 2.0 (22M)](https://huggingface.co/Datadog/Toto-2.0-22m), "
    "compared against the [NWS hourly forecast](https://www.weather.gov/documentation/services-web-api). "
    "The scoreboard tracks who's been more accurate over the past 48 hours."
)

with gr.Blocks(title="Toto Weather Forecast", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Toto on my home weather station")
    gr.Markdown(HOOK)
    gr.Markdown(SUBTITLE)

    hero_md = gr.Markdown()
    gr.Markdown(f"### 📅 {VIEW_WEEK['label']}")
    week_plot = gr.Plot(label="Weekly")
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

    gr.Markdown(
        "<span style='opacity:0.55'>🔄 Live data + forecast auto-refresh every 15 minutes.</span>"
    )

    gr.Markdown("### 🏆 How has each model done so far?")
    scoreboard_md = gr.Markdown()
    residual_plot = gr.Plot(label="Forecast residual")

    with gr.Accordion("How the scoreboard is calculated", open=False):
        gr.Markdown(
            "We score each model on **how close its prediction was to the actual Ecowitt reading** "
            "for the same hour, averaged over the last 48 hours.\n\n"
            "**Picking which forecast counts.** Every refresh logs both models' forecasts for the "
            "next 24-72 hours along with `forecast_made_at` and `target_ts`. For each past target "
            "hour we keep only the **most recent forecast issued *before* that hour** — so neither "
            "model is allowed to peek at data it couldn't have seen at prediction time.\n\n"
            "**The math.** For each metric, per source:\n\n"
            "&nbsp;&nbsp;`abs_err = |p50 − actual|`\n\n"
            "&nbsp;&nbsp;`MAE = mean(abs_err)` over target hours in the last 48 h\n\n"
            "&nbsp;&nbsp;`n` = number of (target hour, source) pairs that had both a forecast and an Ecowitt actual\n\n"
            "The lower MAE wins. NWS doesn't forecast barometric pressure, so the pressure row shows Toto only.\n\n"
            "**What this is NOT.** We score the point prediction (p50) — which throws away Toto's "
            "uncertainty. A scoring rule like CRPS or pinball loss would credit a well-calibrated "
            "10–90% band; MAE doesn't. Folded across all horizons too — Toto's +6 h call and +24 h "
            "call both contribute to the same number. Per-horizon breakdowns are a likely follow-up.\n\n"
            "Full spec: [`docs/toto-inference.md`](https://huggingface.co/spaces/bitsofchris/time-series-ai-weather-forecast/blob/main/docs/toto-inference.md#scoreboard--how-the-accuracy-is-calculated)."
        )

    with gr.Accordion("How the forecast is made", open=False):
        gr.Markdown(
            "**Model.** [Datadog/Toto-2.0-22m](https://huggingface.co/Datadog/Toto-2.0-22m) "
            "(~22 M params, CPU). Second-smallest variant of Toto 2.0; the larger ones (313 M / 1 B / 2.5 B) would tighten the band further.\n\n"
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

    outputs = [hero_md, comparison_md, week_plot, scoreboard_md, residual_plot]
    demo.load(refresh, outputs=outputs)


if __name__ == "__main__":
    persist.pull_all()       # bootstrap forecast log + archive from the HF Dataset
    _sync_5min_only()        # ensure the archive has fresh 5-min data before first paint
    _start_autorefresh()
    demo.launch()
