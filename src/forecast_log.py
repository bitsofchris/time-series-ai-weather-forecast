"""SQLite logging of Toto + NWS forecasts and Ecowitt actuals.

Every refresh appends:
  - actuals(target_ts, metric, value)            ← Ecowitt history rows
  - forecast_snapshots(forecast_made_at, target_ts, source, metric, p10, p50, p90)
                                                  ← one row per future hour, per source, per metric

A scoreboard joins the two and computes per-source MAE over a rolling window.

NOTE: On HuggingFace Spaces' free CPU tier the DB lives in ephemeral storage
and resets when the Space rebuilds (i.e. on `git push`). Restarts without
rebuild keep the file. For longer-lived tracking, mount persistent storage
or push to an HF Dataset.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Iterable

import pandas as pd

from .forecast import TotoForecast

DEFAULT_DB_PATH = os.environ.get("FORECAST_LOG_DB", "data/forecasts.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    forecast_made_at INTEGER NOT NULL,
    target_ts        INTEGER NOT NULL,
    source           TEXT    NOT NULL,        -- 'toto' | 'nws'
    metric           TEXT    NOT NULL,        -- 'temp_f' | 'humidity' | 'pressure_inhg'
    p10              REAL,
    p50              REAL,
    p90              REAL,
    PRIMARY KEY (forecast_made_at, target_ts, source, metric)
);
CREATE INDEX IF NOT EXISTS idx_fs_target ON forecast_snapshots(target_ts, metric, source);

CREATE TABLE IF NOT EXISTS actuals (
    target_ts INTEGER NOT NULL,
    metric    TEXT    NOT NULL,
    value     REAL    NOT NULL,
    PRIMARY KEY (target_ts, metric)
);
"""


def connect(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ts(t) -> int:
    return int(pd.Timestamp(t).tz_convert("UTC").timestamp())


def record_actuals(
    conn: sqlite3.Connection,
    history: pd.DataFrame,
    only_hourly: bool = True,
) -> int:
    """Upsert actuals from a history DataFrame (UTC-indexed; column = metric).

    By default only hourly-aligned target_ts are stored so the scoreboard
    table stays small even when the source history is at 5-min cadence.
    """
    rows = []
    for metric in history.columns:
        s = history[metric].dropna()
        for ts, val in s.items():
            tsu = _ts(ts)
            if only_hourly and tsu % 3600 != 0:
                continue
            rows.append((tsu, metric, float(val)))
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO actuals (target_ts, metric, value) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def record_toto(
    conn: sqlite3.Connection,
    metric: str,
    fcst: TotoForecast,
    forecast_made_at: int | None = None,
    only_hourly: bool = True,
) -> int:
    """Persist a Toto forecast.

    `only_hourly`: when True (default), only the hourly-aligned target_ts
    rows are written. Forecast inference may run at 5-min cadence, but the
    scoreboard score is the same regardless of cadence and the log grows
    linearly per refresh — hourly keeps it manageable.
    """
    made = forecast_made_at if forecast_made_at is not None else int(time.time())
    rows = []
    for t, p10, p50, p90 in zip(
        fcst.median.index, fcst.p10.values, fcst.median.values, fcst.p90.values
    ):
        tsu = _ts(t)
        if only_hourly and tsu % 3600 != 0:
            continue
        rows.append((made, tsu, "toto", metric, float(p10), float(p50), float(p90)))
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO forecast_snapshots "
        "(forecast_made_at, target_ts, source, metric, p10, p50, p90) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def record_nws(
    conn: sqlite3.Connection,
    metric: str,
    series: pd.Series,
    forecast_made_at: int | None = None,
) -> int:
    """NWS gives a point forecast only — store as p50 with NULL p10/p90."""
    made = forecast_made_at if forecast_made_at is not None else int(time.time())
    s = series.dropna()
    rows = [(made, _ts(t), "nws", metric, None, float(v), None) for t, v in s.items()]
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO forecast_snapshots "
        "(forecast_made_at, target_ts, source, metric, p10, p50, p90) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def scoreboard(
    conn: sqlite3.Connection,
    metric: str = "temp_f",
    window_hours: int = 48,
) -> pd.DataFrame:
    """Per-source MAE over the last `window_hours`, restricted to forecasts
    whose target time is in the past and where we have an actual value.

    Each (target_ts, source) pair is scored against the *most recent* forecast
    issued for that target — i.e. the latest snapshot before target_ts.
    """
    cutoff = int(time.time()) - window_hours * 3600
    sql = """
    WITH latest AS (
        SELECT source, target_ts, metric,
               MAX(forecast_made_at) AS forecast_made_at
        FROM forecast_snapshots
        WHERE metric = ?
          AND forecast_made_at <= target_ts
          AND target_ts <= ?
          AND target_ts >= ?
        GROUP BY source, target_ts, metric
    )
    SELECT f.source,
           f.target_ts,
           f.p50         AS prediction,
           a.value       AS actual,
           ABS(f.p50 - a.value) AS abs_err
    FROM forecast_snapshots f
    JOIN latest l USING (source, target_ts, metric, forecast_made_at)
    JOIN actuals a USING (target_ts, metric)
    """
    now = int(time.time())
    df = pd.read_sql_query(sql, conn, params=[metric, now, cutoff])
    if df.empty:
        return df
    return df


def historical_predictions(
    conn: sqlite3.Connection,
    source: str,
    metric: str,
    since_unix: int | None = None,
    until_unix: int | None = None,
    lag_hours: float | None = 6.0,
) -> pd.DataFrame:
    """For each target_ts in [since, until], return one historical forecast row.

    Two modes:

    - `lag_hours=None`: legacy 'latest-pre-target' behavior — for each
      target hour, return the most-recent forecast issued before it. This
      mixes different forecast lags depending on autorefresh timing, which
      visually produces a sawtooth on the overlay.

    - `lag_hours=N` (default 6.0): for each target hour, return the
      forecast whose `forecast_made_at` is closest to `target_ts − N
      hours`. Constant lag = consistent prediction difficulty = smooth
      line on the chart. Semantics: 'what did Toto predict for this hour,
      N hours before it happened?'.

    `until_unix` defaults to now and caps the overlay so it never crosses
    into the future side of the chart.
    """
    import time as _time  # noqa: PLC0415
    if until_unix is None:
        until_unix = int(_time.time())

    if lag_hours is None:
        # Original 'latest before target' query.
        params: list = [source, metric, until_unix]
        where_extra = ""
        if since_unix is not None:
            where_extra = " AND target_ts >= ?"
            params.append(since_unix)
        sql = f"""
        WITH latest AS (
            SELECT source, target_ts, metric,
                   MAX(forecast_made_at) AS forecast_made_at
            FROM forecast_snapshots
            WHERE source = ? AND metric = ?
              AND forecast_made_at <= target_ts
              AND target_ts <= ?
              {where_extra}
            GROUP BY source, target_ts, metric
        )
        SELECT f.target_ts, f.p10, f.p50, f.p90
        FROM forecast_snapshots f
        JOIN latest l USING (source, target_ts, metric, forecast_made_at)
        ORDER BY f.target_ts
        """
    else:
        # Fixed-horizon pick: forecast_made_at closest to target_ts − lag.
        lag_seconds = int(lag_hours * 3600)
        params = [lag_seconds, source, metric, until_unix]
        where_extra = ""
        if since_unix is not None:
            where_extra = " AND target_ts >= ?"
            params.append(since_unix)
        sql = f"""
        WITH ranked AS (
            SELECT target_ts, forecast_made_at, p10, p50, p90,
                   ABS(forecast_made_at - (target_ts - ?)) AS lag_err,
                   ROW_NUMBER() OVER (
                       PARTITION BY target_ts
                       ORDER BY ABS(forecast_made_at - (target_ts - ?))
                   ) AS rk
            FROM forecast_snapshots
            WHERE source = ? AND metric = ?
              AND forecast_made_at <= target_ts
              AND target_ts <= ?
              {where_extra}
        )
        SELECT target_ts, p10, p50, p90
        FROM ranked
        WHERE rk = 1
        ORDER BY target_ts
        """
        # The window function references the lag twice — easier to pass it
        # twice than juggle indexes in the prepared statement.
        params.insert(1, lag_seconds)

    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["target_ts"], unit="s", utc=True)
    df = df.drop(columns=["target_ts"])
    return df


def residuals(
    conn: sqlite3.Connection,
    metric: str,
    window_hours: int = 48,
    lag_hours: float = 3.0,
) -> pd.DataFrame:
    """For each hourly target_ts in the last `window_hours`, return each
    model's prediction (picked at a fixed lookahead) and the Ecowitt
    actual side-by-side, plus signed residuals (prediction − actual).
    """
    import time as _time  # noqa: PLC0415
    now = int(_time.time())
    cutoff = now - window_hours * 3600
    lag_seconds = int(lag_hours * 3600)
    sql = """
    WITH ranked AS (
        SELECT source, target_ts, p50,
               ROW_NUMBER() OVER (
                   PARTITION BY source, target_ts
                   ORDER BY ABS(forecast_made_at - (target_ts - ?))
               ) AS rk
        FROM forecast_snapshots
        WHERE metric = ?
          AND forecast_made_at <= target_ts
          AND target_ts BETWEEN ? AND ?
    ),
    picked AS (
        SELECT source, target_ts, p50 FROM ranked WHERE rk = 1
    )
    SELECT a.target_ts,
           MAX(CASE WHEN p.source='toto' THEN p.p50 END) AS toto_p50,
           MAX(CASE WHEN p.source='nws'  THEN p.p50 END) AS nws_p50,
           a.value AS actual
    FROM actuals a
    LEFT JOIN picked p USING (target_ts)
    WHERE a.metric = ?
      AND a.target_ts BETWEEN ? AND ?
    GROUP BY a.target_ts
    ORDER BY a.target_ts
    """
    params = [lag_seconds, metric, cutoff, now, metric, cutoff, now]
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["target_ts"], unit="s", utc=True)
    df = df.drop(columns=["target_ts"])
    df["toto_residual"] = df["toto_p50"] - df["actual"]
    df["nws_residual"] = df["nws_p50"] - df["actual"]
    return df


def scoreboard_summary(
    conn: sqlite3.Connection,
    metric: str = "temp_f",
    window_hours: int = 48,
) -> pd.DataFrame:
    df = scoreboard(conn, metric=metric, window_hours=window_hours)
    if df.empty:
        return pd.DataFrame(columns=["source", "n", "mae"])
    return (
        df.groupby("source")["abs_err"]
        .agg(n="count", mae="mean")
        .reset_index()
    )


def scoreboard_at_lag(
    conn: sqlite3.Connection,
    metric: str,
    lag_hours: float,
    window_hours: int | None = 48,
) -> pd.DataFrame:
    """Per-source MAE at a specific forecast lookahead.

    For each past target hour in the window, pick each source's forecast
    whose `forecast_made_at` is closest to `target_ts - lag_hours`. With
    autorefresh ticking every 15 min that picker selects a forecast within
    ~7-8 min of the requested lag, so the MAE genuinely reflects the
    'how good was the N-hours-ahead prediction?' question.

    `window_hours=None` removes the rolling-window filter — useful for
    'lifetime' MAE since the very first logged snapshot.
    """
    import time as _time  # noqa: PLC0415
    lag_seconds = int(lag_hours * 3600)
    now = int(_time.time())
    where_window = ""
    params: list = [lag_seconds, metric]
    if window_hours is not None:
        cutoff = now - int(window_hours) * 3600
        where_window = " AND target_ts BETWEEN ? AND ?"
        params.extend([cutoff, now])
    else:
        where_window = " AND target_ts <= ?"
        params.append(now)
    params.append(metric)  # for the outer WHERE on actuals
    sql = f"""
    WITH ranked AS (
        SELECT source, target_ts, forecast_made_at, p50,
               ROW_NUMBER() OVER (
                   PARTITION BY source, target_ts
                   ORDER BY ABS(forecast_made_at - (target_ts - ?))
               ) AS rk
        FROM forecast_snapshots
        WHERE metric = ?
          AND forecast_made_at <= target_ts
          {where_window}
    )
    SELECT r.source,
           COUNT(*) AS n,
           AVG(ABS(r.p50 - a.value)) AS mae
    FROM ranked r
    JOIN actuals a USING (target_ts)
    WHERE r.rk = 1 AND a.metric = ?
    GROUP BY r.source
    """
    df = pd.read_sql_query(sql, conn, params=params)
    return df
