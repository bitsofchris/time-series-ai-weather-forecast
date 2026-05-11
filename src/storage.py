"""SQLite archive for Ecowitt history data.

One row per (cycle_type, channel, metric, ts_unix). Idempotent upsert via
INSERT OR REPLACE on the natural key, so re-running with overlapping windows
is safe.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    cycle_type TEXT NOT NULL,
    channel    TEXT NOT NULL,
    metric     TEXT NOT NULL,
    ts_unix    INTEGER NOT NULL,
    value      TEXT,
    unit       TEXT,
    PRIMARY KEY (cycle_type, channel, metric, ts_unix)
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts_unix);
CREATE INDEX IF NOT EXISTS idx_readings_metric ON readings(channel, metric, ts_unix);

CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_type  TEXT NOT NULL,
    start_ts    INTEGER NOT NULL,
    end_ts      INTEGER NOT NULL,
    fetched_at  INTEGER NOT NULL,
    rows_upserted INTEGER NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def iter_history_rows(
    response_data: dict,
    cycle_type: str,
) -> Iterator[tuple[str, str, str, int, str, str]]:
    """Walk a /device/history response's `data` dict and yield rows.

    The response is consistently 2 levels deep: data[channel][metric] = {unit, list}.
    We handle deeper nesting too (channel.subchannel.metric) by recursing until
    we hit a node that has both `unit` and `list`.
    """
    def walk(node, path: list[str]):
        if isinstance(node, dict) and "list" in node and isinstance(node.get("list"), dict):
            unit = node.get("unit", "")
            channel = path[0] if path else ""
            metric = ".".join(path[1:]) if len(path) > 1 else ""
            for ts_str, val in node["list"].items():
                try:
                    ts = int(ts_str)
                except (TypeError, ValueError):
                    continue
                yield (cycle_type, channel, metric, ts, str(val), str(unit))
            return
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, path + [k])

    yield from walk(response_data, [])


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO readings "
        "(cycle_type, channel, metric, ts_unix, value, unit) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def log_fetch(
    conn: sqlite3.Connection,
    cycle_type: str,
    start_ts: int,
    end_ts: int,
    fetched_at: int,
    rows_upserted: int,
) -> None:
    conn.execute(
        "INSERT INTO fetch_log (cycle_type, start_ts, end_ts, fetched_at, rows_upserted)"
        " VALUES (?,?,?,?,?)",
        (cycle_type, start_ts, end_ts, fetched_at, rows_upserted),
    )
    conn.commit()


def max_ts(conn: sqlite3.Connection, cycle_type: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts_unix) FROM readings WHERE cycle_type = ?", (cycle_type,)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def read_history_dataframe(
    conn: sqlite3.Connection,
    since_unix: int,
    until_unix: int | None = None,
    cycle_type: str = "5min",
    fields: dict[str, tuple[str, str]] | None = None,
    resample: str | None = None,
):
    """Read a multi-metric history slice from the local archive.

    Returns a UTC-indexed pandas DataFrame whose columns are the keys of
    `fields` (default: ecowitt.HISTORY_FIELDS) — temp_f, humidity,
    pressure_inhg, rain_in_hr. Each column is pulled from the readings
    table at the requested `cycle_type`; optionally resampled to a uniform
    cadence with `.mean()`.
    """
    import time as _time
    import pandas as pd  # local import keeps storage importable without pandas

    if fields is None:
        from . import ecowitt
        fields = ecowitt.HISTORY_FIELDS
    if until_unix is None:
        until_unix = int(_time.time())

    series_dict: dict[str, pd.Series] = {}
    for col, (channel, metric) in fields.items():
        rows = conn.execute(
            "SELECT ts_unix, value FROM readings"
            " WHERE cycle_type=? AND channel=? AND metric=?"
            "   AND ts_unix BETWEEN ? AND ?"
            " ORDER BY ts_unix",
            (cycle_type, channel, metric, since_unix, until_unix),
        ).fetchall()
        if not rows:
            continue
        idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
        vals = pd.to_numeric([r[1] for r in rows], errors="coerce")
        series_dict[col] = pd.Series(vals, index=idx, name=col).sort_index()

    if not series_dict:
        return pd.DataFrame()
    df = pd.concat(series_dict.values(), axis=1)
    df.columns = list(series_dict.keys())
    if resample:
        df = df.resample(resample).mean()
    return df


def stats(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT cycle_type, COUNT(*), MIN(ts_unix), MAX(ts_unix),"
        " COUNT(DISTINCT channel || '.' || metric)"
        " FROM readings GROUP BY cycle_type ORDER BY cycle_type"
    ).fetchall()
