"""Incremental Ecowitt → SQLite archive.

For each cycle_type we want a complete archive of what the API still has:

  cycle_type   resolution   API retention
  5min         5  min       last 90  days
  30min        30 min       last 365 days
  4hour        4  hours     last 730 days

On each run we figure out the start of the window we need to fetch:

  - First run (or empty table): start = now - retention.
  - Subsequent runs: start = max(ts_in_db) - overlap (default 1 day) so we
    re-fetch a small tail to catch any late-arriving values.

Then we walk that window in chunks (per-cycle chunk size below) so no single
request gets pathologically large, and upsert each chunk.

Run:

  python -m src.sync                   # full update across all cycle_types
  python -m src.sync --cycle 30min     # just one cycle_type
  python -m src.sync --db data/ecowitt.db
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import ecowitt, storage

# Per-cycle config: how far back the API keeps data, and how big a chunk to
# request at once. Chunk sizes are conservative — we'd rather make a few extra
# calls than have one fail or get truncated.
@dataclass(frozen=True)
class CycleConfig:
    name: str
    retention: timedelta
    chunk: timedelta


CYCLES: list[CycleConfig] = [
    CycleConfig("5min",  timedelta(days=90),  timedelta(days=7)),
    CycleConfig("30min", timedelta(days=365), timedelta(days=30)),
    CycleConfig("4hour", timedelta(days=730), timedelta(days=90)),
]

DEFAULT_OVERLAP = timedelta(days=1)
DEFAULT_DB_PATH = "data/ecowitt.db"
# /device/history rejects call_back=all (40016). Pass explicit channels.
# This list covers everything a GW3000B exposes; extras are silently ignored.
CALL_BACK = "outdoor,indoor,solar_and_uvi,rainfall_piezo,rainfall,wind,pressure,battery"


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fetch_with_retry(cfg, s, e, cycle_name, *, retries=2, base_sleep=60, verbose=True):
    """Wrap ecowitt.fetch_history with one or two backoff retries on rate-limit."""
    for attempt in range(retries + 1):
        try:
            return ecowitt.fetch_history(cfg, s, e, cycle_type=cycle_name, call_back=CALL_BACK)
        except ecowitt.EcowittRateLimitError:
            if attempt == retries:
                raise
            sleep_s = base_sleep * (2 ** attempt)
            if verbose:
                print(f"[{cycle_name}]   rate-limited; sleeping {sleep_s}s and retrying ({attempt+1}/{retries})")
            time.sleep(sleep_s)


def _chunks(start: datetime, end: datetime, size: timedelta):
    cur = start
    while cur < end:
        nxt = min(cur + size, end)
        yield cur, nxt
        cur = nxt


def sync_cycle(
    cfg: ecowitt.EcowittConfig,
    conn,
    cycle: CycleConfig,
    overlap: timedelta = DEFAULT_OVERLAP,
    *,
    verbose: bool = True,
) -> int:
    now = _utcnow_naive()
    earliest_available = now - cycle.retention

    last_ts = storage.max_ts(conn, cycle.name)
    if last_ts is None:
        start = earliest_available
        reason = "first run"
    else:
        last_dt = datetime.utcfromtimestamp(last_ts)
        start = max(earliest_available, last_dt - overlap)
        reason = f"resume from {last_dt.isoformat()}Z (−{overlap})"

    if verbose:
        print(f"[{cycle.name}] {reason}: {start.isoformat()}Z → {now.isoformat()}Z")

    total_rows = 0

    for s, e in _chunks(start, now, cycle.chunk):
        try:
            resp = _fetch_with_retry(cfg, s, e, cycle.name, verbose=verbose)
        except ecowitt.EcowittRateLimitError as err:
            if verbose:
                print(f"[{cycle.name}]   rate limit hit, stopping early: {err}")
                print(f"[{cycle.name}]   re-run later to resume from {s.date()}")
            break
        rows = list(storage.iter_history_rows(resp.get("data") or {}, cycle.name))
        n = storage.upsert_rows(conn, rows)
        total_rows += n
        storage.log_fetch(
            conn,
            cycle.name,
            int(s.replace(tzinfo=timezone.utc).timestamp()),
            int(e.replace(tzinfo=timezone.utc).timestamp()),
            int(time.time()),
            n,
        )
        if verbose:
            print(f"[{cycle.name}]   {s.date()} → {e.date()}: upserted {n} rows")
        # Be nice to the API.
        time.sleep(0.5)

    if verbose:
        print(f"[{cycle.name}] done. total upserted this run: {total_rows}")
    return total_rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--cycle", choices=[c.name for c in CYCLES], help="Run only this cycle_type")
    parser.add_argument("--overlap-hours", type=int, default=24)
    args = parser.parse_args(argv)

    ecowitt._load_dotenv_if_present()
    cfg = ecowitt.EcowittConfig.from_env()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = storage.connect(args.db)

    cycles = [c for c in CYCLES if (args.cycle is None or c.name == args.cycle)]
    overlap = timedelta(hours=args.overlap_hours)

    grand_total = 0
    for cycle in cycles:
        grand_total += sync_cycle(cfg, conn, cycle, overlap=overlap)

    print("\n=== summary ===")
    for ct, count, mn, mx, distinct in storage.stats(conn):
        mn_s = datetime.utcfromtimestamp(mn).isoformat() + "Z" if mn else "-"
        mx_s = datetime.utcfromtimestamp(mx).isoformat() + "Z" if mx else "-"
        print(f"  {ct:>6}: {count:>8} rows, {distinct} metrics, {mn_s} → {mx_s}")
    print(f"  upserted this run: {grand_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
