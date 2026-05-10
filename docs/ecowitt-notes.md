# Ecowitt Cloud API v3 — Notes from Live Calls

Source: live calls against my GW3000B on 2026-05-10. Cross-checked against
[doc.ecowitt.net](https://doc.ecowitt.net/web/#/apiv3en?page_id=1) (SPA — not
fetchable; learnings here come from the actual responses).

## Auth

Every request takes three query params:

- `application_key` — from User Center → Private Center → API Keys
- `api_key` — same place
- `mac` — device MAC, e.g. `6C:C8:40:38:94:DB`

No headers required. HTTPS only.

## Base URL

`https://api.ecowitt.net/api/v3`

## Response envelope

Every response is wrapped:

```json
{ "code": 0, "msg": "success", "time": "1778435131", "data": { ... } }
```

`code != 0` means error; `msg` carries the reason. `time` is server unix seconds (string).

## Endpoints we use

### `GET /device/info`

Sanity check + a real-time snapshot. No `call_back` needed. Returns device
metadata (`name`, `mac`, `date_zone_id`, lat/lon, `stationtype`, `device_status`)
and a `last_update` block whose shape matches the real-time payload below.

### `GET /device/real_time`

Current snapshot. Required: auth + `call_back`. Real-time leaves have shape
`{time: unix_str, unit: str, value: str}` — different from history (see below).

### `GET /device/history`

Historical series. Required params:

- auth (3 keys)
- `start_date`, `end_date` — `YYYY-MM-DD HH:MM:SS`. We send naive UTC; the API
  returns unix seconds that align with UTC, so no tz conversion needed despite
  the device tz being `America/New_York`.
- `cycle_type` — see below
- `call_back` — comma-separated channel list. `all` is **rejected** for
  `/device/history` (`40016 all is invalid`) even though it works for
  `/device/real_time`. Pass explicit channels like
  `outdoor,indoor,pressure,wind,solar_and_uvi,rainfall_piezo,rainfall,battery`.

#### `cycle_type` values + storage tiers

Ecowitt retains data at three resolutions:

| cycle_type | resolution | retention |
|------------|------------|-----------|
| `5min`     | 5 minutes  | last 90 days |
| `30min`    | 30 minutes | last 1 year  |
| `4hour`    | 4 hours    | last 2 years |

`auto` is also accepted and picks resolution based on the requested range.

**Probed empirically** — the docs *suggest* names like `240min`, but those
return `40015 Invalid cycle_type`. The four valid values are `5min`, `30min`,
`4hour`, and `auto`. Anything else is rejected.

#### History response shape

Different from real_time — values are stored in a `list` map keyed by unix seconds:

```json
{
  "data": {
    "outdoor": {
      "temperature": {
        "unit": "ºF",
        "list": {
          "1778277600": "57.1",
          "1778279400": "56.8",
          ...
        }
      },
      "humidity": { "unit": "%", "list": { ... } },
      ...
    },
    "pressure": { "relative": { ... }, "absolute": { ... } },
    "wind":     { "wind_speed": { ... }, ... },
    "outdoor":  { "temperature": { ... }, "humidity": { ... }, ... },
    "indoor":   { ... },
    "solar_and_uvi":  { "solar": { ... }, "uvi": { ... } },
    "rainfall_piezo": { "rain_rate": { ... }, ... },
    "battery":  { "haptic_array_battery": { ... }, ... }
  }
}
```

Every metric leaf is `{ unit: str, list: { unix_seconds_str: value_str } }`. Both
the timestamp keys and the values are strings — cast on the way in.

The structure is consistently 2 levels deep: `data.{channel}.{metric}` for the
metrics we've seen on this station. `pressure` only has `relative` and
`absolute`; everything else has multiple metrics.

## Running the SQLite sync

The archive lives at `data/ecowitt.db` (gitignored). The sync is idempotent:
re-running is safe, and only new rows are written.

```bash
# full update across all three cycle_types (5min, 30min, 4hour)
.venv/bin/python -m src.sync

# just one cycle_type
.venv/bin/python -m src.sync --cycle 5min

# different DB path
.venv/bin/python -m src.sync --db data/other.db

# bigger overlap when resuming (default is 24h)
.venv/bin/python -m src.sync --overlap-hours 48
```

How "incremental" works: for each `cycle_type` the sync queries
`MAX(ts_unix)` from the DB and starts the next fetch at `max_ts − overlap`
(default 1 day). Overlap re-fetches a small tail to catch late-arriving
points; dedup is handled by the primary key
`(cycle_type, channel, metric, ts_unix)` with `INSERT OR REPLACE`.

Suggested cadence:

- **Hourly** for `5min` (most active tier).
- **Daily** for `30min` and `4hour` — cheap and keeps the long-range archive
  current.

Pure-cron example (run hourly, only the active tier; do a full sweep nightly):

```cron
0 * * * *  cd ~/repos/time-series-ai-weather-forecast && .venv/bin/python -m src.sync --cycle 5min  >> data/sync.log 2>&1
30 3 * * * cd ~/repos/time-series-ai-weather-forecast && .venv/bin/python -m src.sync             >> data/sync.log 2>&1
```

Inspect the archive directly with sqlite:

```bash
sqlite3 data/ecowitt.db "SELECT cycle_type, COUNT(*), MIN(datetime(ts_unix,'unixepoch')), MAX(datetime(ts_unix,'unixepoch')) FROM readings GROUP BY cycle_type"
sqlite3 data/ecowitt.db "SELECT * FROM fetch_log ORDER BY id DESC LIMIT 10"
```

If the run aborts on a rate-limit (`code=-1`), the script stops cleanly after
backing off twice — just re-run later and it picks up from where it left off.

## Quirks

- **Strings, not numbers.** Timestamps and values both arrive as strings.
- **Sparse data is possible.** Different metrics may have slightly different
  sets of timestamps (sensors drop in and out).
- **Per-call size.** For `cycle_type=30min` over 7 days with a few channels,
  responses are ~50 KB. `5min` for the same window is ~10× larger. Chunk
  multi-month ranges to be safe.
- **Rate limit (observed).** The API returns `code=-1`,
  `msg="The number of interface accesses reached the upper limit"` after
  ~30 calls in quick succession. The exact threshold and reset window aren't
  documented; in practice a 60–120s sleep clears it. Sync handles this with
  exponential backoff (60s → 120s) and stops gracefully if still throttled
  so a re-run can resume from where it left off.
- **Channel availability is station-specific.** GW3000B exposes `outdoor`,
  `indoor`, `solar_and_uvi`, `rainfall_piezo`, `wind`, `pressure`, `battery`.
  Other stations differ.
- **Units may change.** Returned in whatever unit the device is configured for
  (mine is imperial). Persist `unit` alongside the value so we don't lose it.
