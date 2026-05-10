# Toto inference: how the demo makes its forecasts

This is a precise spec of every Toto-related knob used by `app.py` so the
post / footnote can quote it accurately.

## Model

| | |
|---|---|
| Model ID | `Datadog/Toto-2.0-4m` |
| Parameters | ~4 M |
| Source | https://huggingface.co/Datadog/Toto-2.0-4m |
| Loaded via | `Toto2Model.from_pretrained(...)` (the `toto-2` package from DataDog/toto's `toto2/` subdir, pinned in `requirements.txt`) |
| Hardware | CPU (HF Space free tier — no GPU) |
| Patch size | `model.config.patch_size = 32` (constraint: context length must be a multiple of 32 or padded to one) |

We picked the smallest variant intentionally so the post can show the
weakest model in the Toto-2.0 family doing something useful zero-shot; the
22 M / 313 M / 1 B variants would tighten the confidence band considerably.

## Input data

| | |
|---|---|
| Source | Ecowitt Cloud API v3 (`/device/history`) |
| Station | Ecowitt GW3000B, Westhampton Beach NY |
| Channels forecasted | `outdoor.temperature` (°F), `outdoor.humidity` (%), `pressure.relative` (inHg), `rainfall_piezo.rain_rate` (in/hr) |
| Native upload cadence | 30 min (the device uploads to Ecowitt's cloud every half hour) |
| `cycle_type` requested | depends on the **Display cadence** dropdown — `30min` (default, resampled to 1 h or 30 min) or `4hour` |
| History window pulled | 7 days for `30min` cycle, 30 days for `4hour` cycle |
| Resampling | pandas `df.resample(R).mean()` where R matches the dropdown (`1h` / `30min` / `4h`) |
| Cleaning | `Series.interpolate(limit_direction="both")` fills resample gaps before the tensor goes to Toto |
| NWS comparison | `https://api.weather.gov/points/{lat},{lon}` → `forecastHourly` (point forecast, no distribution) |

## Context length

Toto requires the time axis of the input tensor to be a multiple of `patch_size = 32`.

```text
n_raw = len(history_series_after_resample_and_interpolate)
if n_raw >= 32:
    n_ctx = (n_raw // 32) * 32                # truncate oldest points
    target_mask = ones(n_ctx)                  # all valid
else:
    n_ctx = 32                                 # pad up to one patch
    pad = 32 - n_raw
    target = [first_value]*pad + raw           # left-pad with the first value
    target_mask = [False]*pad + [True]*n_raw   # tell Toto to ignore the padded steps
```

With ~10 days of station history and an hourly resample, this gives a
context of ~160 hourly points (5 patches). On the `4hour` cycle the
station's short history means we often hit the pad path.

## Tensor shape

```text
target:      torch.float32, shape (batch=1, n_variates=1, time=n_ctx)
target_mask: torch.bool,    shape (batch=1, n_variates=1, time=n_ctx)
series_ids:  torch.long,    shape (batch=1, n_variates=1)            (all zeros — univariate)
```

We forecast each metric **independently** (univariate). Multivariate
inference is a follow-up; the inference cost is comparable but the chart
gets noisier and the post hook is easier to read one metric at a time.

## Prediction length

`horizon_steps = round(horizon_hours / step_hours)` where:

| Display cadence | `step_hours` |
|---|---|
| Hourly | 1.0 |
| 30-min | 0.5 |
| 4-hour | 4.0 |

| Horizon dropdown | `horizon_hours` |
|---|---|
| 24 h (default) | 24 |
| 48 h | 48 |
| 72 h | 72 |

So default: 24 hourly steps. 4-hour cycle × 72 h = 18 steps (smallest). 30-min × 72 h = 144 steps (largest typical).

## Distribution → quantiles

We do **not** Monte-Carlo sample. Toto's output head is a parametric
Student-t mixture (see the Toto 2.0 paper), and `model.forecast()`
returns analytical quantiles directly:

```python
quantiles = model.forecast(
    {"target": target, "target_mask": target_mask, "series_ids": series_ids},
    horizon=horizon_steps,
)
# quantiles shape: (9, batch=1, n_variates=1, horizon)
# quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

We pluck three of them for the chart and the scoreboard:

| Display | Index | Quantile |
|---|---|---|
| Lower edge of the shaded band | `quantiles[0, 0, 0]` | p10 |
| Toto median line | `quantiles[4, 0, 0]` | p50 |
| Upper edge of the shaded band | `quantiles[8, 0, 0]` | p90 |

The shaded band is therefore the **80 % central interval** (p10–p90).
The "±X°F at +24 h" chip on the hero is half of `(p90 − p10)` at the
last forecast step.

## Inference cadence

| | |
|---|---|
| Trigger | a daemon thread inside the Space (`_autorefresh_loop`) and `demo.load` on a visitor's first request |
| Interval | every 15 minutes |
| Cache TTL | 14 minutes — slightly less than the autorefresh interval so the next tick always misses the cache and refetches Ecowitt + NWS |
| Per-tick cost | one `/device/real_time` + one `/device/history` per cycle_type touched + one NWS `/points` + one NWS `forecastHourly` + four univariate Toto forwards (one per metric) |
| CPU forward time | ~hundreds of milliseconds per metric on the free CPU tier; total wallclock per refresh is dominated by the network calls, not the model |

## Persistence

Every refresh writes to `data/forecasts.db` (SQLite):

- `forecast_snapshots(forecast_made_at, target_ts, source, metric, p10, p50, p90)`
- `actuals(target_ts, metric, value)`

`source ∈ {toto, nws}`. NWS rows store the point forecast in `p50` and
leave `p10/p90` NULL.

A second SQLite, `data/ecowitt.db`, is the all-channel raw archive
(populated by `src/sync.py`). Both DBs are pushed to a private HF Dataset
(`bitsofchris/toto-weather-forecast-log`) on every autorefresh tick so
they survive Space rebuilds.

## Scoreboard

For each (target_ts, source, metric) we keep the **most recent** forecast
issued *before* that target hour, join against `actuals`, and report:

```text
MAE_source = mean(|p50 − actual|)  over last 48 h
```

The "Past forecasts" overlay on the chart uses the same query so the
scoreboard number and the chart line refer to identical predictions.
