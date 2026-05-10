# Toto Weather Forecasting Demo — Plan

A public Hugging Face Space that pulls live data from a personal Ecowitt GW3000 weather station, runs Datadog's Toto 2.0 (smallest variant) to forecast the next 24h of temperature/humidity/pressure, and shows it next to the National Weather Service forecast.

**Hook:** "Language models predict the next token. What if you could predict the future with the same technology?"

## Architecture (one file, one process)

```
Ecowitt Cloud API v3  ──┐
                        ├──►  app.py (Gradio Blocks)  ──►  HF Space (CPU basic)
NWS API /forecastHourly ┘            │
                                     ├─ Toto 2.0 small (HF Hub, ~4M params, CPU)
                                     ├─ Plotly figs (gr.Plot × 3)
                                     └─ TTL cache (1h) on fetches + inference
```

## Repo layout

```
time-series-ai-weather-forecast/
├── app.py                     # Gradio entry point
├── requirements.txt
├── README.md                  # HF Space frontmatter + description
├── .env.example
├── .gitignore
├── docs/
│   └── plan.md                # this file
└── src/
    ├── ecowitt.py             # Ecowitt API client
    ├── nws.py                 # NWS forecast client
    ├── forecast.py            # Toto load + inference
    ├── plotting.py            # Plotly figure builders
    └── cache.py               # TTL cache decorator
```

Single `app.py` is also fine; splitting into `src/` keeps each concern testable.

## Build order

1. **Ecowitt client** (current focus) — fetch real-time + last 7 days history, return a clean hourly `pandas.DataFrame` with columns `temp_f`, `humidity`, `pressure_inhg`, indexed by UTC timestamp.
2. **NWS client** — `/points/{lat},{lon}` → `forecastHourly` URL → 24h hourly forecast aligned to Ecowitt's cadence.
3. **Toto inference** — load smallest Toto 2.0 from HF Hub, univariate forecast per metric, return median + p10 + p90 over a 24h horizon.
4. **Plotting** — one Plotly figure per metric: past actuals (solid), Toto median (dashed) + p10–p90 band (shaded), NWS forecast (dashed, distinct color), vertical "now" marker.
5. **Gradio app** — `gr.Blocks`, title + hook, "Refresh" button, three `gr.Plot` outputs. `demo.load` runs once on visit; cache prevents repeat inference.
6. **Local smoke test** — `python app.py`, verify all three plots render with real data.
7. **Push to HF** — set secrets in Space settings, watch build, verify public URL.

## Ecowitt API v3 reference (verified URLs)

Base: `https://api.ecowitt.net/api/v3`

- `GET /device/real_time` — current snapshot. Params: `application_key`, `api_key`, `mac`, `call_back=all`.
- `GET /device/history` — historical data. Params: `application_key`, `api_key`, `mac`, `start_date`, `end_date`, `cycle_type`, `call_back`. Cycle types per Ecowitt's storage tiers: `5min` (last 90 days), `30min` (last year), `240min` (last 2 years), `auto`. Date format and exact `call_back` values to be confirmed against the live API on first call.
- `GET /device/info` — sanity check that creds + MAC are valid.

For the demo we want hourly cadence over the last 7 days, so `cycle_type=30min` and we resample to 1h locally. (5min would also work; 30min is lighter.)

## Secrets / config

Local `.env` (gitignored):
```
ECOWITT_APPLICATION_KEY=...
ECOWITT_API_KEY=...
ECOWITT_DEVICE_MAC=...
LAT=...
LON=...
```

On HF Space → Settings → Variables and Secrets:
- Secrets: `ECOWITT_APPLICATION_KEY`, `ECOWITT_API_KEY`
- Variables: `ECOWITT_DEVICE_MAC`, `LAT`, `LON`

## Out of scope

- Multivariate Toto inference
- Fine-tuning
- Auth, rate limiting, monitoring
- Metrics beyond temp/humidity/pressure
