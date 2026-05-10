"""National Weather Service forecast client.

Two-step flow per https://www.weather.gov/documentation/services-web-api:

  GET /points/{lat},{lon}             → properties.forecastHourly (URL)
  GET <forecastHourly URL>            → properties.periods[]      (hourly forecast)

A `User-Agent` header is required; NWS uses it as a contact string and may
block requests without one. No auth, no API key.

Run standalone:

  python -m src.nws                   # uses LAT / LON from .env
  python -m src.nws --hours 24
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from . import ecowitt  # for _load_dotenv_if_present

# NWS asks for a contact string. Update if you fork.
USER_AGENT = "toto-weather-demo/0.1 (https://huggingface.co/spaces; lettieri.christopher@gmail.com)"

POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"


def _get(url: str, timeout: int = 30) -> dict[str, Any]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_forecast_hourly_url(lat: float, lon: float) -> str:
    """First leg: resolve the forecast grid for this lat/lon."""
    body = _get(POINTS_URL.format(lat=lat, lon=lon))
    url = body.get("properties", {}).get("forecastHourly")
    if not url:
        raise RuntimeError(f"No forecastHourly URL in /points response: {body}")
    return url


def fetch_hourly_periods(forecast_hourly_url: str) -> list[dict]:
    body = _get(forecast_hourly_url)
    return body.get("properties", {}).get("periods", []) or []


def _f_from_period(p: dict) -> float | None:
    """Return temperature in °F regardless of how NWS reports it."""
    val = p.get("temperature")
    if val is None:
        return None
    unit = (p.get("temperatureUnit") or "").upper()
    if unit == "F":
        return float(val)
    if unit == "C":
        return float(val) * 9.0 / 5.0 + 32.0
    return float(val)


def _quantity_value(node: dict | None) -> float | None:
    """NWS quantity nodes look like {'unitCode': 'wmoUnit:percent', 'value': 65}."""
    if not isinstance(node, dict):
        return None
    v = node.get("value")
    return None if v is None else float(v)


def hourly_forecast_df(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    """Return a UTC-indexed DataFrame with NWS forecast columns aligned to
    Ecowitt's column names where possible (`temp_f`, `humidity`)."""
    url = fetch_forecast_hourly_url(lat, lon)
    periods = fetch_hourly_periods(url)
    if not periods:
        return pd.DataFrame()

    rows = []
    for p in periods[:hours]:
        # startTime is ISO-8601 with offset, e.g. "2026-05-10T14:00:00-04:00"
        ts = pd.to_datetime(p["startTime"], utc=True)
        rows.append(
            {
                "ts": ts,
                "temp_f": _f_from_period(p),
                "humidity": _quantity_value(p.get("relativeHumidity")),
                "dewpoint_c": _quantity_value(p.get("dewpoint")),
                "precip_prob": _quantity_value(p.get("probabilityOfPrecipitation")),
                "wind_speed": p.get("windSpeed"),
                "wind_direction": p.get("windDirection"),
                "short_forecast": p.get("shortForecast"),
            }
        )
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args(argv)

    ecowitt._load_dotenv_if_present()
    lat = float(os.environ["LAT"])
    lon = float(os.environ["LON"])

    df = hourly_forecast_df(lat, lon, hours=args.hours)
    print(df.to_string())
    print(f"\nshape: {df.shape}, range: {df.index.min()} → {df.index.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
