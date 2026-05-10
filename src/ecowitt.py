"""Ecowitt Cloud API v3 client.

Docs: https://doc.ecowitt.net/web/#/apiv3en?page_id=1

Endpoints used:
  GET /device/info       — credential + MAC sanity check
  GET /device/real_time  — current snapshot
  GET /device/history    — historical series

Run standalone to verify creds and inspect raw responses:
  python -m src.ecowitt info
  python -m src.ecowitt real_time
  python -m src.ecowitt history --days 7 --cycle 30min
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BASE_URL = "https://api.ecowitt.net/api/v3"

# Channels we care about for the forecast demo, mapped to a flat column name.
# Path is (channel, metric) into the response `data` dict.
HISTORY_FIELDS: dict[str, tuple[str, str]] = {
    "temp_f": ("outdoor", "temperature"),
    "humidity": ("outdoor", "humidity"),
    "pressure_inhg": ("pressure", "relative"),
}

# Outdoor temp/humidity live under common.outdoor; pressure under pressure.
# call_back is a comma-separated list of channels to return.
DEFAULT_CALL_BACK = "outdoor,indoor,pressure"


@dataclass
class EcowittConfig:
    application_key: str
    api_key: str
    mac: str

    @classmethod
    def from_env(cls) -> "EcowittConfig":
        missing = [
            k for k in ("ECOWITT_APPLICATION_KEY", "ECOWITT_API_KEY", "ECOWITT_DEVICE_MAC")
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
        return cls(
            application_key=os.environ["ECOWITT_APPLICATION_KEY"],
            api_key=os.environ["ECOWITT_API_KEY"],
            mac=os.environ["ECOWITT_DEVICE_MAC"],
        )

    def auth_params(self) -> dict:
        return {
            "application_key": self.application_key,
            "api_key": self.api_key,
            "mac": self.mac,
        }


class EcowittAPIError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"Ecowitt API error: code={code} msg={msg}")
        self.code = code
        self.msg = msg


class EcowittRateLimitError(EcowittAPIError):
    pass


def _get(path: str, params: dict, timeout: int = 30) -> dict:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    # Ecowitt wraps responses as {"code": 0, "msg": "success", "time": "...", "data": {...}}
    code = body.get("code")
    if code in (0, "0"):
        return body
    msg = body.get("msg", "")
    # code=-1 with "upper limit" wording is the per-account rate limit.
    if str(code) == "-1" and "upper limit" in str(msg).lower():
        raise EcowittRateLimitError(code, msg)
    raise EcowittAPIError(code, msg)


def fetch_info(cfg: EcowittConfig) -> dict:
    return _get("/device/info", cfg.auth_params())


def fetch_real_time(cfg: EcowittConfig, call_back: str = DEFAULT_CALL_BACK) -> dict:
    params = {**cfg.auth_params(), "call_back": call_back}
    return _get("/device/real_time", params)


def fetch_history(
    cfg: EcowittConfig,
    start: datetime,
    end: datetime,
    cycle_type: str = "30min",
    call_back: str = DEFAULT_CALL_BACK,
) -> dict:
    """Fetch history between [start, end].

    cycle_type valid values per Ecowitt storage tiers: 5min, 30min, 240min, auto.
    Date format expected by API: 'YYYY-MM-DD HH:mm:ss' in the device's local time.
    """
    params = {
        **cfg.auth_params(),
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_type": cycle_type,
        "call_back": call_back,
    }
    return _get("/device/history", params)


def history_to_dataframe(
    response: dict,
    fields: dict[str, tuple[str, str]] | None = None,
    resample: str | None = "1h",
) -> pd.DataFrame:
    """Flatten an Ecowitt history response into a UTC-indexed DataFrame.

    The history payload looks like:
        data[channel][metric] = {"unit": str, "list": {unix_str: value_str, ...}}

    Returns a DataFrame with one column per entry in `fields`, indexed by UTC
    timestamp, optionally resampled (default hourly mean) for stable input to
    Toto.
    """
    if fields is None:
        fields = HISTORY_FIELDS
    data = response["data"]
    series: dict[str, pd.Series] = {}
    for col, (channel, metric) in fields.items():
        node = data.get(channel, {}).get(metric)
        if not node or "list" not in node:
            raise KeyError(f"Missing {channel}.{metric} in Ecowitt history response")
        items = node["list"]
        idx = pd.to_datetime([int(t) for t in items.keys()], unit="s", utc=True)
        vals = pd.to_numeric(list(items.values()), errors="coerce")
        series[col] = pd.Series(vals, index=idx, name=col).sort_index()
    df = pd.concat(series.values(), axis=1)
    df.columns = list(series.keys())
    if resample:
        df = df.resample(resample).mean()
    return df


def _load_dotenv_if_present() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _main(argv: list[str]) -> int:
    _load_dotenv_if_present()
    cfg = EcowittConfig.from_env()
    cmd = argv[0] if argv else "info"
    if cmd == "info":
        out = fetch_info(cfg)
    elif cmd == "real_time":
        out = fetch_real_time(cfg)
    elif cmd == "history":
        days = 7
        cycle = "30min"
        for i, a in enumerate(argv):
            if a == "--days" and i + 1 < len(argv):
                days = int(argv[i + 1])
            if a == "--cycle" and i + 1 < len(argv):
                cycle = argv[i + 1]
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=days)
        out = fetch_history(cfg, start, end, cycle_type=cycle)
        if "--df" in argv:
            df = history_to_dataframe(out)
            print(df.tail(24).to_string())
            print(f"\nshape: {df.shape}, range: {df.index.min()} → {df.index.max()}")
            return 0
    else:
        print(f"Unknown command: {cmd}. Use info | real_time | history", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
