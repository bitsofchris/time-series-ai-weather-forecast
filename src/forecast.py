"""Toto 2.0 inference wrapper.

We use the smallest Toto 2.0 variant (4M params) for speed on CPU. The model
is downloaded from the HuggingFace Hub on first use and cached.

API confirmed against DataDog/toto's `toto2/notebooks/quick_start.ipynb`:

    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained("Datadog/Toto-2.0-4m", map_location=device)
    quantiles = model.forecast(
        {"target": ..., "target_mask": ..., "series_ids": ...},
        horizon=H,
    )
    # quantiles shape: (9, batch, n_var, horizon)
    # quantile levels:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MODEL_ID = "Datadog/Toto-2.0-22m"

# Index into the 9-quantile output.
Q10_IDX = 0
Q50_IDX = 4
Q90_IDX = 8


@dataclass
class TotoForecast:
    """One metric's forecast.

    `index` is a future-timestamp DatetimeIndex; `median`, `p10`, `p90` are
    pandas Series aligned to it.
    """
    median: pd.Series
    p10: pd.Series
    p90: pd.Series


_MODEL_CACHE: dict[str, object] = {}


def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
    """Lazy-load + cache the Toto model. Imports torch lazily so this module
    is importable in environments without torch (local dev on Intel mac)."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    import torch  # noqa: PLC0415
    from toto2 import Toto2Model  # noqa: PLC0415

    actual_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = Toto2Model.from_pretrained(model_id, map_location=actual_device)
    model = model.to(actual_device).eval()
    _MODEL_CACHE[model_id] = model
    return model


def _series_freq(series: pd.Series) -> pd.Timedelta:
    """Infer the spacing of a regular time series; default to 1 hour."""
    if len(series.index) < 2:
        return pd.Timedelta("1h")
    diffs = pd.Series(series.index).diff().dropna()
    if diffs.empty:
        return pd.Timedelta("1h")
    return diffs.median()


def forecast_series(
    series: pd.Series,
    horizon: int = 24,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> TotoForecast:
    """Univariate forecast for one metric.

    `series` must be regularly-spaced and have a DatetimeIndex (UTC). Returns
    median, p10, p90 over `horizon` future steps at the same cadence.
    """
    import torch  # noqa: PLC0415

    if series.empty:
        raise ValueError("Cannot forecast an empty series")

    import numpy as np  # noqa: PLC0415

    clean = series.astype(float).interpolate(limit_direction="both")

    # Toto requires the context length to be a multiple of the model's
    # patch_size (32 for Toto-2.0-4m). If we have at least one full patch,
    # truncate the oldest points to fit. If we have fewer, left-pad with the
    # first value and mark the padded region False in the mask so Toto
    # ignores it.
    model = load_model(model_id, device=device)
    patch = int(model.config.patch_size)
    raw = clean.to_numpy(dtype=np.float32)
    n_raw = len(raw)

    if n_raw >= patch:
        n = (n_raw // patch) * patch
        arr = raw[-n:]
        mask_vec = np.ones(n, dtype=bool)
    else:
        n = patch
        pad = n - n_raw
        arr = np.concatenate([np.full(pad, raw[0], dtype=np.float32), raw])
        mask_vec = np.concatenate([np.zeros(pad, dtype=bool), np.ones(n_raw, dtype=bool)])

    target = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    target_mask = torch.from_numpy(mask_vec).unsqueeze(0).unsqueeze(0)
    series_ids = torch.zeros(1, 1, dtype=torch.long)

    target = target.to(device)
    target_mask = target_mask.to(device)
    series_ids = series_ids.to(device)

    with torch.no_grad():
        quantiles = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=horizon,
        )
    # quantiles: (9, 1, 1, horizon) → grab three quantile slices
    q = quantiles.detach().cpu().numpy()
    p10 = q[Q10_IDX, 0, 0]
    p50 = q[Q50_IDX, 0, 0]
    p90 = q[Q90_IDX, 0, 0]

    freq = _series_freq(clean)
    last_ts = clean.index[-1]
    future_idx = pd.date_range(start=last_ts + freq, periods=horizon, freq=freq, tz=last_ts.tz)

    return TotoForecast(
        median=pd.Series(p50, index=future_idx, name=f"{series.name}_median"),
        p10=pd.Series(p10, index=future_idx, name=f"{series.name}_p10"),
        p90=pd.Series(p90, index=future_idx, name=f"{series.name}_p90"),
    )
