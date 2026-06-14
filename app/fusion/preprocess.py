"""Turn the sparse epoch list into dense, aligned numpy arrays.

The watch logs one record per epoch *that it captured*, but epochs can be
missing (BT drop, no HR reading, gaps). We expand to a dense timeline indexed
0..max_t so every downstream step can assume uniform spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Params
from ..models import Epoch


@dataclass
class Series:
    """Dense, aligned per-epoch arrays. Length n = number of epochs."""

    n: int
    act_raw: np.ndarray      # int activity counts, 0 where epoch was missing
    act: np.ndarray          # float activity counts rescaled by count_scale
    hr: np.ndarray           # float bpm, np.nan where no reading
    hr_filled: np.ndarray    # hr with gaps linearly interpolated (for smoothing)
    present: np.ndarray      # bool: was an epoch record actually logged here


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs; hold endpoints flat. All-NaN -> zeros."""
    out = x.copy()
    nans = np.isnan(out)
    if nans.all():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out


def build_series(epochs: list[Epoch], params: Params) -> Series:
    if not epochs:
        raise ValueError("no epochs")
    n = max(e.t for e in epochs) + 1

    act_raw = np.zeros(n, dtype=np.int64)
    hr = np.full(n, np.nan, dtype=np.float64)
    present = np.zeros(n, dtype=bool)

    for e in epochs:
        act_raw[e.t] = e.act
        present[e.t] = True
        if e.hr > 0:
            hr[e.t] = float(e.hr)

    # Subtract the per-night noise floor, then rescale movement excess. Floor is
    # estimated from epochs that actually logged a record (missing epochs are 0
    # and would drag the percentile down).
    logged = act_raw[present].astype(np.float64)
    floor = float(np.percentile(logged, params.actigraphy.floor_percentile)) if logged.size else 0.0
    movement = np.clip(act_raw.astype(np.float64) - floor, 0.0, None)
    act = movement * params.actigraphy.count_scale
    return Series(
        n=n,
        act_raw=act_raw,
        act=act,
        hr=hr,
        hr_filled=_interp_nan(hr),
        present=present,
    )


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; window clamped to >=1, edges shrink."""
    window = max(1, int(window))
    if window == 1:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    # 'same' keeps length; normalise edges by the actual count of summed taps.
    counts = np.convolve(np.ones_like(x, dtype=np.float64), kernel * window, mode="same")
    summed = np.convolve(x.astype(np.float64), kernel * window, mode="same")
    return summed / np.maximum(counts, 1.0)
