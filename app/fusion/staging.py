"""Step 3 — deep / REM / light staging heuristics (handoff doc §7.3).

Applied only to epochs already classified asleep (steps 1+2). With a BPM-only
sensor and no training, these are proxies: deep/light is a reasonable split,
REM is explicitly best-effort.

Stage codes (module-wide): 0=AWAKE, 1=LIGHT, 2=DEEP, 3=REM.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from .preprocess import Series, moving_average

AWAKE, LIGHT, DEEP, REM = 0, 1, 2, 3


def _hr_variability(series: Series, window: int) -> np.ndarray:
    """Epoch-to-epoch HR variability: rolling std of |diff| of filled HR."""
    diff = np.abs(np.diff(series.hr_filled, prepend=series.hr_filled[:1]))
    return moving_average(diff, window)


def stage_asleep(
    series: Series,
    asleep: np.ndarray,
    period: tuple[int, int],
    params: Params,
) -> np.ndarray:
    """Return an int stage-code array for the whole timeline.

    `asleep` is the post-actigraphy boolean (True=asleep). Epochs outside the
    sleep period, or awake within it, are AWAKE.
    """
    s = params.staging
    onset, offset = period
    n = series.n
    stages = np.full(n, AWAKE, dtype=np.int64)

    in_period = np.zeros(n, dtype=bool)
    in_period[onset:offset] = True
    asleep_here = asleep & in_period
    if not asleep_here.any():
        return stages

    # Everything asleep starts as LIGHT; deep/REM carve out from there.
    stages[asleep_here] = LIGHT

    # Night HR floor measured within the sleep period's valid readings.
    period_hr = series.hr[onset:offset]
    valid = period_hr[np.isfinite(period_hr)]
    hr_floor = float(np.percentile(valid, s.hr_floor_percentile)) if valid.size else 0.0

    hrv = _hr_variability(series, s.hrv_window_epochs)
    sustained_low_act = moving_average(series.act, s.deep_window_epochs)

    # --- DEEP: HR near floor + sustained low movement + low HR variability ---
    deep_mask = (
        asleep_here
        & (series.hr_filled <= hr_floor + s.deep_hr_margin_bpm)
        & (sustained_low_act <= s.deep_act_max)
        & (hrv <= s.deep_hrv_max)
    )
    stages[deep_mask] = DEEP

    # --- REM: low movement + HR above the deep floor + elevated variability,
    # blocked from the early part of the sleep period (weight toward later). ---
    length = max(1, offset - onset)
    pos = (np.arange(n) - onset) / length  # fractional position within period
    rem_allowed = pos >= s.rem_earliest_fraction
    rem_mask = (
        asleep_here
        & (series.act <= s.rem_act_max)
        & (series.hr_filled >= hr_floor + s.rem_hr_floor_bpm)
        & (hrv >= s.rem_hrv_min)
        & rem_allowed
    )
    # REM takes precedence over an earlier DEEP label only where deep failed;
    # since deep requires low HRV and REM requires high HRV they rarely overlap,
    # but resolve any tie toward REM's stronger HR/variability signal.
    stages[rem_mask] = REM

    return stages
