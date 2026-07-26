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


def _sustained_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """True only where `mask` holds in a run of at least `min_len` epochs."""
    out = np.zeros_like(mask)
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def stage_asleep(
    series: Series,
    asleep: np.ndarray,
    period: tuple[int, int],
    params: Params,
) -> np.ndarray:
    """Return an int stage-code array for the whole timeline.

    `asleep` is the post-actigraphy boolean (True=asleep). Epochs outside the
    sleep period, or awake within it, are AWAKE.

    Deep/REM are carved from the low/high tails of *this night's own* asleep HR
    distribution (relative percentiles), not absolute bpm margins: resting HR and
    its dynamic range vary enough night to night that fixed margins detected all
    deep on one night and none on the next. Percentiles guarantee a plausible
    stage split every night at the cost of exact per-epoch accuracy — honest for
    a BPM-only sensor with no training.

    The tails are applied with HYSTERESIS: a stage is *entered* at a hard
    percentile but *held* across a looser exit percentile, scanned sequentially.
    A single-cutoff classifier flips LIGHT<->REM every time the smoothed HR
    wobbles across the percentile line (minute-scale chatter); the enter/exit
    band means only a decisive move ends a bout, consolidating REM/Deep into
    realistic multi-minute blocks.
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

    # Night HR median measured within the sleep period's valid readings.
    period_hr = series.hr[onset:offset]
    valid = period_hr[np.isfinite(period_hr)]
    hr_median = float(np.median(valid)) if valid.size else 0.0

    hr_smooth = moving_average(series.hr_filled, s.hrv_window_epochs)

    # --- WAKE-BY-HR: quiet-but-roused wake is invisible to actigraphy; a
    # sustained run of smoothed HR well above the session median recovers it. ---
    if valid.size:
        elevated = hr_smooth > hr_median + s.wake_hr_margin_bpm
        asleep_here &= ~_sustained_runs(elevated, s.wake_hr_min_epochs)
    if not asleep_here.any():
        return stages

    # Everything asleep starts as LIGHT; deep/REM carve out from there.
    stages[asleep_here] = LIGHT

    hrv = _hr_variability(series, s.hrv_window_epochs)
    sustained_low_act = moving_average(series.act, s.deep_window_epochs)

    # Per-night HR band edges: percentiles of the *asleep* smoothed HR only, so
    # awake spikes don't skew them. Deep = low tail, REM = high tail. Enter is the
    # hard edge; exit is looser (deep exit above enter, rem exit below enter), so
    # a bout persists across HR wobble near the enter line.
    asleep_hr = hr_smooth[asleep_here]
    deep_enter_thr = float(np.percentile(asleep_hr, s.deep_enter_percentile))
    deep_exit_thr = float(np.percentile(asleep_hr, s.deep_exit_percentile))
    rem_enter_thr = float(np.percentile(asleep_hr, s.rem_enter_percentile))
    rem_exit_thr = float(np.percentile(asleep_hr, s.rem_exit_percentile))

    # REM is weighted toward later cycles: epochs before this fraction of the
    # sleep period are blocked from REM entirely.
    length = max(1, offset - onset)
    pos = (np.arange(n) - onset) / length  # fractional position within period
    rem_allowed = pos >= s.rem_earliest_fraction

    # --- Sequential hysteresis scan ---
    # DEEP  = low HR tail + sustained low movement + stable HR (low variability).
    # REM   = high HR tail + low movement (muscle atonia) + variability gate.
    # REM is resolved before DEEP; the low/high tails are disjoint by construction,
    # so REM just wins any boundary tie (preserves the old "REM overrides deep").
    current = LIGHT
    for i in range(n):
        if not asleep_here[i]:
            current = LIGHT  # AWAKE (or outside period) breaks any bout
            continue

        deep_ok = sustained_low_act[i] <= s.deep_act_max and hrv[i] <= s.deep_hrv_max
        rem_ok = (
            series.act[i] <= s.rem_act_max
            and hrv[i] >= s.rem_hrv_min
            and rem_allowed[i]
        )

        if (current == REM and hr_smooth[i] >= rem_exit_thr and rem_ok) or (
            hr_smooth[i] >= rem_enter_thr and rem_ok
        ):
            current = REM
        elif (current == DEEP and hr_smooth[i] <= deep_exit_thr and deep_ok) or (
            hr_smooth[i] <= deep_enter_thr and deep_ok
        ):
            current = DEEP
        else:
            current = LIGHT
        stages[i] = current

    return stages
