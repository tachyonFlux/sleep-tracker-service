"""Step 1 — HR-baseline sleep-period detection (handoff doc §7.1).

Find the sleep period(s): sustained low-movement blocks (stillness persists
across every sleep stage, REM included), with HR settling below a nightly
resting baseline used to confirm onset. Brief awakenings (< bridge_gap) are
absorbed into a block and surface later as WASO; a *long* awakening splits the
night into MULTIPLE blocks — each is returned as its own period so no real sleep
is ever discarded (the caller emits one session per period).
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from .preprocess import Series, moving_average


def _bridged_blocks(mask: np.ndarray, bridge_gap: int, min_len: int) -> list[tuple[int, int]]:
    """All runs of True (bridging False gaps <= bridge_gap) with length >= min_len."""
    if not mask.any():
        return []

    # Bridge short False gaps so brief arousals don't split a block.
    bridged = mask.copy()
    false_run = 0
    start_gap = -1
    for i, v in enumerate(mask):
        if not v:
            if false_run == 0:
                start_gap = i
            false_run += 1
        else:
            if 0 < false_run <= bridge_gap and start_gap > 0:
                bridged[start_gap:i] = True
            false_run = 0

    blocks: list[tuple[int, int]] = []
    run_start = -1
    for i in range(len(bridged) + 1):
        inside = i < len(bridged) and bridged[i]
        if inside and run_start < 0:
            run_start = i
        elif not inside and run_start >= 0:
            if i - run_start >= min_len:
                blocks.append((run_start, i))
            run_start = -1
    return blocks


def detect_sleep_periods(series: Series, params: Params) -> list[tuple[int, int]]:
    """Return [onset, offset) epoch indices for EVERY sleep period of the night.

    The envelope is driven by sustained low movement: stillness persists across
    every sleep stage (REM included), whereas HR rises during REM and would carve
    real sleep out of the window if used as a per-epoch gate. HR is instead used
    to *confirm onset* of each block — advancing the start to where HR has settled
    below baseline, so a still-but-awake "lying in bed" stretch isn't counted.

    A long awakening (high movement for longer than bridge_gap) yields multiple
    blocks; all are returned so every stretch of real sleep is recorded.
    """
    p = params.hr

    act_ok = series.act <= p.sleep_act_max
    blocks = _bridged_blocks(act_ok, p.bridge_gap_epochs, p.min_block_epochs)
    if not blocks:
        return []

    # Confirm onset of each block via HR settling below baseline (if HR exists).
    # Baseline is computed once over the whole night's valid readings.
    if np.isfinite(series.hr).any():
        smooth_hr = moving_average(series.hr_filled, p.hr_smooth_epochs)
        baseline = float(np.percentile(series.hr[np.isfinite(series.hr)], p.baseline_percentile))
        settled = smooth_hr <= baseline + p.sleep_hr_margin_bpm
    else:
        settled = None

    periods: list[tuple[int, int]] = []
    for onset, offset in blocks:
        if settled is not None:
            idx = np.where(settled[onset:offset])[0]
            if idx.size:
                onset = onset + int(idx[0])
        periods.append((onset, offset))
    return periods
