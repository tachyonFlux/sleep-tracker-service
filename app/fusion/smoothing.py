"""Step 4 — smoothing / minimum bout lengths / fragment merging (doc §7.4).

Removes physiologically implausible flips (e.g. an isolated 30 s REM epoch) by
enforcing per-stage minimum bout lengths, absorbing too-short bouts into their
neighbours. Operates only within the sleep period [onset, offset); epochs
outside stay AWAKE.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from .staging import AWAKE, DEEP, LIGHT, REM


def _runs(arr: np.ndarray, lo: int, hi: int) -> list[tuple[int, int, int]]:
    """List of (start, end, value) runs within [lo, hi)."""
    runs: list[tuple[int, int, int]] = []
    i = lo
    while i < hi:
        j = i + 1
        while j < hi and arr[j] == arr[i]:
            j += 1
        runs.append((i, j, int(arr[i])))
        i = j
    return runs


def smooth_stages(stages: np.ndarray, period: tuple[int, int], params: Params) -> np.ndarray:
    sm = params.smoothing
    onset, offset = period
    out = stages.copy()

    min_len = {
        AWAKE: sm.min_wake_epochs,
        LIGHT: sm.min_light_epochs,
        DEEP: sm.min_deep_epochs,
        REM: sm.min_rem_epochs,
    }

    # Iterate to a fixed point: each pass absorbs the single shortest offending
    # bout, then recomputes runs (absorbing one bout can shorten/merge others).
    # Every merge removes at least one bout, so the loop is bounded by the epoch
    # count of the sleep period.
    for _ in range(offset - onset + 1):
        runs = _runs(out, onset, offset)
        offenders = [
            (k, start, end, val)
            for k, (start, end, val) in enumerate(runs)
            if (end - start) < min_len.get(val, 1)
        ]
        if not offenders:
            break
        # Shortest first for stable, deterministic merging.
        k, start, end, val = min(offenders, key=lambda r: r[2] - r[1])
        prev_val = runs[k - 1][2] if k > 0 else None
        next_val = runs[k + 1][2] if k + 1 < len(runs) else None
        # Absorb into the neighbour; if both exist and agree, take it; otherwise
        # fall back to LIGHT as the neutral in-sleep default.
        if prev_val is not None and next_val is not None:
            target = prev_val if prev_val == next_val else LIGHT
        elif prev_val is not None:
            target = prev_val
        elif next_val is not None:
            target = next_val
        else:
            target = LIGHT
        out[start:end] = target

    return out
