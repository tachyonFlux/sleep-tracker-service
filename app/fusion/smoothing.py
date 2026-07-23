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
        prev = runs[k - 1] if k > 0 else None
        next_ = runs[k + 1] if k + 1 < len(runs) else None
        prev_val = prev[2] if prev is not None else None
        next_val = next_[2] if next_ is not None else None
        # Absorb into the neighbour; if both exist and agree, take it; otherwise
        # fall back to LIGHT as the neutral in-sleep default. But the target must
        # never equal the bout's own value, or the merge is a no-op and the loop
        # stalls (a LIGHT singleton between two different stages would otherwise
        # "merge" to LIGHT forever) — in that case absorb into the longer
        # neighbour instead, which is always a different stage and makes progress.
        if prev_val is not None and next_val is not None:
            target = prev_val if prev_val == next_val else LIGHT
        elif prev_val is not None:
            target = prev_val
        elif next_val is not None:
            target = next_val
        else:
            target = LIGHT
        neighbours = [r for r in (prev, next_) if r is not None]
        if target == val and neighbours:
            longer = max(neighbours, key=lambda r: r[1] - r[0])
            target = longer[2]
        out[start:end] = target

    return out
