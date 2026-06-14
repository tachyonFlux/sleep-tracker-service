"""Step 2 — actigraphy sleep/wake within the sleep period (handoff doc §7.2).

Cole-Kripke (1992) weighted-window classifier. For each epoch:

    D = P * sum_i ( w_i * A_{epoch+i-offset} )
    sleep if D < threshold, else wake

CALIBRATION WARNING (doc §7, §11): the Cole-Kripke weights were validated for a
specific actigraph's *count units*. Our watch emits an arbitrary |delta-mag| sum,
so `ActigraphyParams.count_scale` rescales our counts into that range. The scale
and threshold are the two knobs to calibrate against subjective logs / the SWW
reference coefficients. The algorithm *structure* here is faithful; the constants
are the part you must tune on real PT2 data before trusting wake detection.
"""

from __future__ import annotations

import numpy as np

from ..config import Params
from .preprocess import Series


def cole_kripke_sleep(series: Series, params: Params) -> np.ndarray:
    """Boolean array, True = asleep, computed over the whole timeline.

    The caller restricts attention to the detected sleep period; computing over
    the full series keeps the weighted window valid at the period's edges.
    """
    a = params.actigraphy
    counts = series.act  # already rescaled by count_scale in preprocess
    n = series.n
    weights = np.asarray(a.ck_weights, dtype=np.float64)
    offset = a.ck_offset

    d = np.zeros(n, dtype=np.float64)
    for j, w in enumerate(weights):
        shift = j - offset  # epoch position relative to scored epoch
        # d[i] += w * counts[i + shift], with out-of-range treated as 0.
        if shift == 0:
            d += w * counts
        elif shift > 0:
            d[: n - shift] += w * counts[shift:]
        else:  # shift < 0
            d[-shift:] += w * counts[: n + shift]

    d *= a.ck_p
    return d < a.ck_threshold
