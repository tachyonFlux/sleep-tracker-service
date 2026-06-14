"""All tunable algorithm parameters in one place (handoff doc §8).

Every threshold the fusion pipeline uses lives here so it can be tuned against
subjective sleep logs without touching algorithm code. The defaults are the
documented starting points; expect to calibrate the actigraphy scaling and the
HR margins against your own nights.

NOTE on activity-count units: Cole-Kripke / Sadeh coefficients were validated
against a *specific* actigraph's count units. Our watch produces an arbitrary
"|delta magnitude| sum per epoch" count, so `count_scale` rescales our counts
into the range those formulas expect. This MUST be calibrated on real data —
see actigraphy.py. Treat the default as a placeholder, not gospel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HRPeriodParams:
    """Step 1 — HR-baseline sleep-period (onset/offset) detection."""

    # Resting baseline = this percentile of valid HR across the night.
    baseline_percentile: float = 10.0
    # An epoch is an HR "sleep candidate" if HR <= baseline + this margin (bpm).
    sleep_hr_margin_bpm: float = 6.0
    # ...and activity at/below this count (rescaled units; see count_scale).
    sleep_act_max: float = 0.30
    # Smoothing window (epochs) applied to HR before thresholding.
    hr_smooth_epochs: int = 5
    # Bridge movement gaps up to this many epochs when stitching the sleep
    # block, so a mid-night awakening stays inside ONE session (it surfaces as
    # an AWAKE/WASO interval, not a session split). 240 epochs = 120 min: any
    # awakening up to two hours is absorbed into a single session; only a truly
    # long gap (e.g. a nap hours apart from the main sleep) splits the night.
    # NOTE: no sleep is ever discarded regardless of this value — a split just
    # produces multiple sessions; this only controls session granularity.
    bridge_gap_epochs: int = 240
    # Reject sleep blocks shorter than this (epochs) as noise.
    min_block_epochs: int = 40


@dataclass(frozen=True)
class ActigraphyParams:
    """Step 2 — Cole-Kripke wake-within-sleep classification."""

    # Per-night noise floor: the |delta-mag| sum is non-zero even when still
    # (~2000 milli-g of 50 Hz sensor jitter on real data). We subtract this
    # percentile of the night's counts so the algorithm keys on movement
    # *excess*, which makes count_scale robust to absolute sensor units.
    floor_percentile: float = 10.0
    # Rescales the de-floored activity counts into Cole-Kripke's expected range.
    # Calibrated against the PhysioNet sleep-accel dataset (see tests). Re-verify
    # on real PT2 data — its milli-g scale may differ.
    count_scale: float = 0.001
    # Cole-Kripke (1992) "optimal" coefficients, weights for epochs
    # [-4, -3, -2, -1, 0, +1, +2] relative to the scored epoch.
    ck_weights: tuple[float, ...] = (106.0, 54.0, 58.0, 76.0, 230.0, 74.0, 67.0)
    ck_offset: int = 4  # index of the scored epoch within ck_weights
    ck_p: float = 0.001  # overall scaling P in D = P * sum(w_i * A_i)
    # D < ck_threshold => sleep, else wake.
    ck_threshold: float = 1.0


@dataclass(frozen=True)
class StagingParams:
    """Step 3 — deep / REM / light heuristics for asleep epochs."""

    # "Night minimum" HR = this percentile of valid HR within the sleep period.
    hr_floor_percentile: float = 5.0
    # DEEP: HR within this margin of the night floor (bpm)...
    deep_hr_margin_bpm: float = 4.0
    # ...and activity at/below this (rescaled), sustained over deep_window epochs.
    deep_act_max: float = 0.15
    deep_window_epochs: int = 6
    # ...and epoch-to-epoch HR variability below this (bpm).
    deep_hrv_max: float = 1.5

    # REM: activity at/below this (rescaled)...
    rem_act_max: float = 0.25
    # ...and HR elevated at least this far above the night floor (bpm)...
    rem_hr_floor_bpm: float = 5.0
    # ...and HR variability at least this high (bpm)...
    rem_hrv_min: float = 2.5
    # ...with occurrence weighted toward later cycles: epochs before this
    # fraction of the sleep period are blocked from REM entirely.
    rem_earliest_fraction: float = 0.20

    # Window (epochs) for the epoch-to-epoch HR variability estimate.
    hrv_window_epochs: int = 5


@dataclass(frozen=True)
class SmoothingParams:
    """Step 4 — minimum bout lengths, cycle prior, fragment merging (epochs)."""

    min_wake_epochs: int = 2   # in-sleep wake bouts shorter than this -> sleep
    min_deep_epochs: int = 6   # ~3 min at 30 s
    min_rem_epochs: int = 6
    min_light_epochs: int = 2
    cycle_minutes: float = 90.0  # soft prior; used by staging weighting


@dataclass(frozen=True)
class Params:
    """Top-level parameter bundle passed through the whole pipeline."""

    epoch_seconds: int = 30
    hr: HRPeriodParams = field(default_factory=HRPeriodParams)
    actigraphy: ActigraphyParams = field(default_factory=ActigraphyParams)
    staging: StagingParams = field(default_factory=StagingParams)
    smoothing: SmoothingParams = field(default_factory=SmoothingParams)


DEFAULTS = Params()
