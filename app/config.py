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
    # Calibrated 2026-06 against real PT2 HealthService VMC nights with
    # user-confirmed wake windows (0.001 was the PhysioNet |delta-mag| value;
    # VMC units run lower, so wake needs a bigger scale). Sweep showed 0.003 =
    # knee of catch-vs-false-wake curve on nights 6/7.
    count_scale: float = 0.003
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

    # Deep/REM are carved from RELATIVE percentiles of this night's own asleep
    # HR distribution, not absolute bpm margins — resting HR and its range vary
    # night to night, so fixed margins found all deep on one night and none on
    # the next (validated on PhysioNet PSG: deep sens swung 0.36 -> 0.00 across
    # subjects). See staging.py.
    # DEEP: smoothed HR at/below this percentile of asleep HR (the low tail)...
    deep_hr_percentile: float = 45.0
    # ...and activity at/below this (rescaled), sustained over deep_window epochs.
    deep_act_max: float = 0.15
    deep_window_epochs: int = 6
    # ...and epoch-to-epoch HR variability below this (bpm). Loose: only rejects
    # the most volatile epochs from the low-HR tail.
    deep_hrv_max: float = 3.0

    # REM: smoothed HR at/above this percentile of asleep HR (the high tail)...
    rem_hr_percentile: float = 60.0
    # ...and activity at/below this (rescaled; muscle atonia keeps REM still)...
    rem_act_max: float = 0.30
    # ...and HR variability at least this high (bpm). 0 = off: the percentile
    # band carries REM; the strict >=2.5 gate collapsed recall to 0.06 on PSG.
    rem_hrv_min: float = 0.0
    # ...with occurrence weighted toward later cycles: epochs before this
    # fraction of the sleep period are blocked from REM entirely.
    rem_earliest_fraction: float = 0.15

    # Window (epochs) for the epoch-to-epoch HR variability estimate.
    hrv_window_epochs: int = 5

    # WAKE-BY-HR: actigraphy cannot see a quiet-but-roused awakening (lying
    # still, tending a child...). Smoothed HR sustained this far above the
    # session median HR marks the epoch AWAKE even if movement was low.
    # Margin sits above the REM band (REM ~ floor+5; median is above floor,
    # so median+12 clears it) — validated on real nights 6/7.
    wake_hr_margin_bpm: float = 12.0
    # ...sustained for at least this many epochs (6 = 3 min at 30 s).
    wake_hr_min_epochs: int = 6


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
    # Data-quality gate: staging is 100% HR-driven, so a night where the HR
    # sensor barely read is un-stageable. Below this fraction of epochs carrying
    # an HR reading, emit NO sessions (a "not enough signal" night) rather than
    # inventing a hypnogram from interpolation. Real failure seen at 0.077 (night
    # 10, sensor dropout); healthy nights run 0.96-1.00, so 0.5 cleanly separates.
    min_hr_coverage: float = 0.5
    hr: HRPeriodParams = field(default_factory=HRPeriodParams)
    actigraphy: ActigraphyParams = field(default_factory=ActigraphyParams)
    staging: StagingParams = field(default_factory=StagingParams)
    smoothing: SmoothingParams = field(default_factory=SmoothingParams)


DEFAULTS = Params()
