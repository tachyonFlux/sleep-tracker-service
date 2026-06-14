"""Generate plausible synthetic nights of epoch features for testing.

Produces the same shape the watch logs: per-epoch activity count + HR. Activity
counts are emitted on the watch's milli-g scale, matching the real PhysioNet
data: a per-night NOISE FLOOR (~2000 milli-g of 50 Hz jitter even when still)
plus movement excess that the pipeline recovers by floor subtraction. This keeps
synth and real-data tests on a single count_scale calibration.
"""

from __future__ import annotations

import random

from app.models import Epoch, NightIn

EPOCH_S = 30
EPOCHS_PER_MIN = 60 // EPOCH_S
NOISE_FLOOR = 2000.0  # milli-g; matches the observed real-data still floor


def _push(epochs, t, act, hr):
    epochs.append(Epoch(t=t, act=max(0, int(act)), hr=max(0, int(round(hr)))))


def _act(kind: str, rng: random.Random) -> int:
    """Activity count (milli-g) by stage: noise floor + stage-typical movement."""
    base = NOISE_FLOOR + rng.gauss(0, 120)
    if kind == "deep":
        mv = abs(rng.gauss(0, 25))
    elif kind == "light":
        mv = abs(rng.gauss(0, 110)) + (rng.uniform(150, 700) if rng.random() < 0.10 else 0)
    elif kind == "rem":
        mv = abs(rng.gauss(0, 70)) + (rng.uniform(100, 400) if rng.random() < 0.05 else 0)
    else:  # wake / restless: frequent large movement spikes
        mv = abs(rng.gauss(0, 1500)) + (rng.uniform(3000, 18000) if rng.random() < 0.5 else 0)
    return int(base + mv)


def make_night(
    seed: int = 0,
    pre_wake_min: int = 25,
    latency_min: int = 12,
    sleep_min: int = 430,
    post_wake_min: int = 20,
    resting_hr: float = 55.0,
    awake_hr: float = 72.0,
    night_start_utc: int = 1_736_899_200,
    tz_offset_min: int = -300,
    drop_fraction: float = 0.0,
    mid_wake_min: int = 0,
    mid_wake_at_frac: float = 0.5,
) -> NightIn:
    """Build a night: [awake pad] [latency] [sleep w/ cycles] [awake pad].

    The sleep block cycles light->deep->REM on a ~90 min period so staging has
    something to find. `drop_fraction` randomly omits epochs to simulate BT gaps.
    `mid_wake_min` injects a restless awakening that many minutes long at
    `mid_wake_at_frac` through the sleep block (for the mid-night-wake tests).
    """
    rng = random.Random(seed)
    epochs: list[Epoch] = []
    t = 0

    def block(kind: str, minutes: float, hr_fn):
        nonlocal t
        for _ in range(int(minutes * EPOCHS_PER_MIN)):
            _push(epochs, t, act=_act(kind, rng), hr=hr_fn())
            t += 1

    # Pre-sleep awake (in bed, restless) and sleep-onset latency.
    block("wake", pre_wake_min, lambda: awake_hr + rng.gauss(0, 4))
    block("wake", latency_min, lambda: awake_hr + rng.gauss(0, 4))

    # Main sleep with ~90 min cycles; optionally interrupted by a mid-night wake.
    total_sleep_epochs = int(sleep_min * EPOCHS_PER_MIN)
    cycle_epochs = int(90 * EPOCHS_PER_MIN)
    wake_at = int(total_sleep_epochs * mid_wake_at_frac) if mid_wake_min else -1
    for i in range(total_sleep_epochs):
        if i == wake_at:
            block("wake", mid_wake_min, lambda: awake_hr + rng.gauss(0, 4))
        phase = (i % cycle_epochs) / cycle_epochs  # 0..1 within a cycle
        night_frac = i / total_sleep_epochs
        if phase < 0.35:  # deep
            kind, hr = "deep", resting_hr + rng.gauss(0, 0.6)
        elif phase < 0.7:  # light
            kind, hr = "light", resting_hr + 3 + rng.gauss(0, 1.2)
        else:  # REM (more so later in the night)
            kind, hr = "rem", resting_hr + (6 + 4 * night_frac) + rng.gauss(0, 3.5)
        if rng.random() < 0.01:  # occasional brief arousal
            kind, hr = "wake", awake_hr + rng.gauss(0, 3)
        _push(epochs, t, act=_act(kind, rng), hr=hr)
        t += 1

    block("wake", post_wake_min, lambda: awake_hr + rng.gauss(0, 4))

    if drop_fraction > 0:
        epochs = [e for e in epochs if rng.random() >= drop_fraction]

    return NightIn(
        night_start_utc=night_start_utc,
        tz_offset_min=tz_offset_min,
        epoch_seconds=EPOCH_S,
        epochs=epochs,
    )
