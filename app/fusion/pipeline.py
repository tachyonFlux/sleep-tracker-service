"""Orchestrate the 4-step fusion and emit the hypnogram + metrics (doc §7).

Emits one session per detected sleep period: a long mid-night awakening splits
the night into multiple sessions rather than discarding the smaller stretch, so
all sleep is recorded (and each session maps to a Health Connect record).
"""

from __future__ import annotations

import numpy as np

from ..config import DEFAULTS, Params
from ..models import (
    HypnogramOut,
    Metrics,
    NightIn,
    NightSummary,
    SleepSession,
    Stage,
    StageInterval,
)
from .actigraphy import cole_kripke_sleep
from .hr_period import detect_sleep_periods
from .preprocess import build_series
from .smoothing import smooth_stages
from .staging import AWAKE, DEEP, LIGHT, REM, stage_asleep

_CODE_TO_STAGE = {AWAKE: Stage.AWAKE, LIGHT: Stage.LIGHT, DEEP: Stage.DEEP, REM: Stage.REM}


def _to_intervals(stages: np.ndarray, night_start: int, epoch_s: int) -> list[StageInterval]:
    intervals: list[StageInterval] = []
    i = 0
    n = len(stages)
    while i < n:
        j = i + 1
        while j < n and stages[j] == stages[i]:
            j += 1
        intervals.append(
            StageInterval(
                start_utc=night_start + i * epoch_s,
                end_utc=night_start + j * epoch_s,
                stage=_CODE_TO_STAGE[int(stages[i])],
            )
        )
        i = j
    return intervals


def run_fusion(night: NightIn, params: Params = DEFAULTS) -> HypnogramOut:
    # Honour the night's epoch length over the default if they differ.
    if night.epoch_seconds != params.epoch_seconds:
        params = Params(
            epoch_seconds=night.epoch_seconds,
            hr=params.hr,
            actigraphy=params.actigraphy,
            staging=params.staging,
            smoothing=params.smoothing,
        )
    epoch_s = params.epoch_seconds

    series = build_series(night.epochs, params)

    # Actigraphy is computed once over the whole timeline (its weighted window
    # stays valid at every period's edges); staging/smoothing run per period.
    asleep = cole_kripke_sleep(series, params)

    sessions: list[SleepSession] = []
    for onset, offset in detect_sleep_periods(series, params):
        stages = stage_asleep(series, asleep, (onset, offset), params)
        stages = smooth_stages(stages, (onset, offset), params)

        # Trim leading/trailing AWAKE so the session reflects detected sleep, not
        # the low-movement envelope (doc §8 "capture vs actual sleep").
        asleep_codes = stages != AWAKE
        if not asleep_codes[onset:offset].any():
            continue  # a still-but-awake block with no actual sleep — skip it
        sleeping_idx = np.where(asleep_codes)[0]
        s_start = int(max(onset, sleeping_idx[0]))
        s_end = int(min(offset, sleeping_idx[-1] + 1))

        intervals = _to_intervals(
            stages[s_start:s_end], night.night_start_utc + s_start * epoch_s, epoch_s
        )
        sessions.append(
            SleepSession(
                session_start_utc=night.night_start_utc + s_start * epoch_s,
                session_end_utc=night.night_start_utc + s_end * epoch_s,
                stages=intervals,
                metrics=_compute_metrics(stages, s_start, s_end, epoch_s),
            )
        )

    return HypnogramOut(
        night_start_utc=night.night_start_utc,
        tz_offset_min=night.tz_offset_min,
        sessions=sessions,
        summary=_summarise(sessions),
    )


def _summarise(sessions: list[SleepSession]) -> NightSummary:
    return NightSummary(
        session_count=len(sessions),
        total_sleep_min=round(sum(s.metrics.total_sleep_min for s in sessions), 1),
        awake_min=round(sum(s.metrics.awake_min for s in sessions), 1),
        light_min=round(sum(s.metrics.light_min for s in sessions), 1),
        deep_min=round(sum(s.metrics.deep_min for s in sessions), 1),
        rem_min=round(sum(s.metrics.rem_min for s in sessions), 1),
    )


def _compute_metrics(stages: np.ndarray, s_start: int, s_end: int, epoch_s: int) -> Metrics:
    seg = stages[s_start:s_end]
    per_min = epoch_s / 60.0

    def minutes(code: int) -> float:
        return float(np.count_nonzero(seg == code)) * per_min

    awake = minutes(AWAKE)
    light = minutes(LIGHT)
    deep = minutes(DEEP)
    rem = minutes(REM)
    total_sleep = light + deep + rem
    time_in_session = float(len(seg)) * per_min

    # Sleep latency: minutes of AWAKE before the first asleep epoch in session.
    asleep = seg != AWAKE
    first_sleep = int(np.argmax(asleep)) if asleep.any() else len(seg)
    latency = first_sleep * per_min
    waso = max(0.0, awake - latency)

    efficiency = (total_sleep / time_in_session) if time_in_session > 0 else 0.0
    return Metrics(
        total_sleep_min=round(total_sleep, 1),
        time_in_bed_min=round(time_in_session, 1),
        efficiency=round(efficiency, 3),
        sleep_latency_min=round(latency, 1),
        waso_min=round(waso, 1),
        awake_min=round(awake, 1),
        light_min=round(light, 1),
        deep_min=round(deep, 1),
        rem_min=round(rem, 1),
    )
