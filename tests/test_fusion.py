"""End-to-end fusion tests over synthetic nights + API/storage smoke tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULTS
from app.db import Store
from app.fusion import run_fusion
from app.fusion.preprocess import build_series
from app.models import Epoch, NightIn, Stage
from tests.synth import EPOCH_S, NOISE_FLOOR, _act, make_night


def only(res):
    """Assert exactly one session and return it (for single-session nights)."""
    assert len(res.sessions) == 1, f"expected 1 session, got {len(res.sessions)}"
    return res.sessions[0]


def _interval_seconds(session, stage: Stage) -> int:
    return sum(iv.end_utc - iv.start_utc for iv in session.stages if iv.stage == stage)


def test_detects_a_plausible_session():
    night = make_night(seed=1)
    res = run_fusion(night)
    s = only(res)

    # Session should sit inside the captured window and have real duration.
    assert s.session_start_utc >= night.night_start_utc
    assert s.session_end_utc > s.session_start_utc
    # ~430 min of intended sleep; allow generous slack for the heuristics.
    assert 300 <= res.summary.total_sleep_min <= 470
    assert 0.7 <= s.metrics.efficiency <= 1.0


def test_all_four_stages_present():
    res = run_fusion(make_night(seed=2))
    # Sleep/wake must work; the synthetic night has clear deep/REM structure.
    assert res.summary.light_min > 0
    assert res.summary.deep_min > 0
    assert res.summary.rem_min > 0


def test_stage_durations_sum_to_session():
    s = only(run_fusion(make_night(seed=3)))
    span_min = (s.session_end_utc - s.session_start_utc) / 60.0
    staged = sum(_interval_seconds(s, st) for st in Stage) / 60.0
    assert staged == pytest.approx(span_min, abs=0.6)
    summed = (
        s.metrics.awake_min + s.metrics.light_min + s.metrics.deep_min + s.metrics.rem_min
    )
    assert summed == pytest.approx(s.metrics.time_in_bed_min, abs=0.6)


def test_intervals_are_contiguous_and_ordered():
    s = only(run_fusion(make_night(seed=4)))
    assert s.stages[0].start_utc == s.session_start_utc
    assert s.stages[-1].end_utc == s.session_end_utc
    for a, b in zip(s.stages, s.stages[1:]):
        assert a.end_utc == b.start_utc  # no gaps/overlaps
        assert a.stage != b.stage        # runs are merged


def test_mid_night_wake_under_60min_stays_one_session():
    # A 45-min awakening must NOT split the night: one session should span it,
    # with sleep on BOTH sides of the wake and the wake surfacing as WASO.
    baseline = run_fusion(make_night(seed=10))
    res = run_fusion(make_night(seed=10, mid_wake_min=45, mid_wake_at_frac=0.5))
    s = only(res)

    span_min = (s.session_end_utc - s.session_start_utc) / 60.0
    assert span_min >= 430, f"session truncated to {span_min:.0f} min"
    # Both sleep halves retained (total sleep close to the no-wake baseline).
    assert res.summary.total_sleep_min >= baseline.summary.total_sleep_min - 60
    # The wake is detected as WASO inside the session, not dropped.
    assert s.metrics.waso_min >= 25

    # Sleep exists before AND after an interior AWAKE bout (proves no split).
    stages = [iv.stage for iv in s.stages]
    interior = [i for i, st in enumerate(stages) if st == Stage.AWAKE and 0 < i < len(stages) - 1]
    assert interior, "no interior wake bout found"
    w = interior[len(interior) // 2]
    assert any(st != Stage.AWAKE for st in stages[:w])
    assert any(st != Stage.AWAKE for st in stages[w + 1:])


@pytest.mark.parametrize("wake_min", [15, 30, 50])
def test_short_mid_wakes_preserve_total_sleep(wake_min):
    baseline = run_fusion(make_night(seed=12)).summary.total_sleep_min
    res = run_fusion(make_night(seed=12, mid_wake_min=wake_min, mid_wake_at_frac=0.4))
    # Sub-hour wakes never cost us a sleep half.
    assert res.summary.total_sleep_min >= baseline - 60


def test_long_wake_splits_into_two_sessions_keeping_all_sleep():
    # 120 min sleep + 150 min sustained restless wake + 120 min sleep. The wake
    # exceeds the 120 min bridge so the night splits, but ALL sleep must be
    # retained across two sessions (the prior behaviour discarded one half).
    rng = random.Random(3)
    ep, t = [], 0

    def floor():
        return NOISE_FLOOR + rng.gauss(0, 120)

    for _ in range(240):  # 120 min sleep
        ep.append(Epoch(t=t, act=int(floor() + abs(rng.gauss(0, 30))), hr=54 + int(rng.gauss(0, 1)))); t += 1
    for _ in range(300):  # 150 min sustained wake (longer than the 120 min bridge)
        ep.append(Epoch(t=t, act=int(floor() + rng.uniform(6000, 20000)), hr=72 + int(rng.gauss(0, 4)))); t += 1
    for _ in range(240):  # 120 min sleep
        ep.append(Epoch(t=t, act=int(floor() + abs(rng.gauss(0, 30))), hr=54 + int(rng.gauss(0, 1)))); t += 1

    res = run_fusion(NightIn(night_start_utc=0, tz_offset_min=0, epoch_seconds=EPOCH_S, epochs=ep))

    assert len(res.sessions) == 2, f"expected 2 sessions, got {len(res.sessions)}"
    # Both ~120 min sleep stretches recorded (≈240 total), not truncated to one.
    assert res.summary.total_sleep_min >= 200
    for s in res.sessions:
        assert s.metrics.total_sleep_min >= 80
    # Sessions are ordered and disjoint.
    a, b = res.sessions
    assert a.session_end_utc <= b.session_start_utc


def test_survives_bluetooth_gaps():
    # 20% of epochs dropped (simulated BT batching loss) should still resolve.
    res = run_fusion(make_night(seed=5, drop_fraction=0.20))
    assert res.summary.total_sleep_min > 250


def test_no_sleep_when_only_movement():
    # Pure restless-wake night: spiky, variable high movement (not a flat value,
    # which floor-subtraction would zero). Expect no sleep session of substance.
    rng = random.Random(99)
    epochs = [Epoch(t=i, act=_act("wake", rng), hr=72 + int(rng.gauss(0, 4))) for i in range(200)]
    night = NightIn(night_start_utc=0, tz_offset_min=0, epoch_seconds=EPOCH_S, epochs=epochs)
    res = run_fusion(night)
    assert res.summary.total_sleep_min < 30


def test_min_bout_lengths_enforced():
    res = run_fusion(make_night(seed=6))
    sm = DEFAULTS.smoothing
    floor = {
        Stage.AWAKE: sm.min_wake_epochs,
        Stage.LIGHT: sm.min_light_epochs,
        Stage.DEEP: sm.min_deep_epochs,
        Stage.REM: sm.min_rem_epochs,
    }
    for s in res.sessions:
        for iv in s.stages:
            # Interior bouts must meet the minimum; the first/last may be clipped
            # by the session-trim, so exempt each session's boundary intervals.
            if iv is s.stages[0] or iv is s.stages[-1]:
                continue
            epochs = (iv.end_utc - iv.start_utc) // EPOCH_S
            assert epochs >= floor[iv.stage], f"{iv.stage} bout too short: {epochs}"


def test_preprocess_fills_hr_gaps():
    night = make_night(seed=7)
    series = build_series(night.epochs, DEFAULTS)
    assert not np.isnan(series.hr_filled).any()  # gaps interpolated
    assert series.n == max(e.t for e in night.epochs) + 1


def test_api_and_storage_roundtrip():
    from app import main as main_mod

    main_mod.store = Store(":memory:")
    client = TestClient(main_mod.app)

    assert client.get("/healthz").json()["status"] == "ok"

    night = make_night(seed=8)
    r = client.post("/night", json=night.model_dump())
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_sleep_min"] > 0
    assert body["sessions"]
    assert body["sessions"][0]["session_end_utc"] > body["sessions"][0]["session_start_utc"]

    nights = client.get("/nights").json()["nights"]
    assert len(nights) == 1
    assert nights[0]["result"]["summary"]["total_sleep_min"] > 0
