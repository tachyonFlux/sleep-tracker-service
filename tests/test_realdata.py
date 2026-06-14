"""Validate the fusion pipeline against REAL PSG-labeled wrist data.

Fixtures under tests/fixtures/*.json are derived from the PhysioNet 'sleep-accel'
dataset (Walch et al. 2019) — real Apple Watch accel + PPG HR with gold-standard
polysomnography labels — via tests/build_real_fixture.py.

What this proves: the no-training actigraphy + HR fusion recovers sleep/wake at
accuracy/sensitivity consistent with the published actigraphy literature
(~80-90% accuracy, high sleep sensitivity, modest wake specificity — quiet wake
is genuinely hard to distinguish from sleep without training).

What it does NOT prove: 4-stage accuracy. In this lab dataset HR barely differs
between wake and sleep (subjects lie still throughout), so stage separation is
weak — exactly the BPM-only ceiling the handoff doc flags. Stage accuracy must
be validated on real PT2 nights against subjective logs.
"""

from __future__ import annotations

import pytest

from tests.grade import grade_fixture, load_fixtures

FIXTURES = load_fixtures()
IDS = [f["subject"] for f in FIXTURES]


@pytest.mark.skipif(not FIXTURES, reason="no real-data fixtures present")
@pytest.mark.parametrize("fix", FIXTURES, ids=IDS)
def test_sleep_wake_agreement_per_subject(fix):
    sw = grade_fixture(fix)
    # Per-subject floors with margin below observed (0.79-0.91 acc, 0.81-0.97 sens).
    assert sw.accuracy >= 0.72, f"{fix['subject']} accuracy {sw.accuracy:.3f}"
    assert sw.sleep_sensitivity >= 0.75, f"{fix['subject']} sens {sw.sleep_sensitivity:.3f}"


@pytest.mark.skipif(not FIXTURES, reason="no real-data fixtures present")
def test_sleep_wake_agreement_pooled():
    tp = tn = fp = fn = 0
    for fix in FIXTURES:
        sw = grade_fixture(fix)
        tp += sw.tp
        tn += sw.tn
        fp += sw.fp
        fn += sw.fn
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n
    sleep_sens = tp / (tp + fn)
    # Pooled over all subjects; observed ~0.84 acc / ~0.86 sens at count_scale=0.001.
    assert accuracy >= 0.78, f"pooled accuracy {accuracy:.3f}"
    assert sleep_sens >= 0.80, f"pooled sleep sensitivity {sleep_sens:.3f}"
