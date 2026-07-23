"""Stage-level (deep/REM) diagnostic against PhysioNet PSG ground truth.

This is a DIAGNOSTIC, deliberately NOT a tight gate. Full 4-stage classification
from a BPM-only sensor with no training is a genuinely hard problem, and we have
only three PhysioNet subjects (none on PT2 hardware) — tuning to hit precise
per-stage numbers here would overfit. The sleep/wake test (test_realdata.py) is
the real accuracy gate; this test only guards against staging *regressing back to
broken*, and prints the confusion report for eyeballing when run with -s.

Baseline for reference (2026-07, relative-HR bands, pooled over 3 subjects):
    overall acc 0.46, kappa 0.097
    AWAKE sens 0.63 / LIGHT 0.58 / DEEP 0.14 / REM 0.21
Before this rebalance REM recall was 0.06 (true REM swept into LIGHT); the floors
below sit well under the current numbers so normal tuning drift won't flap them,
but a return to the old collapse (REM ~0, or all deep/REM lost) will trip them.
"""

from __future__ import annotations

from tests.grade import load_fixtures
from tests.stage_grade import STAGES, merge, print_report, score_fixture
from tests.synth import LEGACY_PARAMS


def _pooled():
    scores = [score_fixture(fix, LEGACY_PARAMS) for fix in load_fixtures()]
    return merge(scores)


def test_staging_diagnostic_report(capsys):
    pooled = _pooled()
    with capsys.disabled():
        print_report("POOLED (deep/REM diagnostic)", pooled)

    # Loose non-regression floors — "staging is not broken", not "staging is good".
    assert pooled.cohen_kappa() > 0.04, "4-stage agreement collapsed to ~chance"
    assert pooled.overall_accuracy() > 0.35

    # Every stage must actually be produced (the old bug emitted ~0 REM).
    for stage in STAGES:
        assert pooled.pred_total(stage) > 0, f"no {stage} epochs predicted at all"

    # REM must stay well clear of its old broken 0.06 recall.
    assert pooled.sensitivity("REM") > 0.10, "REM recall regressed toward broken"
    # Deep must remain detectable in aggregate.
    assert pooled.sensitivity("DEEP") > 0.05
