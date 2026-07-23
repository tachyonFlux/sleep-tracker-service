"""Stage-level grading of hypnograms against PhysioNet PSG ground truth.

The sleep/wake grader (grade.py) ignores which sleep stage was predicted. This
scores the full 4-stage problem (AWAKE/LIGHT/DEEP/REM) so deep/REM heuristics can
be calibrated. Only epochs the model calls asleep are meaningfully staged, so we
report stage metrics both over all scored epochs and restricted to true-sleep
epochs (where the stage split is what we actually care about).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Params
from app.fusion import run_fusion
from tests.grade import fixture_to_night, predicted_per_epoch
from tests.synth import LEGACY_PARAMS

STAGES = ["AWAKE", "LIGHT", "DEEP", "REM"]


@dataclass
class StageScore:
    # confusion[t][p] = count of truth=t predicted=p
    confusion: dict[str, dict[str, int]] = field(
        default_factory=lambda: {t: {p: 0 for p in STAGES} for t in STAGES}
    )

    def add(self, truth: str, pred: str) -> None:
        self.confusion[truth][pred] += 1

    def truth_total(self, t: str) -> int:
        return sum(self.confusion[t].values())

    def pred_total(self, p: str) -> int:
        return sum(self.confusion[t][p] for t in STAGES)

    def sensitivity(self, s: str) -> float:  # recall of stage s
        d = self.truth_total(s)
        return self.confusion[s][s] / d if d else float("nan")

    def precision(self, s: str) -> float:
        d = self.pred_total(s)
        return self.confusion[s][s] / d if d else float("nan")

    def overall_accuracy(self) -> float:
        correct = sum(self.confusion[s][s] for s in STAGES)
        total = sum(self.truth_total(s) for s in STAGES)
        return correct / total if total else float("nan")

    def cohen_kappa(self) -> float:
        total = sum(self.truth_total(s) for s in STAGES)
        if not total:
            return float("nan")
        po = sum(self.confusion[s][s] for s in STAGES) / total
        pe = sum(self.truth_total(s) * self.pred_total(s) for s in STAGES) / total**2
        return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def score_fixture(fix: dict, params: Params, sleep_only: bool = False) -> StageScore:
    res = run_fusion(fixture_to_night(fix), params)
    pred = predicted_per_epoch(fix, res)
    score = StageScore()
    for p, t in zip(pred, fix["truth"]):
        if t is None:
            continue
        if sleep_only and t == "AWAKE":
            continue
        score.add(t, p)
    return score


def _fmt(v: float) -> str:
    return "  n/a" if v != v else f"{v:5.3f}"


def print_report(name: str, score: StageScore) -> None:
    print(f"\n=== {name} ===")
    print("confusion (rows=truth, cols=pred):")
    header = "        " + "".join(f"{p:>8}" for p in STAGES) + "     tot"
    print(header)
    for t in STAGES:
        row = "".join(f"{score.confusion[t][p]:8d}" for p in STAGES)
        print(f"{t:>8}{row}{score.truth_total(t):8d}")
    print("     pred" + "".join(f"{score.pred_total(p):8d}" for p in STAGES))
    print("per-stage  sensitivity / precision:")
    for s in STAGES:
        print(f"  {s:>6}: sens {_fmt(score.sensitivity(s))}  prec {_fmt(score.precision(s))}")
    print(f"  overall acc {_fmt(score.overall_accuracy())}  kappa {_fmt(score.cohen_kappa())}")


def merge(scores: list[StageScore]) -> StageScore:
    out = StageScore()
    for sc in scores:
        for t in STAGES:
            for p in STAGES:
                out.confusion[t][p] += sc.confusion[t][p]
    return out


if __name__ == "__main__":
    import sys

    from tests.grade import load_fixtures

    sleep_only = "--sleep-only" in sys.argv
    params = LEGACY_PARAMS
    per = []
    for fix in load_fixtures():
        sc = score_fixture(fix, params, sleep_only=sleep_only)
        per.append(sc)
        print_report(fix.get("subject", "?"), sc)
    print_report("POOLED", merge(per))
