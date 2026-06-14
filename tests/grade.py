"""Helpers to grade a hypnogram against per-epoch ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.fusion import run_fusion
from app.models import HypnogramOut, NightIn

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


def fixture_to_night(fix: dict) -> NightIn:
    return NightIn(
        night_start_utc=fix["night_start_utc"],
        tz_offset_min=fix["tz_offset_min"],
        epoch_seconds=fix["epoch_seconds"],
        epochs=fix["epochs"],
    )


def predicted_per_epoch(fix: dict, res: HypnogramOut) -> list[str]:
    """Expand the result's stage intervals back to one label per input epoch.

    Epochs outside the detected session are AWAKE (not in any interval)."""
    n = len(fix["epochs"])
    es, ns = fix["epoch_seconds"], fix["night_start_utc"]
    pred = ["AWAKE"] * n
    for session in res.sessions:
        for iv in session.stages:
            start = (iv.start_utc - ns) // es
            end = (iv.end_utc - ns) // es
            for i in range(max(0, start), min(n, end)):
                pred[i] = iv.stage.value
    return pred


@dataclass
class SleepWake:
    n: int
    tp: int  # truth sleep, predicted sleep
    tn: int  # truth wake, predicted wake
    fp: int  # truth wake, predicted sleep
    fn: int  # truth sleep, predicted wake

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def sleep_sensitivity(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def wake_specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0


def sleep_wake_confusion(pred: list[str], truth: list[str | None]) -> SleepWake:
    tp = tn = fp = fn = 0
    for p, t in zip(pred, truth):
        if t is None:  # unscored PSG epoch
            continue
        ps, ts = p != "AWAKE", t != "AWAKE"
        if ts and ps:
            tp += 1
        elif ts and not ps:
            fn += 1
        elif (not ts) and (not ps):
            tn += 1
        else:
            fp += 1
    return SleepWake(n=tp + tn + fp + fn, tp=tp, tn=tn, fp=fp, fn=fn)


def grade_fixture(fix: dict) -> SleepWake:
    res = run_fusion(fixture_to_night(fix))
    return sleep_wake_confusion(predicted_per_epoch(fix, res), fix["truth"])
