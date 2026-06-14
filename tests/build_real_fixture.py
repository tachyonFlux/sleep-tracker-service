"""Convert raw PhysioNet 'sleep-accel' (Walch et al. 2019) files into compact
epoch-feature fixtures with PSG ground truth, for real-data validation tests.

Dataset: https://physionet.org/content/sleep-accel/1.0.0/ (open access, ODC-BY).
Per subject there are three files (timestamps are seconds from PSG start):
  motion/<id>_acceleration.txt   "<t> <x> <y> <z>"  in g, ~50 Hz
  heart_rate/<id>_heartrate.txt  "<t>,<bpm>"
  labels/<id>_labeled_sleep.txt  "<t> <stage>"  30 s epochs

This computes, for each 30 s epoch, the SAME feature the watch firmware will log:
  - activity count = sum of |Δ magnitude| between consecutive accel samples,
    magnitude in milli-g (so the count is on the watch's int16 milli-g scale,
    making count_scale calibrated here directly reusable on the device);
  - HR = median of bpm readings in the epoch (0 = none);
and records PSG ground truth per epoch.

Raw files (tens of MB) live in ./realdata/ and are gitignored; the emitted
fixture JSON (a few KB) is committed under tests/fixtures/.

Run: .venv/bin/python -m tests.build_real_fixture 46343 [8000685 ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import median

EPOCH_S = 30
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "realdata"
FIXTURES = ROOT / "tests" / "fixtures"

# PSG stage code -> our 4-stage name. N1+N2 collapse to LIGHT; N3 -> DEEP.
PSG_TO_STAGE = {0: "AWAKE", 1: "LIGHT", 2: "LIGHT", 3: "DEEP", 5: "REM"}


def _read_labels(path: Path) -> dict[int, int]:
    """epoch_index -> raw PSG stage code (scored epochs only, i.e. code >= 0)."""
    out: dict[int, int] = {}
    for line in path.read_text().splitlines():
        t_str, stage_str = line.split()
        t, stage = int(float(t_str)), int(stage_str)
        if stage >= 0:  # drop -1 unscored
            out[t // EPOCH_S] = stage
    return out


def _read_hr_by_epoch(path: Path) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        t_str, bpm_str = line.replace(",", " ").split()
        t = float(t_str)
        if t < 0:
            continue
        out.setdefault(int(t // EPOCH_S), []).append(float(bpm_str))
    return out


def _accel_counts_by_epoch(path: Path) -> dict[int, int]:
    """Sum of |Δ magnitude| (milli-g) per epoch, streamed to bound memory."""
    counts: dict[int, float] = {}
    prev_mag: float | None = None
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            t = float(parts[0])
            if t < 0:  # skip the prior ambulatory-week samples
                prev_mag = None
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            mag = math.sqrt(x * x + y * y + z * z) * 1000.0  # g -> milli-g
            if prev_mag is not None:
                counts[int(t // EPOCH_S)] = counts.get(int(t // EPOCH_S), 0.0) + abs(mag - prev_mag)
            prev_mag = mag
    return {e: int(round(v)) for e, v in counts.items()}


def build(subject: str) -> Path:
    labels = _read_labels(RAW / f"{subject}_labeled_sleep.txt")
    hr = _read_hr_by_epoch(RAW / f"{subject}_heartrate.txt")
    acts = _accel_counts_by_epoch(RAW / f"{subject}_acceleration.txt")

    # Use the contiguous run of epochs that have BOTH a PSG label and accel
    # coverage (the watch motion stream can stop before PSG ends).
    common = sorted(e for e in labels if e in acts)
    if not common:
        raise SystemExit(f"{subject}: no overlap between accel and labels")
    start, end = common[0], common[-1]

    epochs, truth = [], []
    for i, e in enumerate(range(start, end + 1)):
        act = acts.get(e, 0)
        readings = hr.get(e, [])
        bpm = int(round(median(readings))) if readings else 0
        epochs.append({"t": i, "act": act, "hr": bpm})
        # ground-truth stage for this epoch (carry None where unscored gap)
        truth.append(PSG_TO_STAGE.get(labels.get(e, -1), None))

    fixture = {
        "source": "PhysioNet sleep-accel v1.0.0 (Walch et al. 2019)",
        "subject": subject,
        "epoch_seconds": EPOCH_S,
        "night_start_utc": 1_700_000_000,
        "tz_offset_min": 0,
        "epochs": epochs,
        "truth": truth,  # our-stage name per epoch index, or null if unscored
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"{subject}.json"
    out.write_text(json.dumps(fixture))
    scored = sum(t is not None for t in truth)
    asleep = sum(t not in (None, "AWAKE") for t in truth)
    print(f"{subject}: {len(epochs)} epochs ({scored} scored, {asleep} asleep) -> {out.name}")
    return out


if __name__ == "__main__":
    for subj in sys.argv[1:] or ["46343"]:
        build(subj)
