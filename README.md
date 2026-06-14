# Sleep Tracker — Fusion Service (Component 3)

FastAPI service that turns watch-computed epoch features (activity count + HR per
30 s epoch) into a 4-stage hypnogram (Awake / Light / Deep / REM) using a
**no-training fusion** of actigraphy + HR-baseline heuristics. Runs in Docker on
the Synology NAS next to Vikunja. The phone POSTs a night, gets back stage
intervals, and writes them to Health Connect (Health Connect is phone-side only).

This is **Component 3** of the sleep tracker. See the handoff doc for the full
architecture (watch app → Android companion → this service).

## Data flow

```
phone  --POST /night-->  this service  --hypnogram JSON-->  phone --> Health Connect
                              |
                              +--> stores raw night + result in SQLite (history /
                                   future model-training upgrade path)
```

## API

| Method | Path       | Purpose                                            |
|--------|------------|----------------------------------------------------|
| GET    | `/healthz` | Liveness probe                                     |
| POST   | `/night`   | Ingest epoch features → fusion → hypnogram (stored)|
| GET    | `/nights`  | Recent stored hypnograms (history / debugging)     |

### `POST /night`
Request (see `app/models.py`):
```json
{
  "night_start_utc": 1736899200,
  "tz_offset_min": -300,
  "epoch_seconds": 30,
  "epochs": [{"t": 0, "act": 142, "hr": 58}, {"t": 1, "act": 0, "hr": 54}]
}
```
- `t` — epoch index from `night_start_utc` (gaps allowed; missing epochs are fine).
- `act` — watch's raw activity count (`|Δ magnitude|` sum over the epoch).
- `hr` — representative HR in bpm; **`0` = no reading**.

Response: a `sessions[]` array plus a night `summary`. Each session has
`session_start_utc`/`session_end_utc` (detected sleep, trimmed to real sleep —
not the capture window), `stages[]` (contiguous intervals), and per-session
`metrics` (total sleep, efficiency, latency, WASO, per-stage minutes).

**Multiple sessions:** an awakening up to ~2 h is absorbed into one session and
shows up as WASO. Only a *very long* gap (> 120 min of sustained movement, e.g. a
nap hours apart from the main sleep) splits the night into **two or more
sessions**. Either way no sleep is ever discarded — splitting just produces more
sessions; the `bridge_gap_epochs` config knob controls this granularity. The
phone writes one Health Connect `SleepSessionRecord` per session.

```json
{ "night_start_utc": ..., "tz_offset_min": 0,
  "sessions": [ {"session_start_utc": ..., "session_end_utc": ..., "stages": [...], "metrics": {...}} ],
  "summary": {"session_count": 1, "total_sleep_min": 442.5, "awake_min": 0, "light_min": ..., "deep_min": ..., "rem_min": ...} }
```

## Algorithm (handoff doc §7)

Pipeline over the epoch series, all thresholds in [`app/config.py`](app/config.py):

1. **Sleep period(s)** (`fusion/hr_period.py`) — every sustained low-movement
   block is a sleep envelope (stillness persists through REM); HR settling below
   a nightly resting baseline confirms onset. Gaps under `bridge_gap_epochs`
   (120 min) are bridged into one session; longer gaps yield separate sessions.
2. **Wake within sleep** (`fusion/actigraphy.py`) — Cole-Kripke (1992) weighted
   window over activity counts.
3. **Staging** (`fusion/staging.py`) — Deep = HR near night floor + sustained low
   movement + low HR variability; REM = low movement + HR above the deep floor +
   elevated HR variability, weighted toward later cycles; Light = the rest.
4. **Smoothing** (`fusion/smoothing.py`) — minimum bout lengths absorb
   physiologically implausible fragments (e.g. isolated 30 s REM).

> ⚠️ **Calibration.** The `|Δ-mag|` sum has a large per-night noise floor (~2000
> milli-g of 50 Hz jitter even when still), so preprocessing subtracts a
> per-night floor (`ActigraphyParams.floor_percentile`) and `count_scale` rescales
> the *movement excess*. `count_scale=0.001` is calibrated against the PhysioNet
> sleep-accel dataset (see below); **re-verify on real PT2 data** — its milli-g
> scale may differ. REM remains best-effort by design (BPM-only, no RR/HRV).

## Validation against real PSG-labeled data

`tests/test_realdata.py` grades the pipeline against the **PhysioNet sleep-accel**
dataset (Walch et al. 2019) — real Apple Watch accel + PPG HR with gold-standard
polysomnography labels. `tests/build_real_fixture.py` computes the *same* epoch
feature the watch will log (floor-subtracted `|Δ-mag|` sum + median HR) from the
raw files and stores compact ground-truth fixtures in `tests/fixtures/`.

Sleep/wake agreement (3 subjects, 2198 scored epochs), `count_scale=0.001`:

| subject | accuracy | sleep sensitivity | wake specificity |
|---------|---------:|------------------:|-----------------:|
| 46343   | 0.909 | 0.972 | 0.513 |
| 8000685 | 0.871 | 0.871 | 0.879 |
| 1455390 | 0.790 | 0.809 | 0.583 |
| **pooled** | **0.841** | **0.857** | **0.628** |

This matches the published actigraphy literature: sleep is detected well, wake
poorly (quiet wake looks like sleep without training). **Stage (Deep/REM)
accuracy is *not* validated here** — in this lab data HR barely differs between
wake and sleep (subjects lie still throughout), the BPM-only ceiling the handoff
doc flags. Stage accuracy needs real PT2 nights vs subjective logs.

To rebuild fixtures (raw files are gitignored under `realdata/`):
```bash
base=https://physionet.org/files/sleep-accel/1.0.0
for id in 46343 8000685 1455390; do
  curl -s $base/motion/${id}_acceleration.txt    -o realdata/${id}_acceleration.txt
  curl -s $base/heart_rate/${id}_heartrate.txt   -o realdata/${id}_heartrate.txt
  curl -s $base/labels/${id}_labeled_sleep.txt   -o realdata/${id}_labeled_sleep.txt
done
.venv/bin/python -m tests.build_real_fixture 46343 8000685 1455390
```

## Develop & test (no hardware needed)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest httpx
.venv/bin/python -m pytest          # synthetic + real-data tests
.venv/bin/uvicorn app.main:app --reload   # local dev server on :8000
```

`tests/synth.py` generates plausible nights on the real milli-g scale (noise
floor + movement spikes, cyclic deep/light/REM, BT-drop simulation, and
injectable mid-night wakes) so the whole pipeline is exercised offline.

## Deploy (Synology)

```bash
docker compose up -d --build       # exposes host port 8765 -> container 8000
```
Mount `./data` for the SQLite store. Keep it on the LAN / behind your existing
reverse proxy or VPN, same as Vikunja. Tune `ports`/proxy to taste.
