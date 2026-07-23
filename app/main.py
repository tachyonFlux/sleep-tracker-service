"""FastAPI entrypoint (handoff doc §6).

Endpoints:
  GET  /healthz                   liveness probe
  POST /night                     ingest epoch features -> fusion -> hypnogram (and store)
  GET  /nights                    recent stored hypnograms (history / debugging)
  GET  /nights/{id}/raw           verbatim input + diagnostic summary
  POST /nights/{id}/reprocess     re-run one stored night through current code
  POST /nights/reprocess          re-run every stored night through current code

Stored results are frozen at upload time — GET /nights never recomputes. After
a fusion change is deployed, the reprocess endpoints replay stored raw through
the new code so history reflects it too (new nights pick it up automatically).

The phone POSTs the night here over the LAN/VPN (same path it reaches Vikunja);
this service returns the hypnogram and the phone writes it to Health Connect.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import __version__
from .db import Store
from .fusion import run_fusion
from .models import HypnogramOut, NightIn

DB_PATH = os.environ.get("SLEEP_DB_PATH", "/data/sleep.db")

store: Store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = Store(DB_PATH)
    yield
    store.close()


app = FastAPI(title="Sleep Tracker Fusion Service", version=__version__, lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/night", response_model=HypnogramOut)
def post_night(night: NightIn) -> HypnogramOut:
    result = run_fusion(night)
    store.save(night, result)
    return result


@app.get("/nights")
def get_nights(limit: int = 30) -> dict:
    return {"nights": store.recent(limit=limit)}


def _summary_delta(before: dict | None, after: HypnogramOut) -> dict:
    """Compact before/after stage-minute summary for a reprocessed night."""
    b = before.get("summary") if before else None
    a = after.summary
    fields = ("total_sleep_min", "awake_min", "light_min", "deep_min", "rem_min")
    return {
        "before": {f: (b.get(f) if b else None) for f in fields},
        "after": {f: getattr(a, f) for f in fields},
    }


@app.post("/nights/{night_id}/reprocess", response_model=HypnogramOut)
def reprocess_night(night_id: int) -> HypnogramOut:
    """Re-run one stored night's raw input through the current fusion code and
    overwrite its stored hypnogram in place (raw input is untouched)."""
    raw = store.get_raw(night_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="night not found")
    result = run_fusion(NightIn(**raw))
    store.update_result(night_id, result)
    return result


@app.post("/nights/reprocess")
def reprocess_all() -> dict:
    """Re-run every stored night through the current fusion code. Returns a
    per-night before/after stage-minute diff so a redeploy's effect is visible."""
    nights = []
    for nid in store.all_ids():
        raw = store.get_raw(nid)
        if raw is None:  # deleted between listing and fetch; skip
            continue
        before = store.get_result(nid)
        result = run_fusion(NightIn(**raw))
        store.update_result(nid, result)
        nights.append({"id": nid, **_summary_delta(before, result)})
    return {"reprocessed": len(nights), "nights": nights}


@app.get("/nights/{night_id}/raw")
def get_night_raw(night_id: int, epochs: bool = True) -> dict:
    """Verbatim input payload for one night, with a diagnostic summary.

    `?epochs=false` returns only the summary (handy for a quick health check
    without dumping ~1140 epoch rows). The summary surfaces the usual reasons a
    real night yields no sessions: HR never read (all hr==0), activity counts
    that don't match the server's calibration, or large gaps in the capture.
    """
    raw = store.get_raw(night_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="night not found")

    eps = raw.get("epochs", [])
    acts = [e["act"] for e in eps]
    hrs = [e["hr"] for e in eps]
    hr_nonzero = [h for h in hrs if h > 0]
    es = raw.get("epoch_seconds", 30)
    max_t = max((e["t"] for e in eps), default=0)
    summary = {
        "n_epochs": len(eps),
        "span_hours": round(max_t * es / 3600, 2),
        "gaps": max_t + 1 - len(eps),          # missing epochs (>0 => dropouts)
        "hr": {
            "n_with_reading": len(hr_nonzero),
            "pct_with_reading": round(100 * len(hr_nonzero) / len(eps), 1) if eps else 0,
            "min": min(hr_nonzero, default=0),
            "max": max(hr_nonzero, default=0),
            "mean": round(sum(hr_nonzero) / len(hr_nonzero), 1) if hr_nonzero else 0,
        },
        "act": {
            "min": min(acts, default=0),
            "max": max(acts, default=0),
            "mean": round(sum(acts) / len(acts), 1) if acts else 0,
            "n_zero": sum(1 for a in acts if a == 0),
        },
    }
    out: dict = {"id": night_id, "summary": summary}
    if epochs:
        out["raw"] = raw
    return out
