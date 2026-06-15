"""FastAPI entrypoint (handoff doc §6).

Endpoints:
  GET  /healthz       liveness probe
  POST /night         ingest epoch features -> fusion -> hypnogram (and store)
  GET  /nights        recent stored hypnograms (history / debugging)

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
