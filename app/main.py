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

from fastapi import FastAPI

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
