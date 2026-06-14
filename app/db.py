"""SQLite storage for raw nights + computed hypnograms.

Keeping every raw night is deliberate (doc §9, §11): it's the upgrade path from
heuristics to a trained model later. The schema stores the input payload and the
output verbatim as JSON so re-running an improved algorithm over history is just
a SELECT + recompute.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .models import HypnogramOut, NightIn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    night_start_utc INTEGER NOT NULL,
    tz_offset_min   INTEGER NOT NULL,
    epoch_seconds   INTEGER NOT NULL,
    received_utc    INTEGER NOT NULL,
    raw_json        TEXT NOT NULL,
    result_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nights_start ON nights(night_start_utc);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the single connection is usable from the
        # threadpool FastAPI runs sync handlers on; writes are serialised by the
        # GIL + SQLite's own locking, which is plenty for one user's nightly POST.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, night: NightIn, result: HypnogramOut) -> int:
        cur = self._conn.execute(
            "INSERT INTO nights (night_start_utc, tz_offset_min, epoch_seconds, "
            "received_utc, raw_json, result_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                night.night_start_utc,
                night.tz_offset_min,
                night.epoch_seconds,
                int(time.time()),
                night.model_dump_json(),
                result.model_dump_json(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 30) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, night_start_utc, tz_offset_min, result_json "
            "FROM nights ORDER BY night_start_utc DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for rid, start, tz, result_json in rows:
            out.append(
                {
                    "id": rid,
                    "night_start_utc": start,
                    "tz_offset_min": tz,
                    "result": json.loads(result_json),
                }
            )
        return out

    def close(self) -> None:
        self._conn.close()
