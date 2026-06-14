"""Request/response schemas for the /night endpoint (handoff doc §6)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Stage(str, Enum):
    AWAKE = "AWAKE"
    LIGHT = "LIGHT"
    DEEP = "DEEP"
    REM = "REM"


class Epoch(BaseModel):
    """One watch-computed epoch feature. `hr == 0` means "no reading"."""

    t: int = Field(..., ge=0, description="Epoch index from night_start_utc")
    act: int = Field(..., ge=0, description="Activity count (|delta-magnitude| sum)")
    hr: int = Field(..., ge=0, le=255, description="Representative HR in bpm; 0 = none")


class NightIn(BaseModel):
    night_start_utc: int = Field(..., description="UTC seconds at epoch index 0")
    tz_offset_min: int = Field(0, description="Local tz offset from UTC, minutes")
    epoch_seconds: int = Field(30, ge=5, le=120)
    epochs: list[Epoch] = Field(..., min_length=1)


class StageInterval(BaseModel):
    start_utc: int
    end_utc: int
    stage: Stage


class Metrics(BaseModel):
    total_sleep_min: float
    time_in_bed_min: float
    efficiency: float = Field(..., description="total_sleep / time_in_session")
    sleep_latency_min: float
    waso_min: float = Field(..., description="Wake after sleep onset")
    awake_min: float
    light_min: float
    deep_min: float
    rem_min: float


class SleepSession(BaseModel):
    """One detected sleep session. The phone writes one SleepSessionRecord per
    session to Health Connect; a long mid-night awakening produces two sessions
    rather than discarding either stretch of sleep."""

    session_start_utc: int
    session_end_utc: int
    stages: list[StageInterval]
    metrics: Metrics


class NightSummary(BaseModel):
    """Aggregate across all sessions of the night."""

    session_count: int
    total_sleep_min: float
    awake_min: float
    light_min: float
    deep_min: float
    rem_min: float


class HypnogramOut(BaseModel):
    night_start_utc: int
    tz_offset_min: int
    sessions: list[SleepSession]
    summary: NightSummary
