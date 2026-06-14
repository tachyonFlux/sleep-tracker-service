"""No-training fusion pipeline: HR-baseline sleep period + actigraphy wake +
HR/movement staging heuristics -> 4-stage hypnogram (handoff doc §7)."""

from .pipeline import run_fusion

__all__ = ["run_fusion"]
