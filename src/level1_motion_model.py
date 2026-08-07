"""Level-1 fixed-camera temporal motion baseline.

This is an explicit experimental baseline, not a replacement for the custom
RF-DETR detector. It estimates scoop-cycle candidates by observing motion first
in a configured tub zone and then in a configured serving zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MotionState(str, Enum):
    IDLE = "idle"
    LOADED_CANDIDATE = "loaded_candidate"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class MotionCandidate:
    timestamp: float
    loading_started_at: float
    tub_motion: float
    serving_motion: float


class MotionScoopStateMachine:
    """Convert calibrated zone activity into de-duplicated scoop candidates."""

    def __init__(
        self,
        tub_threshold: float,
        serving_threshold: float,
        minimum_loading_frames: int = 2,
        minimum_serving_frames: int = 2,
        minimum_transfer_seconds: float = 0.20,
        transfer_timeout_seconds: float = 5.0,
        cooldown_seconds: float = 1.0,
    ) -> None:
        if tub_threshold <= 0 or serving_threshold <= 0:
            raise ValueError("motion thresholds must be positive")
        if minimum_loading_frames < 1 or minimum_serving_frames < 1:
            raise ValueError("minimum frame counts must be positive")
        if minimum_transfer_seconds < 0:
            raise ValueError("minimum_transfer_seconds cannot be negative")
        if transfer_timeout_seconds <= minimum_transfer_seconds:
            raise ValueError("transfer timeout must exceed minimum transfer time")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.tub_threshold = tub_threshold
        self.serving_threshold = serving_threshold
        self.minimum_loading_frames = minimum_loading_frames
        self.minimum_serving_frames = minimum_serving_frames
        self.minimum_transfer_seconds = minimum_transfer_seconds
        self.transfer_timeout_seconds = transfer_timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self.state = MotionState.IDLE
        self.loading_streak = 0
        self.serving_streak = 0
        self.loading_started_at = 0.0
        self.cooldown_started_at = 0.0

    def reset(self) -> None:
        self.state = MotionState.IDLE
        self.loading_streak = 0
        self.serving_streak = 0
        self.loading_started_at = 0.0
        self.cooldown_started_at = 0.0

    def update(
        self,
        tub_motion: float,
        serving_motion: float,
        timestamp: float,
    ) -> MotionCandidate | None:
        tub_active = tub_motion >= self.tub_threshold
        serving_active = serving_motion >= self.serving_threshold

        if self.state is MotionState.IDLE:
            self.loading_streak = self.loading_streak + 1 if tub_active else 0
            if self.loading_streak >= self.minimum_loading_frames:
                self.state = MotionState.LOADED_CANDIDATE
                self.loading_started_at = timestamp
                self.serving_streak = 0
            return None

        if self.state is MotionState.LOADED_CANDIDATE:
            elapsed = timestamp - self.loading_started_at
            if elapsed > self.transfer_timeout_seconds:
                self.reset()
                return None
            eligible = elapsed >= self.minimum_transfer_seconds
            self.serving_streak = (
                self.serving_streak + 1 if eligible and serving_active else 0
            )
            if self.serving_streak < self.minimum_serving_frames:
                return None
            candidate = MotionCandidate(
                timestamp=timestamp,
                loading_started_at=self.loading_started_at,
                tub_motion=tub_motion,
                serving_motion=serving_motion,
            )
            self.state = MotionState.COOLDOWN
            self.cooldown_started_at = timestamp
            self.loading_streak = 0
            self.serving_streak = 0
            return candidate

        if timestamp - self.cooldown_started_at >= self.cooldown_seconds:
            self.state = MotionState.IDLE
            self.loading_streak = 1 if tub_active else 0
        return None
