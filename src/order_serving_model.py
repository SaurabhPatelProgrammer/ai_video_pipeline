"""Order-level container-preparation to customer-handoff motion baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ServedOrderState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class ServedOrderCandidate:
    timestamp: float
    preparation_started_at: float
    handoff_activity_started_at: float
    tub_motion: float
    serving_motion: float


class ServedOrderStateMachine:
    """Emit one review candidate after preparation followed by handoff activity."""

    def __init__(
        self,
        *,
        tub_threshold: float,
        serving_threshold: float,
        minimum_loading_frames: int = 3,
        minimum_serving_frames: int = 2,
        minimum_preparation_seconds: float = 20.0,
        order_timeout_seconds: float = 45.0,
        cooldown_seconds: float = 20.0,
        nominal_fps: float = 12.0,
    ) -> None:
        values = (tub_threshold, serving_threshold, minimum_preparation_seconds,
                  order_timeout_seconds, cooldown_seconds, nominal_fps)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("served-order parameters must be finite")
        if tub_threshold <= 0 or serving_threshold <= 0:
            raise ValueError("motion thresholds must be positive")
        if minimum_loading_frames < 1 or minimum_serving_frames < 1:
            raise ValueError("minimum frame counts must be positive")
        if minimum_preparation_seconds < 0:
            raise ValueError("minimum_preparation_seconds cannot be negative")
        if order_timeout_seconds <= minimum_preparation_seconds:
            raise ValueError("order timeout must exceed minimum preparation time")
        if cooldown_seconds < 0 or nominal_fps <= 0:
            raise ValueError("cooldown must be non-negative and FPS positive")
        self.tub_threshold = tub_threshold
        self.serving_threshold = serving_threshold
        self.minimum_loading_frames = minimum_loading_frames
        self.minimum_serving_frames = minimum_serving_frames
        self.minimum_preparation_seconds = minimum_preparation_seconds
        self.order_timeout_seconds = order_timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self.nominal_fps = nominal_fps
        self.state = ServedOrderState.IDLE
        self.loading_streak = 0
        self.serving_streak = 0
        self.preparation_started_at = 0.0
        self.handoff_activity_started_at = 0.0
        self.cooldown_until = 0.0
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.state = ServedOrderState.IDLE
        self.loading_streak = 0
        self.serving_streak = 0
        self.preparation_started_at = 0.0
        self.handoff_activity_started_at = 0.0
        self.cooldown_until = 0.0
        self.last_timestamp = None

    def update(
        self,
        tub_motion: float,
        serving_motion: float,
        timestamp: float,
    ) -> ServedOrderCandidate | None:
        if not all(math.isfinite(value) and value >= 0 for value in (tub_motion, serving_motion, timestamp)):
            raise ValueError("motion and timestamp values must be finite and non-negative")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self.last_timestamp = timestamp
        tub_active = tub_motion >= self.tub_threshold
        serving_active = serving_motion >= self.serving_threshold

        if self.state is ServedOrderState.COOLDOWN:
            if timestamp < self.cooldown_until:
                return None
            self.state = ServedOrderState.IDLE
            self.loading_streak = 0

        if self.state is ServedOrderState.IDLE:
            self.loading_streak = self.loading_streak + 1 if tub_active else 0
            if self.loading_streak >= self.minimum_loading_frames:
                self.state = ServedOrderState.PREPARING
                self.preparation_started_at = timestamp - (
                    self.minimum_loading_frames - 1
                ) / self.nominal_fps
                self.serving_streak = 0
            return None

        age = timestamp - self.preparation_started_at
        if age > self.order_timeout_seconds:
            self.state = ServedOrderState.IDLE
            self.loading_streak = 1 if tub_active else 0
            self.serving_streak = 0
            return None
        if age < self.minimum_preparation_seconds:
            self.serving_streak = 0
            return None
        if serving_active:
            if self.serving_streak == 0:
                self.handoff_activity_started_at = timestamp
            self.serving_streak += 1
        else:
            self.serving_streak = 0
        if self.serving_streak < self.minimum_serving_frames:
            return None
        candidate = ServedOrderCandidate(
            timestamp=timestamp,
            preparation_started_at=self.preparation_started_at,
            handoff_activity_started_at=self.handoff_activity_started_at,
            tub_motion=tub_motion,
            serving_motion=serving_motion,
        )
        self.state = ServedOrderState.COOLDOWN
        self.cooldown_until = timestamp + self.cooldown_seconds
        self.loading_streak = 0
        self.serving_streak = 0
        return candidate
