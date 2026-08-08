"""Thread-safe health state used by capture implementations."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone

from .models import CaptureHealth, CaptureState


class HealthTracker:
    def __init__(self, source_id: str) -> None:
        self._lock = threading.Lock()
        self._health = CaptureHealth.initial(source_id)

    def snapshot(self) -> CaptureHealth:
        with self._lock:
            return self._health

    def transition(
        self,
        state: CaptureState,
        detail: str,
        *,
        frame_received: bool = False,
        failure: bool = False,
        reconnect: bool = False,
        at_utc: datetime | None = None,
        at_monotonic: float | None = None,
        sequence: int | None = None,
    ) -> CaptureHealth:
        now = at_utc or datetime.now(timezone.utc)
        with self._lock:
            current = self._health
            frame_rate = current.frames_per_second
            if frame_received and current.last_frame_monotonic_seconds is not None and at_monotonic is not None:
                elapsed = at_monotonic - current.last_frame_monotonic_seconds
                if elapsed > 0:
                    frame_rate = 1.0 / elapsed
            self._health = replace(
                current,
                state=state,
                detail=detail,
                frames_received=current.frames_received + int(frame_received),
                consecutive_failures=(
                    current.consecutive_failures + 1
                    if failure
                    else 0 if frame_received else current.consecutive_failures
                ),
                reconnect_attempts=current.reconnect_attempts + int(reconnect),
                updated_at_utc=now,
                last_frame_at_utc=now if frame_received else current.last_frame_at_utc,
                last_frame_monotonic_seconds=(
                    at_monotonic
                    if frame_received
                    else current.last_frame_monotonic_seconds
                ),
                last_sequence=sequence if frame_received else current.last_sequence,
                frames_per_second=frame_rate,
            )
            return self._health
