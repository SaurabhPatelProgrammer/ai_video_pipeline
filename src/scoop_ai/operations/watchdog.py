"""Bounded watchdog for capture and inference worker liveness."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .health import HealthRegistry, HealthState


LOGGER = logging.getLogger("scoop-ai.watchdog")


class ServiceWatchdog:
    """Monitor worker progress and request a bounded service restart on stalls."""

    def __init__(
        self,
        health: HealthRegistry,
        stop_event: threading.Event,
        *,
        reader_supplier: Callable[[], Any | None],
        inference_completed_at: Callable[[], float],
        interval_seconds: float = 2.0,
        stale_after_seconds: float = 15.0,
        failure_after_seconds: float = 45.0,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if failure_after_seconds <= stale_after_seconds:
            raise ValueError("failure_after_seconds must exceed stale_after_seconds")
        self.health = health
        self.stop_event = stop_event
        self.reader_supplier = reader_supplier
        self.inference_completed_at = inference_completed_at
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.failure_after_seconds = failure_after_seconds
        self._monotonic = monotonic or time.monotonic
        self._thread: threading.Thread | None = None
        self._capture_stalled = False
        self._inference_stalled = False

    def start(self) -> "ServiceWatchdog":
        if self._thread is not None:
            raise RuntimeError("watchdog cannot be started twice")
        self._thread = threading.Thread(
            target=self._run,
            name="scoop-ai-watchdog",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, timeout))

    def _run(self) -> None:
        last_frames: int | None = None
        last_frame_progress = self._monotonic()
        while not self.stop_event.wait(self.interval_seconds):
            now = self._monotonic()
            reader = self.reader_supplier()
            if reader is not None:
                capture_health = reader.health
                frames = capture_health.frames_received
                sequence = getattr(capture_health, "last_sequence", None)
                progress_marker = sequence if sequence is not None else frames
                if progress_marker != last_frames:
                    last_frames = progress_marker
                    last_frame_progress = now
                    if self._capture_stalled:
                        self.health.report("capture", HealthState.HEALTHY, "capture progress recovered")
                        self._capture_stalled = False
                else:
                    age = now - last_frame_progress
                    if age > self.stale_after_seconds:
                        state = HealthState.UNHEALTHY if age > self.failure_after_seconds else HealthState.DEGRADED
                        self.health.report(
                            "capture",
                            state,
                            "capture frame sequence is not progressing",
                            details={
                                "stalled_seconds": round(age, 3),
                                "frames_received": frames,
                                "last_sequence": sequence,
                            },
                        )
                        self._capture_stalled = True
                        if state is HealthState.UNHEALTHY:
                            self.stop_event.set()

            inference_age = max(0.0, now - self.inference_completed_at())
            if inference_age > self.stale_after_seconds:
                state = HealthState.UNHEALTHY if inference_age > self.failure_after_seconds else HealthState.DEGRADED
                self.health.report(
                    "inference",
                    state,
                    "inference loop is not completing",
                    details={"stalled_seconds": round(inference_age, 3)},
                )
                self._inference_stalled = True
                if state is HealthState.UNHEALTHY:
                    self.stop_event.set()
            elif self._inference_stalled:
                self.health.report("inference", HealthState.HEALTHY, "inference progress recovered")
                self._inference_stalled = False

        LOGGER.info("service watchdog stopped")
