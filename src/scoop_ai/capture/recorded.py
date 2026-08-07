"""Ordered, deterministic frame capture for recorded video files."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ._health import HealthTracker
from ._opencv import CaptureFactory, VideoCaptureLike, default_capture_factory
from .models import CaptureHealth, CaptureState, FramePacket, TimestampDomain


class RecordedFrameSource:
    """Read a recording in order without wall-clock throttling or frame drops."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str,
        capture_factory: CaptureFactory | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id cannot be empty")
        self.path = Path(path)
        self.source_id = source_id
        self._capture_factory = capture_factory or default_capture_factory
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._capture: VideoCaptureLike | None = None
        self._sequence = -1
        self._fps = 0.0
        self._last_media_timestamp = -1.0
        self._health = HealthTracker(source_id)

    @property
    def health(self) -> CaptureHealth:
        return self._health.snapshot()

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> "RecordedFrameSource":
        if self._capture is not None:
            return self
        self._health.transition(CaptureState.CONNECTING, "opening recorded source")
        capture = self._capture_factory(str(self.path))
        if not capture.isOpened():
            capture.release()
            self._health.transition(
                CaptureState.FAILED,
                "recorded source could not be opened",
                failure=True,
            )
            raise RuntimeError(f"Could not open recorded source: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps = fps if math.isfinite(fps) and fps > 0 else 0.0
        self._capture = capture
        self._health.transition(CaptureState.HEALTHY, "recorded source ready")
        return self

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise RuntimeError("RecordedFrameSource.start() must be called before read()")
        if self.health.state is CaptureState.EOF:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._health.transition(CaptureState.EOF, "end of recorded source")
            return None

        self._sequence += 1
        media_timestamp = self._resolve_media_timestamp()
        observed_at = self._utc_now()
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        packet = FramePacket(
            source_id=self.source_id,
            sequence=self._sequence,
            frame=frame,
            timestamp_seconds=media_timestamp,
            timestamp_domain=TimestampDomain.MEDIA_TIME,
            observed_at_utc=observed_at,
            observed_monotonic_seconds=self._monotonic(),
        )
        self._health.transition(
            CaptureState.HEALTHY,
            "reading recorded frames",
            frame_received=True,
            at_utc=observed_at,
            at_monotonic=packet.observed_monotonic_seconds,
        )
        return packet

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._health.transition(CaptureState.STOPPED, "source stopped")

    def __enter__(self) -> "RecordedFrameSource":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _resolve_media_timestamp(self) -> float:
        assert self._capture is not None
        reported_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        reported = reported_ms / 1000.0
        fallback = self._sequence / self._fps if self._fps > 0 else None

        if math.isfinite(reported) and reported >= 0:
            # Some backends report 0 for every frame. From frame two onward,
            # prefer the FPS-derived clock when it advances deterministically.
            if self._sequence == 0 or reported > self._last_media_timestamp:
                resolved = reported
            elif fallback is not None and fallback > self._last_media_timestamp:
                resolved = fallback
            else:
                self._fail_timestamp()
        elif fallback is not None and fallback > self._last_media_timestamp:
            resolved = fallback
        else:
            self._fail_timestamp()

        self._last_media_timestamp = resolved
        return resolved

    def _fail_timestamp(self) -> None:
        self._health.transition(
            CaptureState.FAILED,
            "recorded source has no usable media clock",
            failure=True,
        )
        raise RuntimeError(
            "Recorded source has neither increasing media timestamps nor a valid FPS"
        )
