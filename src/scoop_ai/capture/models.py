"""Shared value objects for capture sources."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np


class TimestampDomain(str, Enum):
    """Meaning of :attr:`FramePacket.timestamp_seconds`.

    Live streams use a UTC Unix timestamp recorded when the frame is received;
    camera-origin capture time is not available from OpenCV. Recorded files use
    media time relative to the beginning of the file. Keeping the domain on the
    packet prevents those two clocks from being compared accidentally.
    """

    UTC_EPOCH = "utc_epoch"
    MEDIA_TIME = "media_time"


class CaptureState(str, Enum):
    """Operational state reported by a frame source."""

    CREATED = "created"
    CONNECTING = "connecting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    EOF = "eof"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A frame plus the clocks required for deterministic event processing."""

    source_id: str
    sequence: int
    frame: np.ndarray = field(repr=False, compare=False)
    timestamp_seconds: float
    timestamp_domain: TimestampDomain
    observed_at_utc: datetime
    observed_monotonic_seconds: float

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if self.frame.ndim not in (2, 3) or self.frame.size == 0:
            raise ValueError("frame must be a non-empty image array")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if not math.isfinite(self.observed_monotonic_seconds):
            raise ValueError("observed_monotonic_seconds must be finite")
        if self.observed_at_utc.tzinfo is None:
            raise ValueError("observed_at_utc must be timezone-aware")

    @property
    def media_timestamp_seconds(self) -> float | None:
        """Return recorded-media time without conflating it with wall time."""

        if self.timestamp_domain is TimestampDomain.MEDIA_TIME:
            return self.timestamp_seconds
        return None

    @property
    def observed_timestamp_seconds(self) -> float:
        """Return the UTC observation time as Unix seconds."""

        return self.observed_at_utc.timestamp()

    def copy(self) -> "FramePacket":
        """Return a packet whose image buffer is safe for caller mutation."""

        return FramePacket(
            source_id=self.source_id,
            sequence=self.sequence,
            frame=self.frame.copy(),
            timestamp_seconds=self.timestamp_seconds,
            timestamp_domain=self.timestamp_domain,
            observed_at_utc=self.observed_at_utc,
            observed_monotonic_seconds=self.observed_monotonic_seconds,
        )


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    """Immutable health snapshot suitable for logs and health endpoints."""

    source_id: str
    state: CaptureState
    detail: str
    frames_received: int
    consecutive_failures: int
    reconnect_attempts: int
    updated_at_utc: datetime
    last_frame_at_utc: datetime | None = None
    last_frame_monotonic_seconds: float | None = None
    last_sequence: int | None = None
    frames_per_second: float = 0.0

    @property
    def heartbeat_at_utc(self) -> datetime:
        """Time of the most recent state/health update."""

        return self.updated_at_utc

    def frame_age_seconds(self, now_monotonic_seconds: float) -> float | None:
        """Return monotonic frame age, or ``None`` before the first frame."""

        if self.last_frame_monotonic_seconds is None:
            return None
        return max(0.0, now_monotonic_seconds - self.last_frame_monotonic_seconds)

    @classmethod
    def initial(cls, source_id: str) -> "CaptureHealth":
        now = datetime.now(timezone.utc)
        return cls(
            source_id=source_id,
            state=CaptureState.CREATED,
            detail="source created",
            frames_received=0,
            consecutive_failures=0,
            reconnect_attempts=0,
            updated_at_utc=now,
        )
