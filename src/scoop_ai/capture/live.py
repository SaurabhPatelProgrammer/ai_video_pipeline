"""Low-latency latest-frame capture for live cameras and network streams."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from ._health import HealthTracker
from ._opencv import CaptureFactory, open_live_capture
from .models import CaptureHealth, CaptureState, FramePacket, TimestampDomain

LOGGER = logging.getLogger(__name__)


class LiveFrameSource:
    """Continuously read a live source and expose only its newest frame.

    Slow inference intentionally drops old frames. Recorded files must use
    :class:`RecordedFrameSource`, which preserves every frame and media time.
    """

    def __init__(
        self,
        source: int | str,
        *,
        source_id: str,
        reconnect_seconds: float = 3.0,
        reconnect_max_seconds: float = 30.0,
        reconnect_jitter_ratio: float = 0.20,
        rtsp_transport: str = "tcp",
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        capture_factory: CaptureFactory | None = None,
        source_resolver: Callable[[], int | str] | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        random_value: Callable[[], float] | None = None,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id cannot be empty")
        if reconnect_seconds < 0:
            raise ValueError("reconnect_seconds cannot be negative")
        if reconnect_max_seconds < reconnect_seconds:
            raise ValueError("reconnect_max_seconds cannot be below reconnect_seconds")
        if not 0 <= reconnect_jitter_ratio <= 1:
            raise ValueError("reconnect_jitter_ratio must be between 0 and 1")
        if rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("capture timeouts must be positive")
        self.source = source
        self.source_id = source_id
        self.reconnect_seconds = reconnect_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self.reconnect_jitter_ratio = reconnect_jitter_ratio
        self.rtsp_transport = rtsp_transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self._capture_factory = capture_factory
        self._source_resolver = source_resolver
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._random_value = random_value or random.random
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._sequence = -1
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._health = HealthTracker(source_id)

    @property
    def health(self) -> CaptureHealth:
        return self._health.snapshot()

    def start(self) -> "LiveFrameSource":
        if self._thread is not None:
            if self._thread.is_alive():
                return self
            raise RuntimeError("a stopped LiveFrameSource cannot be restarted")
        self._health.transition(CaptureState.CONNECTING, "opening live source")
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"capture-{self.source_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def read(
        self,
        after_sequence: int = -1,
        timeout: float = 2.0,
    ) -> FramePacket | None:
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._latest is not None
                    and self._latest.sequence != after_sequence
                )
                or self._stopping.is_set(),
                timeout=timeout,
            )
            if self._latest is None or self._latest.sequence == after_sequence:
                return None
            return self._latest.copy()

    def stop(self) -> None:
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            join_timeout = max(
                1.0,
                self.open_timeout_ms / 1000.0 + self.read_timeout_ms / 1000.0 + 1.0,
            )
            self._thread.join(timeout=join_timeout)
        self._health.transition(CaptureState.STOPPED, "source stopped")

    def frame_age_seconds(self, now_monotonic_seconds: float | None = None) -> float | None:
        """Return age of the latest frame using the monotonic process clock."""

        now = self._monotonic() if now_monotonic_seconds is None else now_monotonic_seconds
        return self.health.frame_age_seconds(now)

    def is_stale(
        self,
        maximum_age_seconds: float,
        now_monotonic_seconds: float | None = None,
    ) -> bool:
        if maximum_age_seconds <= 0:
            raise ValueError("maximum_age_seconds must be positive")
        age = self.frame_age_seconds(now_monotonic_seconds)
        return age is None or age > maximum_age_seconds

    def _reader_loop(self) -> None:
        attempted_open = False
        while not self._stopping.is_set():
            reconnecting = attempted_open
            self._health.transition(
                CaptureState.CONNECTING,
                "opening live source",
                reconnect=reconnecting,
            )
            attempted_open = True
            capture = None
            try:
                current_source = self._source_for_attempt(reconnecting)
                capture = open_live_capture(
                    current_source,
                    rtsp_transport=self.rtsp_transport,
                    open_timeout_ms=self.open_timeout_ms,
                    read_timeout_ms=self.read_timeout_ms,
                    capture_factory=self._capture_factory,
                )
                if not capture.isOpened():
                    self._mark_failure("live source could not be opened")
                    continue

                while not self._stopping.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        self._mark_failure("live frame read failed")
                        break
                    observed_at = self._utc_now()
                    if observed_at.tzinfo is None:
                        observed_at = observed_at.replace(tzinfo=timezone.utc)
                    observed_at = observed_at.astimezone(timezone.utc)
                    observed_monotonic = self._monotonic()
                    self._sequence += 1
                    packet = FramePacket(
                        source_id=self.source_id,
                        sequence=self._sequence,
                        frame=frame,
                        timestamp_seconds=observed_at.timestamp(),
                        timestamp_domain=TimestampDomain.UTC_EPOCH,
                        observed_at_utc=observed_at,
                        observed_monotonic_seconds=observed_monotonic,
                    )
                    with self._condition:
                        self._latest = packet
                        self._condition.notify_all()
                    self._health.transition(
                        CaptureState.HEALTHY,
                        "receiving live frames",
                        frame_received=True,
                        at_utc=observed_at,
                        at_monotonic=observed_monotonic,
                        sequence=self._sequence,
                    )
            except Exception as exc:  # keep the capture worker observable and alive
                LOGGER.exception("Capture failure for %s", self.source_id)
                self._mark_failure(f"capture error: {type(exc).__name__}: {exc}")
            finally:
                if capture is not None:
                    capture.release()

            if not self._stopping.is_set():
                self._stopping.wait(
                    self._reconnect_delay(self.health.consecutive_failures)
                )

    def _source_for_attempt(self, reconnecting: bool) -> int | str:
        """Refresh a secret-backed source before reconnecting to a live camera."""

        if not reconnecting or self._source_resolver is None:
            return self.source
        try:
            resolved = self._source_resolver()
        except Exception as exc:
            # Do not include the original exception text: third-party credential
            # backends may embed the secret source in their error message.
            raise RuntimeError(
                f"source resolver failed with {type(exc).__name__}"
            ) from None
        if isinstance(resolved, bool) or not isinstance(resolved, (int, str)):
            raise RuntimeError("source resolver returned an invalid source type")
        if isinstance(resolved, str) and not resolved.strip():
            raise RuntimeError("source resolver returned an empty source")
        return resolved

    def _mark_failure(self, detail: str) -> None:
        self._health.transition(
            CaptureState.DEGRADED,
            detail,
            failure=True,
        )

    def _reconnect_delay(self, consecutive_failures: int) -> float:
        """Calculate capped exponential delay with bounded symmetric jitter."""

        # The cap keeps exponentiation bounded even after a camera has been
        # offline for days and accumulated thousands of failed attempts.
        exponent = min(30, max(0, consecutive_failures - 1))
        base = min(
            self.reconnect_max_seconds,
            self.reconnect_seconds * (2**exponent),
        )
        centered_random = min(1.0, max(0.0, self._random_value())) * 2.0 - 1.0
        jittered = base * (1.0 + centered_random * self.reconnect_jitter_ratio)
        return min(self.reconnect_max_seconds, max(0.0, jittered))
