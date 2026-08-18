"""Deterministic tests for production frame-source semantics."""

from __future__ import annotations

import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.capture import (  # noqa: E402
    CaptureState,
    LiveFrameSource,
    RecordedFrameSource,
    TimestampDomain,
)


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        timestamps_ms: list[float] | None = None,
        fps: float = 0.0,
        opened: bool = True,
        failure_delay: float = 0.0,
    ) -> None:
        self.frames = frames
        self.timestamps_ms = timestamps_ms or [0.0] * len(frames)
        self.fps = fps
        self.opened = opened
        self.failure_delay = failure_delay
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            if self.failure_delay:
                time.sleep(self.failure_delay)
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps_ms[max(0, self.index - 1)]
        return 0.0

    def set(self, _property_id: int, _value: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class RecordedFrameSourceTests(unittest.TestCase):
    def test_preserves_order_and_uses_media_timestamps(self) -> None:
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        capture = FakeCapture(frames, timestamps_ms=[0.0, 250.0, 500.0], fps=30.0)
        source = RecordedFrameSource(
            "recording.mp4",
            source_id="recording-01",
            capture_factory=lambda _: capture,
            utc_now=lambda: datetime(2026, 8, 5, tzinfo=timezone.utc),
            monotonic=lambda: 100.0,
        ).start()

        packets = [source.read(), source.read(), source.read()]

        self.assertEqual([packet.sequence for packet in packets if packet], [0, 1, 2])
        self.assertEqual(
            [packet.timestamp_seconds for packet in packets if packet],
            [0.0, 0.25, 0.5],
        )
        self.assertTrue(
            all(packet.timestamp_domain is TimestampDomain.MEDIA_TIME for packet in packets if packet)
        )
        self.assertEqual([int(packet.frame[0, 0, 0]) for packet in packets if packet], [1, 2, 3])
        self.assertIsNone(source.read())
        self.assertEqual(source.health.state, CaptureState.EOF)
        self.assertEqual(source.health.frames_received, 3)
        source.stop()
        self.assertTrue(capture.released)

    def test_uses_fps_when_backend_timestamp_does_not_advance(self) -> None:
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        capture = FakeCapture(frames, timestamps_ms=[0.0, 0.0, 0.0], fps=2.0)
        source = RecordedFrameSource(
            "recording.mp4",
            source_id="recording-01",
            capture_factory=lambda _: capture,
        ).start()

        timestamps = [source.read().timestamp_seconds for _ in range(3)]  # type: ignore[union-attr]

        self.assertEqual(timestamps, [0.0, 0.5, 1.0])
        source.stop()

    def test_fails_if_no_deterministic_media_clock_exists(self) -> None:
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        capture = FakeCapture(frames, timestamps_ms=[0.0, 0.0], fps=0.0)
        source = RecordedFrameSource(
            "recording.mp4",
            source_id="recording-01",
            capture_factory=lambda _: capture,
        ).start()

        self.assertIsNotNone(source.read())
        with self.assertRaisesRegex(RuntimeError, "media timestamps"):
            source.read()
        self.assertEqual(source.health.state, CaptureState.FAILED)
        source.stop()


class LiveFrameSourceTests(unittest.TestCase):
    def test_exposes_latest_frame_with_live_timestamp_and_frame_age(self) -> None:
        frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (1, 2)]
        first_capture = FakeCapture(frames, failure_delay=0.05)
        closed_capture = FakeCapture([], opened=False)
        captures = iter([first_capture])

        def factory(_: int | str) -> FakeCapture:
            return next(captures, closed_capture)

        source = LiveFrameSource(
            0,
            source_id="camera-01",
            reconnect_seconds=1.0,
            reconnect_max_seconds=1.0,
            reconnect_jitter_ratio=0.0,
            capture_factory=factory,
            utc_now=lambda: datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
            monotonic=lambda: 100.0,
        ).start()
        deadline = time.monotonic() + 1.0
        while source.health.frames_received < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        packet = source.read(timeout=0.2)

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.sequence, 1)
        self.assertEqual(int(packet.frame[0, 0, 0]), 2)
        self.assertEqual(packet.timestamp_domain, TimestampDomain.UTC_EPOCH)
        self.assertEqual(source.frame_age_seconds(103.5), 3.5)
        self.assertTrue(source.is_stale(3.0, 103.5))
        self.assertFalse(source.is_stale(4.0, 103.5))

        packet.frame[:] = 99
        fresh_copy = source.read(timeout=0)
        self.assertIsNotNone(fresh_copy)
        assert fresh_copy is not None
        self.assertEqual(int(fresh_copy.frame[0, 0, 0]), 2)
        source.stop()
        self.assertEqual(source.health.state, CaptureState.STOPPED)

    def test_reconnect_backoff_is_exponential_capped_and_injectable(self) -> None:
        source = LiveFrameSource(
            0,
            source_id="camera-01",
            reconnect_seconds=1.0,
            reconnect_max_seconds=5.0,
            reconnect_jitter_ratio=0.2,
            random_value=lambda: 0.5,
        )
        self.assertEqual(source._reconnect_delay(1), 1.0)
        self.assertEqual(source._reconnect_delay(2), 2.0)
        self.assertEqual(source._reconnect_delay(3), 4.0)
        self.assertEqual(source._reconnect_delay(20), 5.0)

        low_jitter = LiveFrameSource(
            0,
            source_id="camera-02",
            reconnect_seconds=2.0,
            reconnect_max_seconds=10.0,
            reconnect_jitter_ratio=0.25,
            random_value=lambda: 0.0,
        )
        self.assertEqual(low_jitter._reconnect_delay(1), 1.5)

    def test_reconnect_re_resolves_a_secret_backed_source(self) -> None:
        frame = np.full((2, 2, 3), 7, dtype=np.uint8)
        captures = iter([
            FakeCapture([], opened=False),
            FakeCapture([frame], failure_delay=0.05),
        ])
        opened_sources: list[int | str] = []

        def factory(source: int | str) -> FakeCapture:
            opened_sources.append(source)
            return next(captures, FakeCapture([], opened=False))

        source = LiveFrameSource(
            "rtsp://old-camera/stream",
            source_id="camera-01",
            reconnect_seconds=0,
            reconnect_max_seconds=0,
            reconnect_jitter_ratio=0,
            capture_factory=factory,
            source_resolver=lambda: "rtsp://new-camera/stream",
        ).start()
        deadline = time.monotonic() + 1.0
        while source.health.frames_received < 1 and time.monotonic() < deadline:
            time.sleep(0.005)

        packet = source.read(timeout=0.2)
        source.stop()

        self.assertIsNotNone(packet)
        self.assertGreaterEqual(len(opened_sources), 2)
        self.assertEqual(opened_sources[:2], [
            "rtsp://old-camera/stream",
            "rtsp://new-camera/stream",
        ])

    def test_reconnect_without_resolver_reuses_original_source(self) -> None:
        captures = iter([
            FakeCapture([], opened=False),
            FakeCapture([], opened=False),
        ])
        opened_sources: list[int | str] = []

        def factory(source: int | str) -> FakeCapture:
            opened_sources.append(source)
            return next(captures, FakeCapture([], opened=False))

        source = LiveFrameSource(
            0,
            source_id="webcam-01",
            reconnect_seconds=0.01,
            reconnect_max_seconds=0.01,
            reconnect_jitter_ratio=0,
            capture_factory=factory,
        ).start()
        deadline = time.monotonic() + 1.0
        while len(opened_sources) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        source.stop()

        self.assertGreaterEqual(len(opened_sources), 2)
        self.assertEqual(opened_sources[:2], [0, 0])


if __name__ == "__main__":
    unittest.main()
