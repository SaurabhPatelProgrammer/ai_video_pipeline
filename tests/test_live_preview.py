"""The live view must reuse one capture and release it when nobody is watching."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.dashboard.product import LivePreview  # noqa: E402


class _FakeCapture:
    """Stand in for cv2.VideoCapture without opening a real device."""

    def __init__(self, *, opened: bool = True, frames: int | None = None) -> None:
        self._opened = opened
        self._frames = frames
        self.reads = 0
        self.released = threading.Event()

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV name
        return self._opened

    def read(self):
        if self._frames is not None and self.reads >= self._frames:
            return False, None
        self.reads += 1
        time.sleep(0.005)
        return True, np.full((48, 64, 3), 90, dtype=np.uint8)

    def release(self) -> None:
        self.released.set()


class LivePreviewTests(unittest.TestCase):
    def test_serves_jpeg_frames_from_a_single_capture(self) -> None:
        capture = _FakeCapture()
        opened = []

        def opener(source):
            opened.append(source)
            return capture

        preview = LivePreview(lambda: "rtsp://camera/stream", capture_opener=opener)
        self.addCleanup(preview.stop)

        for _ in range(4):
            payload, dimensions = preview.snapshot()
            self.assertTrue(payload.startswith(b"\xff\xd8"))  # JPEG start of image
            self.assertEqual(dimensions, (64, 48))

        # Four snapshots must not mean four camera connections.
        self.assertEqual(opened, ["rtsp://camera/stream"])

    def test_capture_is_released_once_nobody_asks_for_frames(self) -> None:
        capture = _FakeCapture()
        preview = LivePreview(lambda: 0, capture_opener=lambda _source: capture)
        self.addCleanup(preview.stop)
        preview.IDLE_TIMEOUT_SECONDS = 0.15

        preview.snapshot()
        self.assertTrue(capture.released.wait(timeout=5))

    def test_a_camera_that_refuses_the_connection_explains_why(self) -> None:
        preview = LivePreview(
            lambda: "rtsp://camera/stream",
            capture_opener=lambda _source: _FakeCapture(opened=False),
        )
        self.addCleanup(preview.stop)
        with self.assertRaises(ValueError) as caught:
            preview.snapshot(wait_seconds=3)
        self.assertIn("only one stream", str(caught.exception))

    def test_a_stream_that_stops_delivering_is_reported(self) -> None:
        preview = LivePreview(
            lambda: 0, capture_opener=lambda _source: _FakeCapture(frames=0)
        )
        self.addCleanup(preview.stop)
        with self.assertRaises(ValueError) as caught:
            preview.snapshot(wait_seconds=3)
        self.assertIn("stopped returning frames", str(caught.exception))

    def test_a_credential_failure_does_not_take_the_dashboard_down(self) -> None:
        def explode():
            raise RuntimeError("credential store is locked")

        preview = LivePreview(explode, capture_opener=lambda _source: _FakeCapture())
        self.addCleanup(preview.stop)
        with self.assertRaises(ValueError) as caught:
            preview.snapshot(wait_seconds=3)
        self.assertIn("credential store is locked", str(caught.exception))

    def test_stopping_releases_the_camera_immediately(self) -> None:
        capture = _FakeCapture()
        preview = LivePreview(lambda: 0, capture_opener=lambda _source: capture)
        preview.snapshot()
        preview.stop()
        self.assertTrue(capture.released.wait(timeout=5))

    def test_a_stopped_preview_reopens_on_the_next_request(self) -> None:
        captures = [_FakeCapture(), _FakeCapture()]
        preview = LivePreview(lambda: 0, capture_opener=lambda _source: captures.pop(0))
        self.addCleanup(preview.stop)

        preview.snapshot()
        preview.stop()
        preview.snapshot()
        self.assertEqual(captures, [])


if __name__ == "__main__":
    unittest.main()
