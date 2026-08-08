"""Phase 6 replay determinism and controlled failure recovery tests."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.storage import EventRecord, SessionRecord, SQLiteEventRepository  # noqa: E402
from scoop_ai.testing import (  # noqa: E402
    FaultSpec,
    FaultInjector,
    FaultyCaptureSource,
    InjectedDatabaseLock,
    InjectedDiskFull,
    InjectedGPUOutOfMemory,
    fault_injected,
)
from scoop_ai.training import run_replay  # noqa: E402


class _Capture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, property_id: int) -> float:
        import cv2
        if property_id == cv2.CAP_PROP_FPS:
            return 10.0
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return (self.index - 1) * 100.0
        return 0.0

    def release(self) -> None:
        pass


class _Detector:
    class _Manifest:
        confidence_threshold = 0.35

    manifest = _Manifest()

    def predict(self, frame: object, timestamp: float):
        return []


class _Tracker:
    def update(self, detections, timestamp: float):
        return detections


class Phase6Tests(unittest.TestCase):
    def test_faults_are_typed_and_one_shot(self) -> None:
        injector = FaultInjector(FaultSpec("database_lock", after=2))
        injector.checkpoint()
        with self.assertRaises(InjectedDatabaseLock):
            injector.checkpoint()
        injector.checkpoint()

        @fault_injected("gpu_oom")
        def operation():
            return "unreachable"

        with self.assertRaises(InjectedGPUOutOfMemory):
            operation()

        @fault_injected("disk_full")
        def write():
            return True

        with self.assertRaises(InjectedDiskFull):
            write()

    def test_capture_dropout_resumes_and_latency_is_injected(self) -> None:
        source = _Capture([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)])
        faulty = FaultyCaptureSource(source, drop_every=2, latency_seconds=0.01)
        self.assertIsNotNone(faulty.read())
        self.assertIsNone(faulty.read())
        started = time.perf_counter()
        self.assertIsNotNone(faulty.read())
        self.assertGreaterEqual(time.perf_counter() - started, 0.01)

    def test_database_lock_retry_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SQLiteEventRepository(Path(directory) / "events.sqlite3")
            try:
                repository.start_session(SessionRecord(
                    session_id="session-1", camera_id="cam-01",
                    started_at="2026-08-08T00:00:00+00:00",
                ))
                event = EventRecord(
                    event_id="replay-event-1", session_id="session-1", camera_id="cam-01",
                    event_type="deposit_confirmed", occurred_at="2026-08-08T00:00:00+00:00",
                    confidence=0.9,
                )

                @fault_injected("database_lock")
                def write_once() -> bool:
                    return repository.insert_event(event)

                with self.assertRaises(InjectedDatabaseLock):
                    write_once()
                self.assertTrue(write_once())
                self.assertFalse(repository.insert_event(event))
                self.assertEqual(len(repository.list_events(limit=10)), 1)
            finally:
                repository.close()

    def test_same_video_and_configs_have_same_replay_checksum(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "session.bin"
            video.write_bytes(b"recorded-session")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            service = root / "service.toml"
            service.write_text(
                "[service]\nname='test'\nenvironment='test'\nartifact_root='artifacts'\n",
                encoding="utf-8",
            )
            camera = root / "camera.toml"
            camera.write_text(
                """[camera]
camera_id='cam-01'
enabled=true
mode='recorded'
source='session.bin'
analysis_fps=10.0
[zones]
tub=[[0.1,0.1],[0.9,0.1],[0.9,0.9]]
serving=[[0.1,0.1],[0.9,0.1],[0.9,0.9]]
[quality]
minimum_blur_variance=0.0
""",
                encoding="utf-8",
            )
            frames = [np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8) for i in range(4)]
            factory = lambda _: _Capture(frames)
            first = run_replay(video, service, camera, manifest, detector_factory=lambda _: _Detector(), tracker_factory=lambda *_: _Tracker(), capture_factory=factory)
            second = run_replay(video, service, camera, manifest, detector_factory=lambda _: _Detector(), tracker_factory=lambda *_: _Tracker(), capture_factory=factory)
            self.assertEqual(first["events_sha256"], second["events_sha256"])
            self.assertEqual(first["replay_checksum"], second["replay_checksum"])
            self.assertEqual(first["frames_processed"], 4)


if __name__ == "__main__":
    unittest.main()
