"""End-to-end coverage for the run_service main loop.

The service loop was previously exercised by no test at all, which allowed a
NameError on the telemetry write path to survive a full green suite. These
tests drive run_service over a scripted recorded session with injected
detector, tracker, and capture doubles so that every statement between frame
ingestion and the SQLite commit is executed in-process.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.application import service as service_module  # noqa: E402
from scoop_ai.application.service import run_service  # noqa: E402
from scoop_ai.capture import RecordedFrameSource  # noqa: E402
from scoop_ai.inference import Detection, expected_reviewer_signature  # noqa: E402
from scoop_ai.inference.checkpoint_manifest import CheckpointManifest  # noqa: E402
from scoop_ai.storage import SQLiteEventRepository  # noqa: E402

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Normalized geometry mirrors tests/test_deposit_fsm.py so the scripted
# sequence is known to drive the calibrated FSM to a confirmed deposit.
CONTAINER_XYXY = (0.45 * FRAME_WIDTH, 0.45 * FRAME_HEIGHT, 0.55 * FRAME_WIDTH, 0.60 * FRAME_HEIGHT)
NEAR_XYXY = (0.40 * FRAME_WIDTH, 0.42 * FRAME_HEIGHT, 0.48 * FRAME_WIDTH, 0.50 * FRAME_HEIGHT)
FAR_XYXY = (0.05 * FRAME_WIDTH, 0.05 * FRAME_HEIGHT, 0.12 * FRAME_WIDTH, 0.12 * FRAME_HEIGHT)
OUTSIDE_XYXY = (0.82 * FRAME_WIDTH, 0.75 * FRAME_HEIGHT, 0.90 * FRAME_WIDTH, 0.85 * FRAME_HEIGHT)

# FAR sits inside the tub polygon; NEAR does not. The container centre sits
# inside the serving polygon.
TUB_ZONE = "[[0.0,0.0],[0.30,0.0],[0.30,0.30],[0.0,0.30]]"
SERVING_ZONE = "[[0.35,0.35],[0.70,0.35],[0.70,0.70],[0.35,0.70]]"

# One completed deposit: enter tub, load, approach, release, withdraw.
SCRIPT: tuple[tuple[str, tuple[float, float, float, float]] | None, ...] = (
    ("scoop", FAR_XYXY),
    ("loaded_scoop", FAR_XYXY),
    ("loaded_scoop", NEAR_XYXY),
    ("loaded_scoop", NEAR_XYXY),
    ("scoop", NEAR_XYXY),
    ("scoop", NEAR_XYXY),
    ("scoop", FAR_XYXY),
    ("scoop", FAR_XYXY),
)


def _scalar(database: Path, query: str) -> object:
    """Read one value and close the handle so Windows can remove the file."""
    connection = sqlite3.connect(database)
    try:
        return connection.execute(query).fetchone()[0]
    finally:
        connection.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _Capture:
    """Deterministic VideoCapture double reporting a stable 10 FPS clock."""

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.index = 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
        return True

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        # A mostly static scene with one moving marker keeps the frame-quality
        # gate satisfied (low changed fraction) while staying non-uniform
        # enough to clear the blur variance floor.
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[::4, ::4] = 210
        offset = self.index * 3
        frame[100 + offset : 140 + offset, 100:140] = 255
        self.index += 1
        return True, frame

    def get(self, property_id: int) -> float:
        import cv2

        if property_id == cv2.CAP_PROP_FPS:
            return 10.0
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return (self.index - 1) * 100.0
        return 0.0

    def release(self) -> None:
        return None


class _ScriptedDetector:
    """Emits the scripted deposit sequence keyed by frame order."""

    class _Manifest:
        confidence_threshold = 0.35
        model_version = "model-v1"
        architecture = "nano"
        checkpoint_sha256 = "0" * 64
        dataset_version = "dataset-v1"
        classes = ("scoop", "loaded_scoop", "serving_container")
        input_resolution = 320

    def __init__(self, _manifest_path: Path, **_kwargs: object) -> None:
        self.manifest = self._Manifest()
        self.calls = 0

    def predict(self, frame: object, timestamp: float) -> list[Detection]:
        step = SCRIPT[self.calls] if self.calls < len(SCRIPT) else None
        self.calls += 1
        detections = [Detection("serving_container", 0.95, CONTAINER_XYXY, track_id=10)]
        if step is not None:
            class_name, box = step
            detections.append(Detection(class_name, 0.90, box, track_id=20))
        return detections


class _HandoverDetector:
    class _Manifest:
        confidence_threshold = 0.10
        model_version = "handover-model-v1"
        architecture = "nano"
        checkpoint_sha256 = "0" * 64
        dataset_version = "handover-dataset-v1"
        classes = ("ice_cream_item",)
        input_resolution = 320

    _boxes = (FAR_XYXY, FAR_XYXY, CONTAINER_XYXY, CONTAINER_XYXY, CONTAINER_XYXY, OUTSIDE_XYXY)

    def __init__(self, _manifest_path: Path, **_kwargs: object) -> None:
        self.manifest = self._Manifest()
        self.calls = 0

    def predict(self, frame: object, timestamp: float) -> list[Detection]:
        box = self._boxes[min(self.calls, len(self._boxes) - 1)]
        self.calls += 1
        return [Detection("ice_cream_item", 0.90, box, track_id=30)]


class _PassthroughTracker:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def update(self, detections, timestamp: float):
        return detections


class ServiceLoopTests(unittest.TestCase):
    def _write_configs(self, root: Path) -> tuple[Path, Path, Path]:
        video = root / "session.mp4"
        video.write_bytes(b"scripted-recorded-session")

        checkpoint = root / "model-v1.pth"
        checkpoint.write_bytes(b"model-v1")
        manifest_model = CheckpointManifest(
            schema_version=1,
            model_family="rfdetr",
            architecture="nano",
            classes=("scoop", "loaded_scoop", "serving_container"),
            checkpoint_file=checkpoint.name,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            dataset_version="dataset-v1",
            input_resolution=320,
            confidence_threshold=0.35,
            model_version="model-v1",
            approved_by="reviewer-a",
        )
        payload = {
            "schema_version": manifest_model.schema_version,
            "model_family": manifest_model.model_family,
            "architecture": manifest_model.architecture,
            "classes": list(manifest_model.classes),
            "checkpoint_file": manifest_model.checkpoint_file,
            "checkpoint_sha256": manifest_model.checkpoint_sha256,
            "dataset_version": manifest_model.dataset_version,
            "input_resolution": manifest_model.input_resolution,
            "confidence_threshold": manifest_model.confidence_threshold,
            "model_version": manifest_model.model_version,
            "approved_by": manifest_model.approved_by,
        }
        payload["reviewer_signature"] = expected_reviewer_signature(manifest_model)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        artifact_root = root / "data"
        service = root / "service.toml"
        service.write_text(
            "[service]\n"
            "name = 'scoop-counter-test'\n"
            "environment = 'test'\n"
            f"artifact_root = '''{artifact_root}'''\n"
            "log_level = 'ERROR'\n"
            "health_host = '127.0.0.1'\n"
            f"health_port = {_free_port()}\n"
            "shutdown_timeout_seconds = 5.0\n"
            "approved_reviewers = ['reviewer-a']\n"
            "billing_mode = 'disabled'\n"
            "automatic_exports = false\n"
            "\n[export]\nenabled = false\n"
            "\n[alerts]\npoll_seconds = 60.0\nstale_after_seconds = 600.0\n",
            encoding="utf-8",
        )

        camera = root / "camera.toml"
        camera.write_text(
            "[camera]\n"
            "camera_id = 'cam-01'\n"
            "enabled = true\n"
            "mode = 'recorded'\n"
            f"source = '''{video}'''\n"
            "analysis_fps = 10.0\n"
            "\n[zones]\n"
            f"tub = {TUB_ZONE}\n"
            f"serving = {SERVING_ZONE}\n"
            "\n[quality]\n"
            "minimum_blur_variance = 0.0\n"
            "maximum_changed_fraction = 1.0\n"
            "\n[event]\n"
            "minimum_confidence = 0.35\n"
            "approach_distance = 0.18\n"
            "exit_distance = 0.24\n"
            "minimum_approach_seconds = 0.10\n"
            "minimum_release_seconds = 0.08\n"
            "sequence_timeout_seconds = 8.0\n"
            "missing_tolerance_seconds = 0.35\n",
            encoding="utf-8",
        )
        return service, camera, manifest

    def _run(self, root: Path) -> tuple[int, Path]:
        service, camera, manifest = self._write_configs(root)
        capture = _Capture(len(SCRIPT))

        def recorded_factory(source, *, source_id):
            return RecordedFrameSource(
                source,
                source_id=source_id,
                capture_factory=lambda _: capture,
            )

        originals = (
            service_module.RFDETRLocalAdapter,
            service_module.SupervisionByteTrackAdapter,
            service_module.RecordedFrameSource,
        )
        service_module.RFDETRLocalAdapter = _ScriptedDetector
        service_module.SupervisionByteTrackAdapter = _PassthroughTracker
        service_module.RecordedFrameSource = recorded_factory
        try:
            exit_code = run_service(
                service_config_path=service,
                camera_config_path=camera,
                checkpoint_manifest_path=manifest,
            )
        finally:
            (
                service_module.RFDETRLocalAdapter,
                service_module.SupervisionByteTrackAdapter,
                service_module.RecordedFrameSource,
            ) = originals
        return exit_code, root / "data"

    def _run_handover(self, root: Path) -> tuple[int, Path]:
        service, camera, manifest = self._write_configs(root)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_data["classes"] = ["ice_cream_item"]
        manifest_data["model_version"] = "handover-model-v1"
        manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
        camera.write_text(
            camera.read_text(encoding="utf-8").replace(
                "mode = 'recorded'", "mode = 'recorded'\npipeline = 'handover'"
            ),
            encoding="utf-8",
        )
        capture = _Capture(len(_HandoverDetector._boxes))

        def recorded_factory(source, *, source_id):
            return RecordedFrameSource(
                source,
                source_id=source_id,
                capture_factory=lambda _: capture,
            )

        originals = (
            service_module.RFDETRLocalAdapter,
            service_module.ProximityTrackerAdapter,
            service_module.RecordedFrameSource,
        )
        service_module.RFDETRLocalAdapter = _HandoverDetector
        service_module.ProximityTrackerAdapter = _PassthroughTracker
        service_module.RecordedFrameSource = recorded_factory
        try:
            exit_code = run_service(
                service_config_path=service,
                camera_config_path=camera,
                checkpoint_manifest_path=manifest,
            )
        finally:
            (
                service_module.RFDETRLocalAdapter,
                service_module.ProximityTrackerAdapter,
                service_module.RecordedFrameSource,
            ) = originals
        return exit_code, root / "data"

    def test_recorded_session_runs_to_eof_and_commits_event_and_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            exit_code, artifact_root = self._run(root)

            # A NameError or any other loop failure sets exit code 1.
            self.assertEqual(exit_code, 0)

            database = artifact_root / "database" / "events.sqlite3"
            self.assertTrue(database.is_file())
            with SQLiteEventRepository(database) as repository:
                events = repository.list_events(limit=50)
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event.camera_id, "cam-01")
                self.assertEqual(event.event_type, "scoop_deposited_candidate")
                self.assertEqual(event.review_state, "unreviewed")
                self.assertEqual(event.container_track_id, 10)
                self.assertEqual(event.scoop_track_id, 20)
                self.assertEqual(event.model_version, "model-v1")
                self.assertIsNotNone(event.evidence_path)

            # The telemetry write is the statement that previously raised
            # NameError before any frame could be processed.
            self.assertGreater(
                _scalar(database, "SELECT COUNT(*) FROM pilot_telemetry"), 0
            )

            evidence = artifact_root / "evidence" / event.evidence_path
            self.assertTrue(evidence.is_file())
            self.assertEqual(
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
                event.evidence_sha256,
            )

    def test_session_is_recorded_as_completed_not_failed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            exit_code, artifact_root = self._run(root)
            self.assertEqual(exit_code, 0)

            database = artifact_root / "database" / "events.sqlite3"
            self.assertEqual(_scalar(database, "SELECT COUNT(*) FROM sessions"), 1)
            self.assertEqual(
                _scalar(database, "SELECT status FROM sessions"), "completed"
            )

    def test_handover_model_uses_canonical_service_and_persists_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            exit_code, artifact_root = self._run_handover(root)
            self.assertEqual(exit_code, 0)

            database = artifact_root / "database" / "events.sqlite3"
            with SQLiteEventRepository(database) as repository:
                [event] = repository.list_events(limit=10)
                self.assertEqual(event.event_type, "ice_cream_handover_candidate")
                self.assertEqual(event.model_version, "handover-model-v1")
                self.assertEqual(event.metadata["item_track_id"], 30)
                self.assertEqual(event.metadata["route"], "pickup_to_customer")
                self.assertIsNone(event.container_track_id)
                self.assertIsNone(event.scoop_track_id)
                self.assertTrue(
                    repository.list_all_evidence()[0].relative_path.endswith(".jpg")
                )


if __name__ == "__main__":
    unittest.main()
