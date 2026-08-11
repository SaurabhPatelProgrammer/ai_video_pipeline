"""End-to-end tests for the ice-cream handover replay pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.inference import Detection  # noqa: E402
from scoop_ai.training import run_handover_replay  # noqa: E402


FRAME_WIDTH = 320
FRAME_HEIGHT = 240
VIDEO_FPS = 10.0
BOX_HALF = 16

# The item walks left (pickup) -> middle -> right (customer) -> middle. The
# per-step displacement stays well inside the proximity tracker's association
# radius so one stable track survives the whole route.
ROUTE = (0.15, 0.25, 0.30, 0.45, 0.60, 0.70, 0.75, 0.75, 0.75, 0.45, 0.45, 0.45)


class _ScriptedDetector:
    """Return boxes purely from the source timestamp, never the pixels.

    Replay writes and re-reads a real MP4, so frame content is lossy. Driving
    detections from the media clock keeps the test deterministic.
    """

    class _Manifest:
        confidence_threshold = 0.5
        model_version = "test-handover-model"

    def __init__(self, lanes: tuple[float, ...]) -> None:
        self.manifest = self._Manifest()
        self.confidence_threshold = 0.5
        self.lanes = lanes
        self.calls = 0

    def predict(self, frame: object, timestamp: float):
        del frame
        self.calls += 1
        step = min(len(ROUTE) - 1, max(0, round(timestamp * VIDEO_FPS)))
        centre_x = ROUTE[step]
        detections = []
        for centre_y in self.lanes:
            x = centre_x * FRAME_WIDTH
            y = centre_y * FRAME_HEIGHT
            detections.append(
                Detection(
                    class_name="ice_cream_item",
                    confidence=0.9,
                    xyxy=(x - BOX_HALF, y - BOX_HALF, x + BOX_HALF, y + BOX_HALF),
                )
            )
        return detections


def _write_video(path: Path, frames: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot write mp4v videos in this environment")
    try:
        for index in range(frames):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (index * 2, 4), (index * 2 + 20, 24), (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_zone_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profile_name": "test-handover-zones",
                # Pickup on the left third, customer on the right, with a
                # deliberate neutral corridor between them.
                "tub_zone": [[0.0, 0.0], [0.35, 0.0], [0.35, 1.0], [0.0, 1.0]],
                "serving_zone": [[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class HandoverReplayTests(unittest.TestCase):
    def _replay(self, directory: Path, lanes: tuple[float, ...], name: str) -> dict:
        video = directory / "session.mp4"
        if not video.exists():
            _write_video(video, len(ROUTE))
        profile = directory / "zones.json"
        if not profile.exists():
            _write_zone_profile(profile)
        manifest = directory / "model-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        detector = _ScriptedDetector(lanes)
        return run_handover_replay(
            video,
            manifest,
            profile,
            directory / name,
            analysis_fps=VIDEO_FPS,
            detector_factory=lambda _path, _confidence: detector,
        )

    def test_single_item_route_is_counted_with_evidence_and_report(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self._replay(directory, (0.5,), "replay")
            output = directory / "replay"

            self.assertEqual(report["event_count"], 1)
            self.assertEqual(report["model_version"], "test-handover-model")
            self.assertEqual(report["minimum_confidence"], 0.5)
            self.assertGreater(report["frames_processed"], 0)

            event = report["events"][0]
            self.assertEqual(event["route"], "pickup_to_customer")
            self.assertTrue((output / "annotated.mp4").is_file())
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / str(event["evidence_file"])).is_file())

            written = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(written["events_sha256"], report["events_sha256"])

    def test_every_event_sharing_one_frame_gets_its_own_evidence_image(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Two items far enough apart to survive duplicate suppression; they
            # leave the customer zone on the very same sampled frame.
            report = self._replay(directory, (0.2, 0.85), "replay")
            output = directory / "replay"

            self.assertEqual(report["event_count"], 2)
            evidence_files = [str(item["evidence_file"]) for item in report["events"]]
            self.assertEqual(len(set(evidence_files)), 2)
            for relative in evidence_files:
                self.assertTrue(
                    (output / relative).is_file(),
                    f"missing evidence image for {relative}",
                )

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._replay(directory, (0.5,), "replay")
            with self.assertRaises(FileExistsError):
                self._replay(directory, (0.5,), "replay")


if __name__ == "__main__":
    unittest.main()
