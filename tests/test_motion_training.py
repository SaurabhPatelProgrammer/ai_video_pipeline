"""Tests for the weak fixed-camera motion trainer."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.training import (  # noqa: E402
    train_motion_baseline,
    train_motion_baseline_from_videos,
)


class MotionBaselineTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        captures = root / "captures"
        captures.mkdir()
        sessions = [("session-a", "a" * 64), ("session-b", "b" * 64)]
        (captures / "sources.json").write_text(
            json.dumps(
                [
                    {"source_session": session, "source_sha256": digest}
                    for session, digest in sessions
                ]
            ),
            encoding="utf-8",
        )
        lines = []
        for session_index, (session, digest) in enumerate(sessions):
            directory = captures / "images" / session
            directory.mkdir(parents=True)
            for index in range(8):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                x = 20 + ((index + session_index) % 4) * 8
                cv2.rectangle(image, (x, 50), (x + 14, 75), (255, 255, 255), -1)
                cv2.rectangle(image, (30 + index, 20), (45 + index, 35), (160, 160, 160), -1)
                relative = Path("images") / session / f"{index:03d}.jpg"
                self.assertTrue(cv2.imwrite(str(captures / relative), image))
                lines.append(
                    json.dumps(
                        {
                            "image": relative.as_posix(),
                            "source_session": session,
                            "source_sha256": digest,
                            "timestamp_seconds": index * 0.5,
                        }
                    )
                )
        (captures / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        profile = root / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "profile_name": "base",
                    "working_zone": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "tub_zone": [[0.1, 0.45], [0.8, 0.45], [0.8, 0.9], [0.1, 0.9]],
                    "serving_zone": [[0.1, 0.1], [0.8, 0.1], [0.8, 0.4], [0.1, 0.4]],
                    "analysis_width": 100,
                    "pixel_difference_threshold": 10,
                    "tub_motion_threshold": 0.01,
                    "serving_motion_threshold": 0.01,
                }
            ),
            encoding="utf-8",
        )
        return captures, profile

    def test_trains_immutable_profile_and_checksum_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, profile = self._fixture(root)
            output = root / "models" / "motion-v1.json"
            result = train_motion_baseline(captures, profile, output)
            self.assertEqual(len(result.sessions), 2)
            self.assertGreaterEqual(result.tub_motion_threshold, 0.005)
            self.assertGreaterEqual(result.serving_motion_threshold, 0.005)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["training"]["production_approved"])
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(manifest["artifact_sha256"], digest)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                train_motion_baseline(captures, profile, output)

    def test_rejects_unknown_source_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, profile = self._fixture(root)
            manifest = captures / "manifest.jsonl"
            manifest.write_text(
                manifest.read_text(encoding="utf-8")
                + json.dumps(
                    {
                        "image": "images/missing.jpg",
                        "source_session": "unknown",
                        "source_sha256": "c" * 64,
                        "timestamp_seconds": 10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown session"):
                train_motion_baseline(captures, profile, root / "model.json")

    def test_accepts_legacy_manifest_when_source_level_digest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, profile = self._fixture(root)
            manifest = captures / "manifest.jsonl"
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            for record in records:
                record.pop("source_sha256")
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output = root / "legacy-model.json"
            train_motion_baseline(captures, profile, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["training"]["provenance_status"], "legacy_source_level_sha256")
            self.assertEqual(payload["training"]["legacy_records_without_source_sha256"], 16)

    def test_video_training_requires_two_distinct_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, profile = self._fixture(root)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (100, 100)
            )
            self.assertTrue(writer.isOpened())
            try:
                for index in range(5):
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    cv2.rectangle(frame, (20 + index, 50), (40 + index, 75), (255, 255, 255), -1)
                    writer.write(frame)
            finally:
                writer.release()
            with self.assertRaisesRegex(ValueError, "at least two videos"):
                train_motion_baseline_from_videos([video], profile, root / "model.json")
            with self.assertRaisesRegex(ValueError, "distinct SHA-256"):
                train_motion_baseline_from_videos([video, video], profile, root / "model.json")


if __name__ == "__main__":
    unittest.main()
