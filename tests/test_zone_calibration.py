"""Tests for safe user-defined calibration polygons."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.calibration import (  # noqa: E402
    SceneQualityGuard,
    load_reference_fingerprint,
    save_calibrated_profile,
    validate_zone,
)


class ZoneValidationTests(unittest.TestCase):
    def test_accepts_simple_normalized_polygon(self) -> None:
        zone = validate_zone([(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)], "zone")
        self.assertEqual(len(zone), 4)

    def test_rejects_small_and_self_crossing_polygons(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            validate_zone([(0.1, 0.1), (0.2, 0.2)], "zone")
        with self.assertRaisesRegex(ValueError, "edges cross"):
            validate_zone([(0.1, 0.1), (0.8, 0.8), (0.1, 0.8), (0.8, 0.1)], "zone")

    def test_saves_calibrated_derivative_with_parent_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "profile_name": "trained-v5",
                        "tub_motion_threshold": 0.07,
                        "serving_motion_threshold": 0.02,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "user.json"
            result = save_calibrated_profile(
                base,
                output,
                container_zone=[(0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.1, 0.9)],
                customer_zone=[(0.55, 0.1), (0.9, 0.1), (0.9, 0.4), (0.55, 0.4)],
                frame_width=960,
                frame_height=1080,
                source_name="sample.mp4",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["tub_motion_threshold"], 0.07)
            self.assertEqual(
                payload["calibration"]["parent_profile_sha256"],
                hashlib.sha256(base.read_bytes()).hexdigest(),
            )
            self.assertEqual(result.profile_path, output.resolve())
            with self.assertRaisesRegex(ValueError, "already exists"):
                save_calibrated_profile(
                    base,
                    output,
                    container_zone=payload["tub_zone"],
                    customer_zone=payload["serving_zone"],
                    frame_width=960,
                    frame_height=1080,
                    source_name="sample.mp4",
                )

    def test_rejects_material_zone_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            base.write_text(
                '{"tub_motion_threshold": 0.1, "serving_motion_threshold": 0.1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                save_calibrated_profile(
                    base,
                    root / "user.json",
                    container_zone=[(0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)],
                    customer_zone=[(0.2, 0.2), (0.7, 0.2), (0.7, 0.7), (0.2, 0.7)],
                    frame_width=100,
                    frame_height=100,
                    source_name="sample.mp4",
                )

    def test_reference_fingerprint_detects_static_scene_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            base.write_text(
                '{"tub_motion_threshold": 0.1, "serving_motion_threshold": 0.1}',
                encoding="utf-8",
            )
            frame = np.zeros((100, 120, 3), dtype=np.uint8)
            frame[:, :60] = 80
            frame[:, 60:] = 180
            output = root / "calibrated.json"
            save_calibrated_profile(
                base,
                output,
                container_zone=[(0.05, 0.55), (0.45, 0.55), (0.45, 0.95), (0.05, 0.95)],
                customer_zone=[(0.55, 0.05), (0.95, 0.05), (0.95, 0.45), (0.55, 0.45)],
                frame_width=120,
                frame_height=100,
                source_name="camera",
                reference_frame=frame,
            )
            fingerprint = load_reference_fingerprint(output)
            guard = SceneQualityGuard(
                fingerprint,
                tub_zone=[(0.05, 0.55), (0.45, 0.55), (0.45, 0.95), (0.05, 0.95)],
                serving_zone=[(0.55, 0.05), (0.95, 0.05), (0.95, 0.45), (0.55, 0.45)],
                drift_threshold=0.05,
                zone_change_threshold=0.2,
                obstruction_seconds=0.0,
                pixel_change_threshold=10,
            )
            self.assertTrue(guard.assess(frame, timestamp_seconds=1.0).acceptable)
            moved = np.full_like(frame, 255)
            result = guard.assess(moved, timestamp_seconds=2.0)
            self.assertFalse(result.acceptable)
            self.assertIn("calibration_drift", result.reasons)


if __name__ == "__main__":
    unittest.main()
