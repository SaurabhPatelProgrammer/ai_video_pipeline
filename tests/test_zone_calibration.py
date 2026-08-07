"""Tests for safe user-defined calibration polygons."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.calibration import save_calibrated_profile, validate_zone  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
