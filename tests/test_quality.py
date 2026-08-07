"""Tests for fixed-camera frame quality checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.application.quality import FrameQualityGate  # noqa: E402


class FrameQualityGateTests(unittest.TestCase):
    def test_rejects_blurred_frame(self) -> None:
        gate = FrameQualityGate(minimum_blur_variance=10)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = gate.assess(frame)
        self.assertFalse(result.acceptable)
        self.assertIn("blurred", result.reasons)

    def test_detects_full_scene_change(self) -> None:
        gate = FrameQualityGate(
            minimum_blur_variance=0,
            pixel_change_threshold=10,
            maximum_changed_fraction=0.5,
        )
        first = np.zeros((100, 100, 3), dtype=np.uint8)
        second = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertTrue(gate.assess(first).acceptable)
        result = gate.assess(second)
        self.assertFalse(result.acceptable)
        self.assertIn("camera_moved_or_obstructed", result.reasons)

    def test_accepts_local_foreground_change(self) -> None:
        gate = FrameQualityGate(
            minimum_blur_variance=0,
            pixel_change_threshold=10,
            maximum_changed_fraction=0.5,
        )
        first = np.zeros((100, 100, 3), dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(second, (40, 40), (59, 59), (255, 255, 255), -1)
        gate.assess(first)
        self.assertTrue(gate.assess(second).acceptable)


if __name__ == "__main__":
    unittest.main()
