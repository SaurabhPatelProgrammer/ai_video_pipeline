"""Unit tests for virtual-line configuration and crossing counts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from object_counter import (  # noqa: E402
    crossing_total,
    filter_detections,
    parse_line,
)


class ParseLineTests(unittest.TestCase):
    def test_empty_line_uses_vertical_frame_center(self) -> None:
        self.assertEqual(parse_line("", 640, 480), (320, 0, 320, 479))

    def test_custom_line_is_parsed(self) -> None:
        self.assertEqual(parse_line("10, 20, 300, 400", 640, 480), (10, 20, 300, 400))

    def test_out_of_bounds_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside"):
            parse_line("0,0,640,479", 640, 480)

    def test_zero_length_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same point"):
            parse_line("10,10,10,10", 640, 480)


class DetectionFilterTests(unittest.TestCase):
    def test_filter_uses_checkpoint_aware_class_names(self) -> None:
        detections = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([1, 2]),
            data={"class_name": np.array(["person", "cup"])},
        )
        model = SimpleNamespace(class_names={1: "legacy-person", 2: "legacy-cup"})

        filtered = filter_detections(detections, model, {"cup"})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.data["class_name"].tolist(), ["cup"])


class CrossingCountTests(unittest.TestCase):
    def test_left_to_right_crossing_increments_total_once(self) -> None:
        line = sv.LineZone(
            start=sv.Point(50, 0),
            end=sv.Point(50, 100),
            minimum_crossing_threshold=1,
        )
        for x in (10, 20, 40, 55, 70):
            detections = sv.Detections(
                xyxy=np.array([[x, 40, x + 10, 60]], dtype=float),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
                tracker_id=np.array([7]),
            )
            line.trigger(detections)

        self.assertEqual(line.in_count, 1)
        self.assertEqual(line.out_count, 0)
        self.assertEqual(crossing_total(line), 1)


if __name__ == "__main__":
    unittest.main()
