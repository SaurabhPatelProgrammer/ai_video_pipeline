"""Regression tests for detection label resolution and annotation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app import annotate, detection_class_name  # noqa: E402


class DetectionClassNameTests(unittest.TestCase):
    def test_prefers_checkpoint_aware_name_for_sparse_coco_id(self) -> None:
        model = SimpleNamespace(
            class_names=[
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
            ]
        )
        detections = SimpleNamespace(data={"class_name": np.array(["bus"])})

        self.assertEqual(detection_class_name(detections, 0, model, 6), "bus")

    def test_falls_back_to_zero_based_name_for_legacy_output(self) -> None:
        model = SimpleNamespace(class_names=["scoop", "cup"])
        detections = SimpleNamespace(data={})

        self.assertEqual(detection_class_name(detections, 0, model, 1), "cup")

    def test_annotation_filter_and_counts_use_resolved_name(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        model = SimpleNamespace(class_names=["person", "bicycle", "car"])
        detections = SimpleNamespace(
            xyxy=np.array([[10, 10, 50, 50]], dtype=float),
            confidence=np.array([0.95]),
            class_id=np.array([1]),
            data={"class_name": np.array(["person"])},
        )

        _, counts = annotate(frame, detections, model, {"person"}, 50)

        self.assertEqual(dict(counts), {"person": 1})

    def test_empty_attached_name_uses_fallback(self) -> None:
        model = SimpleNamespace(class_names={3: "custom-object"})
        detections = SimpleNamespace(data={"class_name": np.array([""])})

        self.assertEqual(
            detection_class_name(detections, 0, model, 3),
            "custom-object",
        )


if __name__ == "__main__":
    unittest.main()
