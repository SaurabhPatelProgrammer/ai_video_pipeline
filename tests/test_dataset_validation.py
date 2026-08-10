"""Tests for COCO integrity, provenance, duplicate, and split checks."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.training import (  # noqa: E402
    DatasetValidationOptions,
    DatasetValidationError,
    deterministic_session_split,
    validate_dataset,
)
from scoop_ai.inference import (  # noqa: E402
    ManifestValidationError,
    RFDETRLocalAdapter,
    load_checkpoint_manifest,
)


CATEGORIES = [
    {"id": 1, "name": "scoop"},
    {"id": 2, "name": "loaded_scoop"},
    {"id": 3, "name": "serving_container"},
]


class DatasetFixture:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_split(
        self,
        split: str,
        session: str,
        *,
        content: bytes,
        bbox: list[float] | None = None,
        category_id: int | None = None,
    ) -> None:
        directory = self.root / split
        directory.mkdir(parents=True, exist_ok=True)
        image_name = f"{split}.jpg"
        (directory / image_name).write_bytes(content)
        annotations = [
            {
                "id": index,
                "image_id": 1,
                "category_id": current_category,
                "bbox": bbox or [5, 5, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
            for index, current_category in enumerate((1, 2, 3), start=1)
        ]
        if category_id is not None:
            annotations[0]["category_id"] = category_id
        data = {
            "images": [
                {
                    "id": 1,
                    "file_name": image_name,
                    "width": 100,
                    "height": 80,
                    "source_session": session,
                    "source_sha256": hashlib.sha256(
                        b"source-video:" + content
                    ).hexdigest(),
                    "timestamp_seconds": 1.5,
                }
            ],
            "annotations": annotations,
            "categories": CATEGORIES,
        }
        (directory / "_annotations.coco.json").write_text(
            json.dumps(data), encoding="utf-8"
        )


class DatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = DatasetFixture(self.root)
        self.fixture.write_split("train", "session-train", content=b"train")
        self.fixture.write_split("valid", "session-valid", content=b"valid")
        self.fixture.write_split("test", "session-test", content=b"test")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_dataset_returns_reproducible_fingerprint(self) -> None:
        first = validate_dataset(self.root)
        second = validate_dataset(self.root)

        self.assertEqual(first.fingerprint_sha256, second.fingerprint_sha256)
        self.assertEqual(first.splits["train"].class_counts["loaded_scoop"], 1)
        self.assertEqual(first.splits["valid"].sessions, ("session-valid",))

    def test_source_session_leakage_is_rejected(self) -> None:
        self.fixture.write_split("valid", "session-train", content=b"new-valid")

        with self.assertRaisesRegex(DatasetValidationError, "leaks across"):
            validate_dataset(self.root)

    def test_same_source_requires_auditable_non_overlapping_segments(self) -> None:
        train_path = self.root / "train" / "_annotations.coco.json"
        valid_path = self.root / "valid" / "_annotations.coco.json"
        train = json.loads(train_path.read_text(encoding="utf-8"))
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        valid["images"][0]["source_sha256"] = train["images"][0]["source_sha256"]
        train_path.write_text(json.dumps(train), encoding="utf-8")
        valid_path.write_text(json.dumps(valid), encoding="utf-8")

        with self.assertRaisesRegex(DatasetValidationError, "without auditable"):
            validate_dataset(self.root)

        train["images"][0].update(
            source_segment_start_seconds=0.0,
            source_segment_end_seconds=2.0,
        )
        valid["images"][0].update(
            timestamp_seconds=3.5,
            source_segment_start_seconds=3.0,
            source_segment_end_seconds=4.0,
        )
        train_path.write_text(json.dumps(train), encoding="utf-8")
        valid_path.write_text(json.dumps(valid), encoding="utf-8")

        report = validate_dataset(self.root)
        self.assertEqual(report.splits["valid"].images, 1)

    def test_overlapping_source_segments_are_rejected(self) -> None:
        train_path = self.root / "train" / "_annotations.coco.json"
        valid_path = self.root / "valid" / "_annotations.coco.json"
        train = json.loads(train_path.read_text(encoding="utf-8"))
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        valid["images"][0]["source_sha256"] = train["images"][0]["source_sha256"]
        train["images"][0].update(
            source_segment_start_seconds=0.0,
            source_segment_end_seconds=2.0,
        )
        valid["images"][0].update(
            source_segment_start_seconds=1.0,
            source_segment_end_seconds=3.0,
        )
        train_path.write_text(json.dumps(train), encoding="utf-8")
        valid_path.write_text(json.dumps(valid), encoding="utf-8")

        with self.assertRaisesRegex(DatasetValidationError, "segments overlap"):
            validate_dataset(self.root)

    def test_out_of_bounds_bbox_and_unknown_category_are_rejected(self) -> None:
        self.fixture.write_split(
            "test",
            "session-test",
            content=b"new-test",
            bbox=[95, 70, 20, 20],
            category_id=99,
        )

        with self.assertRaises(DatasetValidationError) as context:
            validate_dataset(self.root)
        message = str(context.exception)
        self.assertIn("unknown category_id", message)
        self.assertIn("outside image bounds", message)

    def test_duplicate_image_content_across_splits_is_rejected(self) -> None:
        self.fixture.write_split("valid", "session-valid", content=b"train")

        with self.assertRaisesRegex(DatasetValidationError, "duplicate image content"):
            validate_dataset(self.root)

    def test_custom_single_class_dataset_is_supported(self) -> None:
        for split in ("train", "valid", "test"):
            path = self.root / split / "_annotations.coco.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["categories"] = [{"id": 1, "name": "ice_cream_item"}]
            data["annotations"] = [data["annotations"][0]]
            path.write_text(json.dumps(data), encoding="utf-8")

        report = validate_dataset(
            self.root,
            options=DatasetValidationOptions(required_classes=("ice_cream_item",)),
        )

        self.assertEqual(report.splits["train"].class_counts, {"ice_cream_item": 1})


class DeterministicSessionSplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_session_exclusive(self) -> None:
        sessions = [f"session-{index}" for index in range(10)]

        first = deterministic_session_split(sessions, seed=7)
        second = deterministic_session_split(reversed(sessions), seed=7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(sessions))
        self.assertEqual(set(first.values()), {"train", "valid", "test"})

    def test_three_way_split_requires_three_sessions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            deterministic_session_split(["one", "two"])


class CheckpointManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "model.pth"
        self.checkpoint.write_bytes(b"local-checkpoint")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema_version": 1,
            "model_family": "rfdetr",
            "architecture": "nano",
            "classes": ["scoop", "loaded_scoop", "serving_container"],
            "checkpoint_file": "model.pth",
            "checkpoint_sha256": hashlib.sha256(b"local-checkpoint").hexdigest(),
            "dataset_version": "scoop-v1",
            "input_resolution": 576,
            "confidence_threshold": 0.35,
            "model_version": "scoop-rfdetr-nano-v1",
        }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_verifies_local_checkpoint_without_loading_model(self) -> None:
        manifest = load_checkpoint_manifest(
            self.manifest_path, expected_architecture="nano"
        )
        adapter = RFDETRLocalAdapter(self.manifest_path)

        self.assertEqual(manifest.model_version, "scoop-rfdetr-nano-v1")
        self.assertFalse(adapter.loaded)

    def test_detector_confidence_can_be_overridden_without_loading_model(self) -> None:
        adapter = RFDETRLocalAdapter(self.manifest_path, confidence_threshold=0.1)

        self.assertEqual(adapter.confidence_threshold, 0.1)
        self.assertFalse(adapter.loaded)

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            RFDETRLocalAdapter(self.manifest_path, confidence_threshold=1.1)

    def test_checksum_tampering_is_rejected(self) -> None:
        self.checkpoint.write_bytes(b"tampered")

        with self.assertRaisesRegex(ManifestValidationError, "checksum mismatch"):
            load_checkpoint_manifest(self.manifest_path)

    def test_architecture_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ManifestValidationError, "does not match"):
            load_checkpoint_manifest(
                self.manifest_path, expected_architecture="small"
            )

    def test_custom_detector_class_is_supported(self) -> None:
        self.manifest["classes"] = ["ice_cream_item"]
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

        manifest = load_checkpoint_manifest(
            self.manifest_path,
            expected_classes=("ice_cream_item",),
        )

        self.assertEqual(manifest.classes, ("ice_cream_item",))

    def test_custom_detector_class_requires_explicit_selection(self) -> None:
        self.manifest["classes"] = ["ice_cream_item"]
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

        with self.assertRaisesRegex(ManifestValidationError, "classes must be exactly"):
            load_checkpoint_manifest(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
