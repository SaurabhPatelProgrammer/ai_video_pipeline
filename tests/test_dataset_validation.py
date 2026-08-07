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

    def test_checksum_tampering_is_rejected(self) -> None:
        self.checkpoint.write_bytes(b"tampered")

        with self.assertRaisesRegex(ManifestValidationError, "checksum mismatch"):
            load_checkpoint_manifest(self.manifest_path)

    def test_architecture_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ManifestValidationError, "does not match"):
            load_checkpoint_manifest(
                self.manifest_path, expected_architecture="small"
            )


if __name__ == "__main__":
    unittest.main()
