"""Phase 7 model approval, deployment, and rollback governance tests."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.config import CameraConfig  # noqa: E402
from scoop_ai.inference import (  # noqa: E402
    CheckpointManifest,
    expected_reviewer_signature,
    validate_model_camera_compatibility,
    verify_manifest_approval,
)
from scoop_ai.storage import ModelVersionRecord, SQLiteEventRepository  # noqa: E402


class Phase7Tests(unittest.TestCase):
    def _manifest(self, root: Path, version: str = "model-v1") -> Path:
        checkpoint = root / f"{version}.pth"
        checkpoint.write_bytes(version.encode("utf-8"))
        manifest = CheckpointManifest(
            schema_version=1,
            model_family="rfdetr",
            architecture="nano",
            classes=("scoop", "loaded_scoop", "serving_container"),
            checkpoint_file=checkpoint.name,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            dataset_version="dataset-v1",
            input_resolution=576,
            confidence_threshold=0.35,
            model_version=version,
            approved_by="reviewer-a",
        )
        payload = {
            "schema_version": manifest.schema_version,
            "model_family": manifest.model_family,
            "architecture": manifest.architecture,
            "classes": list(manifest.classes),
            "checkpoint_file": manifest.checkpoint_file,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "dataset_version": manifest.dataset_version,
            "input_resolution": manifest.input_resolution,
            "confidence_threshold": manifest.confidence_threshold,
            "model_version": manifest.model_version,
            "approved_by": manifest.approved_by,
        }
        payload["reviewer_signature"] = expected_reviewer_signature(manifest)
        output = root / f"{version}.json"
        output.write_text(json.dumps(payload), encoding="utf-8")
        return output

    def test_missing_or_wrong_reviewer_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._manifest(Path(directory))
            with self.assertRaises(ValueError):
                verify_manifest_approval(path, approved_reviewers=["other"])
            data = json.loads(path.read_text(encoding="utf-8"))
            data["reviewer_signature"] = "sha256:" + ("0" * 64)
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_manifest_approval(path, approved_reviewers=["reviewer-a"])

    def test_model_promote_and_rollback_are_persisted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._manifest(root, "model-v1")
            second = self._manifest(root, "model-v2")
            repository = SQLiteEventRepository(root / "events.sqlite3")
            try:
                for path in (first, second):
                    manifest = verify_manifest_approval(path, approved_reviewers=["reviewer-a"])
                    self.assertTrue(repository.register_model(ModelVersionRecord(
                        model_version=manifest.model_version,
                        model_name="rfdetr-nano",
                        checkpoint_sha256=manifest.checkpoint_sha256,
                        created_at="2026-08-08T00:00:00+00:00",
                        approved_at="2026-08-08T00:00:00+00:00",
                    )))
                repository.promote_model(
                    camera_id="cam-01", model_version="model-v1",
                    manifest_sha256=hashlib.sha256(b"model-v1").hexdigest(),
                    approved_by="reviewer-a", changed_at="2026-08-08T00:00:00+00:00",
                )
                repository.promote_model(
                    camera_id="cam-01", model_version="model-v2",
                    manifest_sha256=hashlib.sha256(b"model-v2").hexdigest(),
                    approved_by="reviewer-a", changed_at="2026-08-08T00:01:00+00:00",
                )
                active = repository.rollback_model(
                    camera_id="cam-01", approved_by="reviewer-a",
                    changed_at="2026-08-08T00:02:00+00:00",
                )
                self.assertEqual(active["model_version"], "model-v1")
            finally:
                repository.close()

    def test_camera_compatibility_requires_zones_and_dimensions(self) -> None:
        manifest = CheckpointManifest(
            1, "rfdetr", "nano", ("scoop", "loaded_scoop", "serving_container"),
            "model.pth", "a" * 64, "dataset-v1", 576, 0.35, "model-v1",
        )
        camera = CameraConfig(
            camera_id="cam-01", enabled=True, mode="recorded", source="x",
            source_env=None, credential_key=None, analysis_fps=10.0,
            expected_width=1920, expected_height=1080,
            tub_zone=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)),
            serving_zone=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)),
        )
        self.assertEqual(validate_model_camera_compatibility(manifest, camera), "deposit")
        incompatible = replace(camera, expected_width=320)
        with self.assertRaises(ValueError):
            validate_model_camera_compatibility(manifest, incompatible)

        handover = replace(
            manifest,
            classes=("ice_cream_item",),
            model_version="handover-v1",
        )
        self.assertEqual(validate_model_camera_compatibility(handover, camera), "handover")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_model_camera_compatibility(
                handover, replace(camera, pipeline="deposit")
            )


if __name__ == "__main__":
    unittest.main()
