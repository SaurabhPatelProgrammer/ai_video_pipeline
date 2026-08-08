"""Phase 11 final production-gate readiness tests."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations import generate_readiness_report  # noqa: E402
from scoop_ai.storage import ModelVersionRecord, SQLiteEventRepository  # noqa: E402


class Phase11Tests(unittest.TestCase):
    def _configs(self, root: Path) -> tuple[Path, Path]:
        calibration = root / "calibration.json"
        calibration.write_text("{}", encoding="utf-8")
        service = root / "service.toml"
        service.write_text(
            "[service]\nname='pilot'\nenvironment='test'\nartifact_root='artifacts'\n"
            "billing_mode='disabled'\nautomatic_exports=false\n",
            encoding="utf-8",
        )
        camera = root / "camera.toml"
        camera.write_text(
            "[camera]\ncamera_id='cam-01'\nenabled=true\nmode='recorded'\nsource='video.mp4'\n"
            f"analysis_fps=10.0\ncalibration_profile='{calibration.as_posix()}'\n"
            "[zones]\ntub=[[0.1,0.1],[0.9,0.1],[0.9,0.9]]\n"
            "serving=[[0.1,0.1],[0.9,0.1],[0.9,0.9]]\n",
            encoding="utf-8",
        )
        return service, camera

    def test_missing_external_evidence_blocks_readiness(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            service, camera = self._configs(root)
            report = generate_readiness_report(
                service_config_path=service, camera_config_path=camera,
                database_path=root / "events.sqlite3",
            )
            self.assertFalse(report["ready"])
            self.assertIn("representative_dataset_present", report["blocking_failures"])
            self.assertIn("backup_drill_evidence", report["blocking_failures"])

    def test_all_supplied_gate_evidence_reports_ready(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            service, camera = self._configs(root)
            dataset = root / "dataset.manifest.json"
            backup = root / "verified.sqlite3"
            dataset.write_text("{}", encoding="utf-8")
            backup.write_bytes(b"verified backup")
            database = root / "events.sqlite3"
            repository = SQLiteEventRepository(database)
            try:
                repository.register_model(ModelVersionRecord(
                    model_version="model-v1", model_name="rfdetr-nano",
                    checkpoint_sha256=hashlib.sha256(b"model").hexdigest(),
                    created_at="2026-08-08T00:00:00+00:00",
                    approved_at="2026-08-08T00:00:00+00:00",
                ))
                repository.promote_model(
                    camera_id="cam-01", model_version="model-v1",
                    manifest_sha256=hashlib.sha256(b"model").hexdigest(),
                    approved_by="reviewer", changed_at="2026-08-08T00:00:00+00:00",
                )
            finally:
                repository.close()
            report = generate_readiness_report(
                service_config_path=service, camera_config_path=camera,
                database_path=database, dataset_path=dataset, backup_path=backup,
            )
            self.assertTrue(report["ready"])
            self.assertEqual(report["blocking_failures"], [])


if __name__ == "__main__":
    unittest.main()
