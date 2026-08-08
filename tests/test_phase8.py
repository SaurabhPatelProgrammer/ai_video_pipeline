"""Phase 8 privacy, audit, and least-privilege control tests."""

from __future__ import annotations

import io
import logging
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations.logging import JsonFormatter  # noqa: E402
from scoop_ai.security import redact_secrets  # noqa: E402
from scoop_ai.storage import AuditLogRecord, SQLiteEventRepository, redact_frame  # noqa: E402


class Phase8Tests(unittest.TestCase):
    def test_audit_logs_are_durable_and_immutable(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SQLiteEventRepository(Path(directory) / "events.sqlite3")
            try:
                record = AuditLogRecord(
                    audit_id="audit-1", occurred_at="2026-08-08T00:00:00+00:00",
                    actor="operator", action="database_backup", target="backup.zip",
                    details={"result": "ok"},
                )
                self.assertTrue(repository.record_audit(record))
                self.assertFalse(repository.record_audit(record))
                self.assertEqual(repository.list_audit_logs()[0].action, "database_backup")
                with self.assertRaises(Exception):
                    repository._connection.execute("DELETE FROM audit_logs WHERE audit_id='audit-1'")
            finally:
                repository.close()

    def test_redaction_removes_url_credentials_from_exception_logs(self) -> None:
        secret = "rtsp://operator:super-secret@camera.local:554/live"
        self.assertNotIn("super-secret", str(redact_secrets({"error": secret})))
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("phase8-test")
        logger.handlers[:] = [handler]
        logger.propagate = False
        try:
            try:
                raise RuntimeError(f"capture failed for {secret}")
            except RuntimeError:
                logger.exception("camera exception")
        finally:
            logger.handlers.clear()
        output = stream.getvalue()
        self.assertNotIn("super-secret", output)
        self.assertNotIn("operator@", output)

    def test_image_export_blurs_selected_region(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[25:75, 25:75] = 255
        frame[48:53, 48:53] = 0
        redacted = redact_frame(frame, boxes=[(0.25, 0.25, 0.75, 0.75)], blur_kernel=9)
        self.assertGreater(int(redacted[50, 50, 0]), 0)
        self.assertLess(int(redacted[50, 50, 0]), 255)
        self.assertEqual(tuple(redacted[0, 0]), (0, 0, 0))

    def test_acl_validation_script_is_present(self) -> None:
        script = PROJECT_ROOT / "scripts" / "validate-acls.ps1"
        self.assertTrue(script.is_file())
        self.assertIn("Get-Acl", script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
