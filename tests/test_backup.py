"""Tests for WAL-safe database backup, restore and generation cleanup."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.storage import (  # noqa: E402
    BackupError,
    EventRecord,
    GroundTruthRecord,
    ModelVersionRecord,
    ReviewRecord,
    SessionRecord,
    SQLiteEventRepository,
    create_backup,
    restore_backup,
)


NOW = "2026-08-08T12:00:00+00:00"


class DatabaseBackupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "events.sqlite3"
        self.backups = self.root / "backups"
        self.repository = SQLiteEventRepository(self.database)
        self.repository.start_session(
            SessionRecord(session_id="session-1", camera_id="camera-1", started_at=NOW)
        )
        self.repository.register_model(
            ModelVersionRecord(
                model_version="model-1",
                model_name="rfdetr-nano",
                checkpoint_sha256="a" * 64,
                created_at=NOW,
            )
        )
        self.repository.insert_event(
            EventRecord(
                event_id="event-1",
                session_id="session-1",
                camera_id="camera-1",
                event_type="scoop_deposited_candidate",
                occurred_at=NOW,
            )
        )
        self.repository.add_review(
            ReviewRecord(
                review_id="review-1",
                event_id="event-1",
                decision="accepted",
                reviewer_id="operator",
                reviewed_at=NOW,
            )
        )
        self.repository.add_ground_truth(
            GroundTruthRecord(
                ground_truth_id="truth-1",
                session_id="session-1",
                camera_id="camera-1",
                occurred_at=NOW,
                is_completed_scoop=True,
                reviewer_id="operator",
            )
        )

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary.cleanup()

    def test_backup_succeeds_while_database_is_active_and_wal_enabled(self) -> None:
        self.assertEqual(self.repository.journal_mode().lower(), "wal")
        result = create_backup(self.database, self.backups)

        self.assertTrue(result.backup_path.is_file())
        self.assertTrue(result.metadata_path.is_file())
        self.assertEqual(result.metadata["source_database_name"], "events.sqlite3")
        self.assertEqual(result.metadata["schema_version"], 2)
        self.assertEqual(
            result.metadata["record_counts"],
            {"events": 1, "event_reviews": 1, "ground_truth_events": 1, "model_versions": 1},
        )
        restored = restore_backup(result.backup_path, self.root / "restored")
        self.assertEqual(restored["integrity"], "ok")
        self.assertEqual(restored["record_counts"]["events"], 1)

    def test_corrupt_backup_is_rejected(self) -> None:
        result = create_backup(self.database, self.backups)
        with result.backup_path.open("ab") as handle:
            handle.write(b"tampered")

        with self.assertRaisesRegex(BackupError, "size|SHA-256"):
            restore_backup(result.backup_path, self.root / "restored")

    def test_incompatible_schema_version_is_rejected(self) -> None:
        result = create_backup(self.database, self.backups)
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = 999
        result.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(BackupError, "schema version"):
            restore_backup(result.backup_path, self.root / "restored")

    def test_existing_restore_target_is_never_overwritten(self) -> None:
        result = create_backup(self.database, self.backups)
        destination = self.root / "restored"
        destination.mkdir()
        target = destination / "events.sqlite3"
        target.write_bytes(b"operator-data")

        with self.assertRaisesRegex(BackupError, "overwrite"):
            restore_backup(result.backup_path, destination)
        self.assertEqual(target.read_bytes(), b"operator-data")

    def test_permission_or_invalid_directory_is_reported(self) -> None:
        result = create_backup(self.database, self.backups)
        invalid_directory = self.root / "not-a-directory"
        invalid_directory.write_bytes(b"file")

        with self.assertRaises(BackupError):
            create_backup(self.database, invalid_directory)
        with self.assertRaises(BackupError):
            restore_backup(result.backup_path, invalid_directory)

    def test_cleanup_retains_configured_number_of_generations(self) -> None:
        create_backup(self.database, self.backups, retain_generations=10)
        create_backup(self.database, self.backups, retain_generations=10)
        create_backup(self.database, self.backups, retain_generations=2)

        self.assertEqual(len(list(self.backups.glob("*.sqlite3"))), 2)
        self.assertEqual(len(list(self.backups.glob("*.metadata.json"))), 2)


if __name__ == "__main__":
    unittest.main()
