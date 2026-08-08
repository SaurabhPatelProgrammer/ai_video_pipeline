"""Startup reconciliation and consistency-check crash-recovery tests.

An event must never become permanently inconsistent with its evidence file,
whether the process crashed mid-write or a file was lost outside the service's
control.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.application.service import reconcile_startup  # noqa: E402
from scoop_ai.cli import main as cli_main  # noqa: E402
from scoop_ai.storage import (  # noqa: E402
    EvidenceRecord,
    EventRecord,
    SessionRecord,
    SQLiteEventRepository,
)

NOW = "2026-08-05T12:00:00+00:00"
FUTURE_DEADLINE = "2026-08-19T12:00:00+00:00"


def _health_event_count(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return connection.execute("SELECT COUNT(*) FROM health_events").fetchone()[0]
    finally:
        connection.close()


class ReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database_path = root / "events.sqlite3"
        self.evidence_root = root / "evidence"
        self.evidence_root.mkdir()
        self.repository = SQLiteEventRepository(self.database_path)
        self.repository.start_session(
            SessionRecord(session_id="session-1", camera_id="camera-1", started_at=NOW)
        )

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary.cleanup()

    def _write_file(self, relative_path: str, content: bytes = b"jpeg-bytes") -> Path:
        path = self.evidence_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _insert_event(self, event_id: str, evidence_path: str) -> None:
        self.repository.insert_event(
            EventRecord(
                event_id=event_id,
                session_id="session-1",
                camera_id="camera-1",
                event_type="scoop_deposited_candidate",
                occurred_at=NOW,
                evidence_path=evidence_path,
                review_state="unreviewed",
            )
        )

    def test_orphan_referenced_by_event_is_recovered(self) -> None:
        # A crash between insert_event() and register_evidence() leaves a file
        # and an event row on disk/DB with no evidence_artifacts row yet.
        self._insert_event("event-1", "session-1/event-1.jpg")
        self._write_file("session-1/event-1.jpg")

        summary = reconcile_startup(self.repository, self.evidence_root)

        self.assertEqual(summary["orphans_recovered"], ["session-1/event-1.jpg"])
        self.assertEqual(summary["orphans_deleted"], [])
        self.assertEqual(summary["missing_flagged"], [])
        [record] = self.repository.list_all_evidence()
        self.assertEqual(record.relative_path, "session-1/event-1.jpg")
        self.assertEqual(record.integrity_status, "valid")

    def test_orphan_not_referenced_by_any_event_is_deleted(self) -> None:
        stray = self._write_file("stray/leftover.jpg")

        summary = reconcile_startup(self.repository, self.evidence_root)

        self.assertEqual(summary["orphans_deleted"], ["stray/leftover.jpg"])
        self.assertEqual(summary["orphans_recovered"], [])
        self.assertFalse(stray.exists())
        self.assertEqual(self.repository.list_all_evidence(), [])

    def test_missing_evidence_file_flags_integrity_event_and_health(self) -> None:
        self._insert_event("event-1", "session-1/event-1.jpg")
        self.repository.register_evidence(
            EvidenceRecord(
                evidence_id="evidence-1",
                event_id="event-1",
                relative_path="session-1/event-1.jpg",
                sha256="c" * 64,
                size_bytes=10,
                media_type="image/jpeg",
                created_at=NOW,
                retention_deadline=FUTURE_DEADLINE,
            )
        )
        # No file written on disk: it was lost outside the service's control.
        before_health = _health_event_count(self.database_path)

        summary = reconcile_startup(self.repository, self.evidence_root)

        self.assertEqual(summary["missing_flagged"], ["session-1/event-1.jpg"])
        [record] = self.repository.list_all_evidence()
        self.assertEqual(record.integrity_status, "missing")
        event = self.repository.get_event("event-1")
        assert event is not None
        self.assertEqual(event.review_state, "needs_review")
        self.assertTrue(event.metadata["evidence_missing"])
        self.assertEqual(_health_event_count(self.database_path), before_health + 1)

    def test_soft_deleted_evidence_is_not_flagged_missing(self) -> None:
        # Regression: a file legitimately removed by the retention job must
        # not be treated as an integrity failure on every future restart.
        self._insert_event("event-1", "session-1/event-1.jpg")
        self.repository.register_evidence(
            EvidenceRecord(
                evidence_id="evidence-1",
                event_id="event-1",
                relative_path="session-1/event-1.jpg",
                sha256="d" * 64,
                size_bytes=10,
                media_type="image/jpeg",
                created_at=NOW,
                retention_deadline=NOW,
            )
        )
        self.repository.mark_evidence_deleted("evidence-1", reason="retention_policy")
        before_health = _health_event_count(self.database_path)

        summary = reconcile_startup(self.repository, self.evidence_root)

        self.assertEqual(summary["missing_flagged"], [])
        self.assertEqual(_health_event_count(self.database_path), before_health)
        event = self.repository.get_event("event-1")
        assert event is not None
        self.assertEqual(event.review_state, "unreviewed")

    def test_reconcile_is_idempotent_across_repeated_runs(self) -> None:
        self._insert_event("event-1", "session-1/event-1.jpg")
        self._write_file("session-1/event-1.jpg")

        first = reconcile_startup(self.repository, self.evidence_root)
        second = reconcile_startup(self.repository, self.evidence_root)

        self.assertEqual(first["orphans_recovered"], ["session-1/event-1.jpg"])
        self.assertEqual(second["orphans_recovered"], [])
        self.assertEqual(second["missing_flagged"], [])
        self.assertEqual(len(self.repository.list_all_evidence()), 1)

    def test_missing_root_directory_is_a_safe_no_op(self) -> None:
        summary = reconcile_startup(self.repository, self.evidence_root / "does-not-exist")
        self.assertEqual(summary, {"orphans_recovered": [], "orphans_deleted": [], "missing_flagged": []})


class ConsistencyCheckCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database_path = root / "events.sqlite3"
        self.evidence_root = root / "evidence"
        self.evidence_root.mkdir()
        with SQLiteEventRepository(self.database_path) as repository:
            repository.start_session(
                SessionRecord(session_id="session-1", camera_id="camera-1", started_at=NOW)
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_cli(self) -> tuple[int, dict]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = cli_main(
                [
                    "consistency-check",
                    "--database",
                    str(self.database_path),
                    "--evidence-root",
                    str(self.evidence_root),
                ]
            )
        return exit_code, json.loads(buffer.getvalue())

    def test_reports_ok_when_disk_and_database_agree(self) -> None:
        content = b"jpeg-bytes"
        path = self.evidence_root / "session-1" / "event-1.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)

        with SQLiteEventRepository(self.database_path) as repository:
            repository.register_evidence(
                EvidenceRecord(
                    evidence_id="evidence-1",
                    relative_path="session-1/event-1.jpg",
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                    media_type="image/jpeg",
                    created_at=NOW,
                    retention_deadline=FUTURE_DEADLINE,
                )
            )

        exit_code, report = self._run_cli()
        self.assertEqual(exit_code, 0)
        self.assertEqual(report, {"status": "ok", "orphans": [], "missing": [], "corrupt": []})

    def test_reports_failure_for_orphan_missing_and_corrupt_evidence(self) -> None:
        orphan = self.evidence_root / "stray.jpg"
        orphan.write_bytes(b"untracked")

        corrupt_path = self.evidence_root / "corrupt.jpg"
        corrupt_path.write_bytes(b"tampered-bytes")

        with SQLiteEventRepository(self.database_path) as repository:
            repository.register_evidence(
                EvidenceRecord(
                    evidence_id="evidence-missing",
                    relative_path="gone.jpg",
                    sha256="e" * 64,
                    size_bytes=5,
                    media_type="image/jpeg",
                    created_at=NOW,
                    retention_deadline=FUTURE_DEADLINE,
                )
            )
            repository.register_evidence(
                EvidenceRecord(
                    evidence_id="evidence-corrupt",
                    relative_path="corrupt.jpg",
                    sha256="f" * 64,
                    size_bytes=999,
                    media_type="image/jpeg",
                    created_at=NOW,
                    retention_deadline=FUTURE_DEADLINE,
                )
            )

        exit_code, report = self._run_cli()
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["orphans"], ["stray.jpg"])
        self.assertEqual(report["missing"], ["gone.jpg"])
        self.assertEqual(report["corrupt"], ["corrupt.jpg"])

    def test_soft_deleted_evidence_does_not_fail_consistency_check(self) -> None:
        with SQLiteEventRepository(self.database_path) as repository:
            repository.register_evidence(
                EvidenceRecord(
                    evidence_id="evidence-1",
                    relative_path="session-1/event-1.jpg",
                    sha256="a" * 64,
                    size_bytes=10,
                    media_type="image/jpeg",
                    created_at=NOW,
                    retention_deadline=NOW,
                )
            )
            repository.mark_evidence_deleted("evidence-1", reason="retention_policy")

        exit_code, report = self._run_cli()
        self.assertEqual(exit_code, 0)
        self.assertEqual(report, {"status": "ok", "orphans": [], "missing": [], "corrupt": []})


if __name__ == "__main__":
    unittest.main()
