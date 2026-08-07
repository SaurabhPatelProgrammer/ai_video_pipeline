"""Tests for durable SQLite persistence and atomic evidence files."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.storage import (  # noqa: E402
    EvidenceRecord,
    EventConflictError,
    EventRecord,
    EvidenceWriter,
    GroundTruthRecord,
    HealthEventRecord,
    ModelVersionRecord,
    ReviewRecord,
    SessionRecord,
    SQLiteEventRepository,
)


NOW = "2026-08-05T12:00:00+00:00"


class SQLiteEventRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "events.sqlite3"
        self.repository = SQLiteEventRepository(self.path)
        self.session = SessionRecord(
            session_id="session-1",
            camera_id="camera-1",
            started_at=NOW,
            model_version="scoop-v1",
            source_name="rtsp://camera.local/stream1",
            metadata={"config_version": "shop-1-v2"},
        )
        self.repository.start_session(self.session)

    def tearDown(self) -> None:
        self.repository.close()
        self.temporary.cleanup()

    def event(self, **changes: object) -> EventRecord:
        values: dict[str, object] = {
            "event_id": "event-1",
            "session_id": "session-1",
            "camera_id": "camera-1",
            "event_type": "scoop_deposited",
            "occurred_at": "2026-08-05T12:00:01+00:00",
            "confidence": 0.93,
            "model_version": "scoop-v1",
            "metadata": {"media_timestamp_seconds": 12.5, "config_version": "shop-1-v2"},
        }
        values.update(changes)
        return EventRecord(**values)  # type: ignore[arg-type]

    def test_wal_schema_and_idempotent_event_insert(self) -> None:
        self.assertEqual(self.repository.journal_mode().lower(), "wal")
        self.assertEqual(self.repository.schema_version, 1)
        self.assertTrue(self.repository.insert_event(self.event()))
        self.assertFalse(self.repository.insert_event(self.event()))
        stored = self.repository.get_event("event-1")
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.metadata["media_timestamp_seconds"], 12.5)
        self.assertTrue(self.repository.integrity_check())

    def test_reused_event_id_with_different_payload_is_rejected(self) -> None:
        self.repository.insert_event(self.event())
        with self.assertRaises(EventConflictError):
            self.repository.insert_event(self.event(confidence=0.1))

    def test_review_is_append_only_and_updates_filter_projection(self) -> None:
        self.repository.insert_event(self.event())
        review = ReviewRecord(
            review_id="review-1",
            event_id="event-1",
            decision="accepted",
            reviewer_id="reviewer-1",
            reviewed_at="2026-08-05T12:01:00+00:00",
        )
        self.assertTrue(self.repository.add_review(review))
        self.assertEqual(len(self.repository.list_events(review_state="accepted")), 1)
        self.assertEqual(self.repository.list_reviews("event-1"), [review])
        with self.assertRaises(sqlite3.IntegrityError):
            with self.repository.transaction() as connection:
                connection.execute(
                    "UPDATE event_reviews SET decision='rejected' WHERE review_id='review-1'"
                )

    def test_model_evidence_ground_truth_and_health_audit_records(self) -> None:
        digest = "a" * 64
        self.assertTrue(
            self.repository.register_model(
                ModelVersionRecord(
                    model_version="scoop-v1",
                    model_name="rfdetr-nano",
                    checkpoint_sha256=digest,
                    created_at=NOW,
                    approved_at=NOW,
                )
            )
        )
        evidence = EvidenceRecord(
            evidence_id="evidence-1",
            relative_path="session-1/event-1.jpg",
            sha256=digest,
            size_bytes=123,
            media_type="image/jpeg",
            created_at=NOW,
            retention_deadline="2026-08-12T12:00:00+00:00",
        )
        self.assertTrue(self.repository.register_evidence(evidence))
        self.assertTrue(
            self.repository.add_ground_truth(
                GroundTruthRecord(
                    ground_truth_id="truth-1",
                    session_id="session-1",
                    camera_id="camera-1",
                    occurred_at=NOW,
                    is_completed_scoop=True,
                    reviewer_id="reviewer-1",
                    evidence_id="evidence-1",
                )
            )
        )
        self.assertTrue(
            self.repository.record_health_event(
                HealthEventRecord(
                    health_event_id="health-1",
                    component="camera",
                    state="healthy",
                    occurred_at=NOW,
                    camera_id="camera-1",
                )
            )
        )
        self.assertEqual(
            self.repository.list_expired_evidence(deadline="2026-08-13T00:00:00+00:00"),
            [evidence],
        )
        self.assertTrue(self.repository.mark_evidence_deleted("evidence-1"))
        self.assertEqual(
            self.repository.list_expired_evidence(deadline="2026-08-13T00:00:00+00:00"),
            [],
        )


class EvidenceWriterTests(unittest.TestCase):
    def test_atomic_write_returns_checksum_and_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = EvidenceWriter(Path(temporary))
            payload = b"test-image-bytes"
            artifact = writer.write_bytes("session-1", "event-1", payload)
            self.assertEqual(artifact.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(artifact.size_bytes, len(payload))
            self.assertTrue(writer.verify(artifact))
            with self.assertRaises(FileExistsError):
                writer.write_bytes("session-1", "event-1", b"replacement")
            self.assertTrue(writer.verify(artifact))
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])

    def test_unsafe_paths_and_extensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = EvidenceWriter(Path(temporary))
            with self.assertRaises(ValueError):
                writer.write_bytes("../outside", "event", b"x")
            with self.assertRaises(ValueError):
                writer.write_bytes("session", "event", b"x", extension=".exe")


if __name__ == "__main__":
    unittest.main()
