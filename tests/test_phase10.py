"""Phase 10 transactional outbox and controlled export tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations import OutboxWorker, sign_export_batch  # noqa: E402
from scoop_ai.storage import EventRecord, ReviewRecord, SessionRecord, SQLiteEventRepository  # noqa: E402


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class Phase10Tests(unittest.TestCase):
    def _repository(self, root: Path) -> SQLiteEventRepository:
        repository = SQLiteEventRepository(root / "events.sqlite3")
        repository.start_session(SessionRecord(
            session_id="session-1", camera_id="cam-01",
            started_at="2026-08-08T00:00:00+00:00",
        ))
        repository.insert_event(EventRecord(
            event_id="event-1", session_id="session-1", camera_id="cam-01",
            event_type="deposit_confirmed", occurred_at="2026-08-08T00:01:00+00:00",
            confidence=0.9,
        ))
        return repository

    def test_accepted_review_creates_transactional_outbox(self) -> None:
        with TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            try:
                repository.add_review(ReviewRecord(
                    review_id="review-1", event_id="event-1", decision="accepted",
                    reviewer_id="operator", reviewed_at="2026-08-08T00:02:00+00:00",
                ))
                rows = repository.list_outbox(states={"pending"})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].event_id, "event-1")
                self.assertEqual(rows[0].payload["event_id"], "event-1")
            finally:
                repository.close()

    def test_worker_signs_idempotent_request_and_acknowledges(self) -> None:
        with TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            calls = []

            def opener(request, timeout):
                calls.append((request, timeout))
                return _Response()

            try:
                repository.add_review(ReviewRecord(
                    review_id="review-1", event_id="event-1", decision="accepted",
                    reviewer_id="operator", reviewed_at="2026-08-08T00:02:00+00:00",
                ))
                import threading
                worker = OutboxWorker(
                    repository, endpoint="https://downstream.example/export",
                    signing_key=b"test-secret", stop_event=threading.Event(),
                    opener=opener, poll_seconds=0.01,
                )
                self.assertEqual(worker.process_once(), 1)
                self.assertEqual(repository.list_outbox(states={"acknowledged"})[0].event_id, "event-1")
                request, _ = calls[0]
                self.assertEqual(request.get_header("Idempotency-key"), "event-1")
                self.assertTrue(request.get_header("X-scoop-export-signature").startswith("hmac-sha256="))
            finally:
                repository.close()

    def test_failures_reach_dead_letter_and_manual_retry_requeues(self) -> None:
        with TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            try:
                repository.add_review(ReviewRecord(
                    review_id="review-1", event_id="event-1", decision="accepted",
                    reviewer_id="operator", reviewed_at="2026-08-08T00:02:00+00:00",
                ))
                repository.claim_outbox(limit=1)
                state = repository.mark_outbox_failure(
                    "event-1", error="network down", next_attempt_at=None, max_attempts=1,
                )
                self.assertEqual(state, "dead_letter")
                self.assertEqual(repository.retry_outbox(event_id="event-1"), 1)
                self.assertEqual(repository.list_outbox(states={"pending"})[0].attempts, 0)
            finally:
                repository.close()

    def test_batch_signature_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            try:
                repository.add_review(ReviewRecord(
                    review_id="review-1", event_id="event-1", decision="accepted",
                    reviewer_id="operator", reviewed_at="2026-08-08T00:02:00+00:00",
                ))
                event = repository.list_outbox(states={"pending"})[0]
                self.assertEqual(sign_export_batch([event], b"key"), sign_export_batch([event], b"key"))
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
