"""Phase 9 silent-pilot reporting and safety tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.config import ConfigurationError, load_service_config  # noqa: E402
from scoop_ai.operations import generate_pilot_report  # noqa: E402
from scoop_ai.storage import (  # noqa: E402
    EventRecord,
    HealthEventRecord,
    ReviewRecord,
    SessionRecord,
    SQLiteEventRepository,
    TelemetryRecord,
)


class Phase9Tests(unittest.TestCase):
    def test_pilot_report_contains_exact_operational_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            repository = SQLiteEventRepository(database)
            try:
                repository.start_session(SessionRecord(
                    session_id="session-1", camera_id="cam-01",
                    started_at="2026-08-08T00:00:00+00:00",
                ))
                repository.insert_event(EventRecord(
                    event_id="event-1", session_id="session-1", camera_id="cam-01",
                    event_type="deposit_confirmed", occurred_at="2026-08-08T00:01:00+00:00",
                    confidence=0.9,
                ))
                repository.add_review(ReviewRecord(
                    review_id="review-1", event_id="event-1", decision="accepted",
                    reviewer_id="operator", reviewed_at="2026-08-08T00:02:00+00:00",
                ))
                repository.record_telemetry(TelemetryRecord(
                    telemetry_id="telemetry-1", camera_id="cam-01",
                    observed_at="2026-08-08T00:01:00+00:00", fps=10.0,
                    blur_variance=40.0, changed_fraction=0.1, accepted=True,
                ))
                repository.record_health_event(HealthEventRecord(
                    health_event_id="health-1", camera_id="cam-01", component="capture",
                    state="healthy", occurred_at="2026-08-08T00:01:00+00:00",
                ))
            finally:
                repository.close()
            report = generate_pilot_report(database, camera_id="cam-01")
            self.assertEqual(report["candidate_events"], 1)
            self.assertEqual(report["manual_reviews"], 1)
            self.assertEqual(report["review_agreement_rate"], 1.0)
            self.assertEqual(report["frame_quality"]["accepted_frames"], 1)
            self.assertEqual(report["silent_pilot"], True)

    def test_service_config_rejects_billing_or_automatic_exports(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "service.toml"
            path.write_text(
                "[service]\nname='pilot'\nenvironment='test'\nartifact_root='artifacts'\n"
                "billing_mode='enabled'\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_service_config(path)


if __name__ == "__main__":
    unittest.main()
