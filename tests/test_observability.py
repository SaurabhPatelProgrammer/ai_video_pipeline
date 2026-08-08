"""Tests for Phase 4 health transitions, alerts and bounded metrics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations import (  # noqa: E402
    AlertMonitor,
    AlertThresholds,
    HealthRegistry,
    HealthState,
    MetricsRegistry,
)
from scoop_ai.storage import SQLiteEventRepository  # noqa: E402


class ObservabilityTests(unittest.TestCase):
    def test_stale_health_can_be_reported_as_degraded(self) -> None:
        health = HealthRegistry(required_components={"capture"})
        health.report("capture", HealthState.HEALTHY, "ok", monotonic_at=10.0)

        snapshot = health.snapshot(
            stale_after_seconds=5.0,
            monotonic_now=20.0,
            stale_state=HealthState.DEGRADED,
        )
        self.assertEqual(snapshot.state, HealthState.DEGRADED)
        self.assertTrue(snapshot.live)

    def test_alerts_are_persisted_once_and_recovery_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.sqlite3"
            with SQLiteEventRepository(database) as repository:
                health = HealthRegistry(required_components={"capture"})
                metrics = MetricsRegistry()
                health.report("capture", HealthState.DEGRADED, "camera moved")
                reader = SimpleNamespace(
                    health=SimpleNamespace(
                        consecutive_failures=0,
                        frames_per_second=8.0,
                        frame_age_seconds=lambda _now: 0.1,
                    )
                )
                monitor = AlertMonitor(
                    health,
                    metrics,
                    repository,
                    camera_id="cam-1",
                    artifact_root=root,
                    reader_supplier=lambda: reader,
                    thresholds=AlertThresholds(
                        low_disk_warning_gb=0.0,
                        low_disk_critical_gb=0.0,
                    ),
                )
                monitor.evaluate_once()
                monitor.evaluate_once()
                rows = repository._connection.execute(
                    "SELECT COUNT(*) FROM health_events WHERE component = 'capture'"
                ).fetchone()[0]
                self.assertEqual(rows, 1)

                health.report("capture", HealthState.HEALTHY, "recovered")
                monitor.evaluate_once()
                rows = repository._connection.execute(
                    "SELECT state FROM health_events WHERE component = 'capture' ORDER BY created_at"
                ).fetchall()
                self.assertEqual([row[0] for row in rows], ["degraded", "healthy"])
                self.assertGreaterEqual(metrics.snapshot()["gauges"]["camera_fps"], 0)
                self.assertIn("free_storage_bytes", metrics.snapshot()["gauges"])
                self.assertIn("database_write_lock_seconds", metrics.snapshot()["gauges"])


if __name__ == "__main__":
    unittest.main()
