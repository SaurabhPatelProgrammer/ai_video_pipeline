"""Integration tests for the new production-facing adapters and local UI."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.request import urlopen

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.application.service import _StatusEndpoint  # noqa: E402
from scoop_ai.inference import Detection, SupervisionByteTrackAdapter  # noqa: E402
from scoop_ai.operations import HealthRegistry, HealthState, MetricsRegistry  # noqa: E402
from scoop_ai.review.app import run_review_app  # noqa: E402
from scoop_ai.storage import SQLiteEventRepository  # noqa: E402
from scoop_ai.windows_service import (  # noqa: E402
    ScoopAIWindowsService,
    _load_settings,
)


class _FakeTracker:
    def update_with_detections(self, detections):
        detections.tracker_id = np.arange(100, 100 + len(detections))
        return detections


class ProductionAdapterTests(unittest.TestCase):
    def test_windows_service_class_is_module_level_and_bom_settings_load(self) -> None:
        self.assertEqual(ScoopAIWindowsService.__module__, "scoop_ai.windows_service")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for key in ("service_config", "camera_config", "checkpoint_manifest"):
                path = root / f"{key}.json"
                path.write_text("{}", encoding="utf-8")
                paths[key] = str(path)
            settings = root / "settings.json"
            settings.write_text(json.dumps(paths), encoding="utf-8-sig")
            old_value = os.environ.get("SCOOP_AI_WINDOWS_SERVICE_SETTINGS")
            os.environ["SCOOP_AI_WINDOWS_SERVICE_SETTINGS"] = str(settings)
            try:
                self.assertEqual(_load_settings(), paths)
            finally:
                if old_value is None:
                    os.environ.pop("SCOOP_AI_WINDOWS_SERVICE_SETTINGS", None)
                else:
                    os.environ["SCOOP_AI_WINDOWS_SERVICE_SETTINGS"] = old_value

    def test_tracker_preserves_names_and_attaches_ids(self) -> None:
        tracker = SupervisionByteTrackAdapter(tracker=_FakeTracker())
        tracked = tracker.update(
            [Detection("loaded_scoop", 0.9, (1.0, 2.0, 5.0, 8.0))],
            timestamp=1.0,
        )
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].track_id, 100)
        self.assertEqual(tracked[0].class_name, "loaded_scoop")

    def test_loopback_health_and_metrics_endpoint(self) -> None:
        health = HealthRegistry(required_components={"capture"})
        health.report("capture", HealthState.HEALTHY, "ok")
        metrics = MetricsRegistry()
        metrics.increment("frames_total", 2)
        endpoint = _StatusEndpoint("127.0.0.1", 0, health, metrics)
        endpoint.start()
        try:
            port = endpoint.server.server_address[1]
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertTrue(payload["live"])
            with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                payload = json.loads(response.read())
                self.assertEqual(payload["counters"]["frames_total"], 2)
        finally:
            endpoint.stop()

    @unittest.skipUnless(os.name == "nt", "PySide production target is Windows")
    def test_review_window_constructs_against_empty_database(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            with SQLiteEventRepository(database):
                pass
            original_exec = QApplication.exec
            QApplication.exec = lambda self: 0  # type: ignore[method-assign]
            try:
                self.assertEqual(run_review_app(database), 0)
            finally:
                QApplication.exec = original_exec  # type: ignore[method-assign]


if __name__ == "__main__":
    unittest.main()
