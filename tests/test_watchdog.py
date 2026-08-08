"""Tests for service watchdog liveness and Windows startup validation."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations import HealthRegistry, HealthState, ServiceWatchdog  # noqa: E402
from scoop_ai.windows_service import validate_startup  # noqa: E402


class WatchdogTests(unittest.TestCase):
    def test_capture_stall_becomes_unhealthy_and_requests_stop(self) -> None:
        stop_event = threading.Event()
        health = HealthRegistry(required_components={"capture", "inference"})
        reader = SimpleNamespace(health=SimpleNamespace(frames_received=0))
        watchdog = ServiceWatchdog(
            health,
            stop_event,
            reader_supplier=lambda: reader,
            inference_completed_at=time.monotonic,
            interval_seconds=0.01,
            stale_after_seconds=0.02,
            failure_after_seconds=0.05,
        ).start()
        try:
            self.assertTrue(stop_event.wait(0.5))
            snapshot = health.snapshot()
            capture = next(item for item in snapshot.components if item.name == "capture")
            self.assertEqual(capture.state, HealthState.UNHEALTHY)
        finally:
            watchdog.stop()

    def test_startup_validation_checks_paths_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            service_config = root / "service.toml"
            camera_config = root / "camera.toml"
            checkpoint = root / "manifest.json"
            checkpoint.write_text("{}", encoding="utf-8")
            service_config.write_text(
                """[service]\nname = \"test\"\nenvironment = \"test\"\nartifact_root = \"{artifact}\"\nminimum_free_space_gb = 0.0\n""".format(
                    artifact=str(artifact_root).replace("\\", "\\\\")
                ),
                encoding="utf-8",
            )
            camera_config.write_text(
                """[camera]\ncamera_id = \"cam-1\"\nmode = \"live\"\nsource = 0\n""",
                encoding="utf-8",
            )
            result = validate_startup(
                {
                    "service_config": str(service_config),
                    "camera_config": str(camera_config),
                    "checkpoint_manifest": str(checkpoint),
                }
            )
            self.assertEqual(result["camera_id"], "cam-1")
            self.assertTrue((artifact_root / "database").is_dir())
            self.assertTrue((artifact_root / "evidence").is_dir())


if __name__ == "__main__":
    unittest.main()
