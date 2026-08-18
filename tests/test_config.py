"""Tests for strict service and camera TOML configuration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.config import (  # noqa: E402
    ConfigurationError,
    load_camera_config,
    load_service_config,
)
from scoop_ai.domain import HandoverFSMConfig  # noqa: E402


class ConfigurationTests(unittest.TestCase):
    def test_example_service_config_is_valid(self) -> None:
        config = load_service_config(PROJECT_ROOT / "configs" / "service.example.toml")

        self.assertEqual(config.name, "scoop-counter")
        self.assertEqual(config.health_port, 8080)
        self.assertEqual(config.capture.rtsp_transport, "tcp")
        self.assertEqual(config.capture.reconnect_max_seconds, 30.0)

    def test_example_camera_uses_credential_reference(self) -> None:
        config = load_camera_config(
            PROJECT_ROOT
            / "configs"
            / "cameras"
            / "shop-01-counter-01.example.toml"
        )

        self.assertEqual(config.camera_id, "shop-01-counter-01")
        self.assertEqual(config.pipeline, "handover")
        self.assertEqual(config.handover.maximum_center_distance_pixels, 120.0)
        self.assertEqual(
            config.resolve_source(
                credential_resolver=lambda key: f"rtsp://resolved/{key}"
            ),
            "rtsp://resolved/scoop-ai/shop-01-counter-01/rtsp-url",
        )
        with self.assertRaisesRegex(ConfigurationError, "credential resolver"):
            config.resolve_source()

    def test_environment_source_remains_available_as_fallback(self) -> None:
        path = self._write(
            """
[camera]
camera_id = "dev-camera-01"
mode = "live"
source_env = "DEV_CAMERA_URL"
analysis_fps = 5
"""
        )
        config = load_camera_config(path)

        self.assertEqual(config.resolve_source({"DEV_CAMERA_URL": "0"}), 0)
        with self.assertRaisesRegex(ConfigurationError, "DEV_CAMERA_URL"):
            config.resolve_source({})

    def test_rejects_unknown_keys_and_ambiguous_secrets(self) -> None:
        unknown = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
credential_key = "camera/001/rtsp"
typo_fps = 10
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "typo_fps"):
            load_camera_config(unknown)

        ambiguous = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
source = 0
credential_key = "camera/001/rtsp"
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            load_camera_config(ambiguous)

    def test_rejects_invalid_backoff_bounds(self) -> None:
        path = self._write(
            """
[service]
name = "test-service"
artifact_root = "artifacts/test"

[capture]
reconnect_seconds = 10
reconnect_max_seconds = 5
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "cannot be below"):
            load_service_config(path)

    def test_rejects_invalid_pipeline_and_handover_thresholds(self) -> None:
        invalid_pipeline = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "unknown"
source = 0
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "camera.pipeline"):
            load_camera_config(invalid_pipeline)

        invalid_handover = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "handover"
source = 0

[handover]
duplicate_iou_threshold = 1.5
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "duplicate_iou_threshold"):
            load_camera_config(invalid_handover)

    def test_customer_only_thresholds_are_configurable(self) -> None:
        path = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "handover"
source = 0

[handover]
customer_only_minimum_seconds = 1.25
customer_only_minimum_observations = 5
customer_only_minimum_movement = 0.08
customer_only_static_minimum_seconds = 4.0
customer_only_static_minimum_observations = 12
"""
        )
        config = load_camera_config(path)

        self.assertEqual(config.handover.customer_only_minimum_seconds, 1.25)
        self.assertEqual(config.handover.customer_only_minimum_observations, 5)
        self.assertEqual(config.handover.customer_only_minimum_movement, 0.08)
        self.assertEqual(config.handover.customer_only_static_minimum_seconds, 4.0)
        self.assertEqual(config.handover.customer_only_static_minimum_observations, 12)

    def test_customer_only_defaults_match_the_state_machine(self) -> None:
        path = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "handover"
source = 0
"""
        )
        handover = load_camera_config(path).handover
        defaults = HandoverFSMConfig(
            pickup_zone=((0.0, 0.0), (0.1, 0.0), (0.1, 0.1)),
            customer_zone=((0.5, 0.5), (0.6, 0.5), (0.6, 0.6)),
        )

        for name in (
            "customer_only_minimum_seconds",
            "customer_only_minimum_observations",
            "customer_only_minimum_movement",
            "customer_only_static_minimum_seconds",
            "customer_only_static_minimum_observations",
        ):
            self.assertEqual(getattr(handover, name), getattr(defaults, name), name)

    def test_rejects_contradictory_customer_only_thresholds(self) -> None:
        shorter_static = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "handover"
source = 0

[handover]
customer_only_minimum_seconds = 3.0
customer_only_static_minimum_seconds = 1.0
"""
        )
        with self.assertRaisesRegex(ConfigurationError, "customer_only_static_minimum_seconds"):
            load_camera_config(shorter_static)

        fewer_static = self._write(
            """
[camera]
camera_id = "camera-001"
mode = "live"
pipeline = "handover"
source = 0

[handover]
customer_only_minimum_observations = 9
customer_only_static_minimum_observations = 4
"""
        )
        with self.assertRaisesRegex(
            ConfigurationError, "customer_only_static_minimum_observations"
        ):
            load_camera_config(fewer_static)

    def _write(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.toml"
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
