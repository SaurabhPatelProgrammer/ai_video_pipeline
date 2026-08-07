"""Tests for source redaction, checkpoint policy and operational status helpers."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.operations.health import HealthRegistry, HealthState, MetricsRegistry  # noqa: E402
from scoop_ai.operations.logging import JsonFormatter, log_context  # noqa: E402
from scoop_ai.security import (  # noqa: E402
    CredentialBackendError,
    CredentialNotFoundError,
    redact_secrets,
    resolve_credential,
    safe_source_name,
    store_credential,
    verify_checkpoint,
)


class SourceSecurityTests(unittest.TestCase):
    def test_source_name_removes_credentials_query_and_fragment(self) -> None:
        safe = safe_source_name("rtsp://user:password@[2001:db8::1]:554/live?token=abc#x")
        self.assertEqual(safe, "rtsp://[2001:db8::1]:554/live")

    def test_recursive_redaction_preserves_non_secret_context(self) -> None:
        clean = redact_secrets(
            {
                "camera_url": "rtsp://user:pass@camera.local/live?token=x",
                "password": "super-secret",
                "message": "connect token=abc to rtsp://u:p@cam/live?auth=x",
                "camera_id": "shop-1",
            }
        )
        self.assertEqual(clean["camera_url"], "rtsp://camera.local/live")
        self.assertEqual(clean["password"], "<redacted>")
        self.assertNotIn("abc", clean["message"])
        self.assertNotIn("u:p", clean["message"])
        self.assertEqual(clean["camera_id"], "shop-1")

    def test_checkpoint_requires_approved_hash_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "approved.pth"
            checkpoint.write_bytes(b"known-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            verified = verify_checkpoint(checkpoint, {digest}, allowed_root=root)
            self.assertEqual(verified.sha256, digest)
            with self.assertRaises(PermissionError):
                verify_checkpoint(checkpoint, {"0" * 64}, allowed_root=root)

    def test_credentials_use_lazy_keyring_without_environment_fallback(self) -> None:
        backend = SimpleNamespace(
            get_password=mock.Mock(return_value="rtsp-secret"),
            set_password=mock.Mock(),
        )
        with mock.patch("scoop_ai.security.importlib.import_module", return_value=backend):
            self.assertEqual(resolve_credential("camera.shop-1"), "rtsp-secret")
            store_credential("camera.shop-1", "new-secret")
        backend.get_password.assert_called_once_with("scoop-ai", "camera.shop-1")
        backend.set_password.assert_called_once_with(
            "scoop-ai", "camera.shop-1", "new-secret"
        )

    def test_hierarchical_credential_reference_matches_camera_config_contract(self) -> None:
        backend = SimpleNamespace(
            get_password=mock.Mock(return_value="rtsp-secret"),
            set_password=mock.Mock(),
        )
        key = "scoop-ai/shop-01-counter-01/rtsp-url"
        with mock.patch("scoop_ai.security.importlib.import_module", return_value=backend):
            self.assertEqual(resolve_credential(key), "rtsp-secret")
            store_credential(key, "new-secret")
        backend.get_password.assert_called_once_with("scoop-ai", key)
        backend.set_password.assert_called_once_with("scoop-ai", key, "new-secret")

    def test_missing_or_unavailable_credential_has_clear_error(self) -> None:
        missing = SimpleNamespace(get_password=mock.Mock(return_value=None))
        with mock.patch("scoop_ai.security.importlib.import_module", return_value=missing):
            with self.assertRaises(CredentialNotFoundError):
                resolve_credential("camera.shop-1")
        with mock.patch(
            "scoop_ai.security.importlib.import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(CredentialBackendError, "keyring"):
                resolve_credential("camera.shop-1")


class OperationsTests(unittest.TestCase):
    def test_json_logging_redacts_secrets_and_adds_context(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter(service_name="test-service"))
        logger = logging.getLogger("security-test-logger")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        with log_context(camera_id="camera-1", session_id="session-1"):
            logger.info("opening rtsp://user:pass@camera/live?token=abc", extra={"api_key": "x"})
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["camera_id"], "camera-1")
        self.assertEqual(payload["api_key"], "<redacted>")
        self.assertNotIn("pass", payload["message"])
        self.assertNotIn("abc", payload["message"])

    def test_required_health_becomes_unhealthy_when_stale(self) -> None:
        registry = HealthRegistry(required_components={"camera", "inference"})
        observed = datetime(2026, 8, 5, tzinfo=timezone.utc)
        registry.report(
            "camera", HealthState.HEALTHY, observed_at=observed, monotonic_at=10.0
        )
        self.assertEqual(registry.snapshot(monotonic_now=10.5).state, HealthState.STARTING)
        registry.report(
            "inference", HealthState.HEALTHY, observed_at=observed, monotonic_at=10.0
        )
        self.assertTrue(registry.snapshot(monotonic_now=10.5).ready)
        stale = registry.snapshot(monotonic_now=25.0, stale_after_seconds=5)
        self.assertEqual(stale.state, HealthState.UNHEALTHY)
        self.assertFalse(stale.live)

    def test_metrics_emit_bounded_summary(self) -> None:
        metrics = MetricsRegistry(sample_limit=3)
        metrics.increment("camera_reconnects_total", 2)
        metrics.gauge("last_frame_age_seconds", 0.2)
        for value in (1.0, 2.0, 3.0, 4.0):
            metrics.observe("inference_latency_ms", value)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["camera_reconnects_total"], 2)
        self.assertEqual(snapshot["summaries"]["inference_latency_ms"]["count"], 3)
        self.assertEqual(snapshot["summaries"]["inference_latency_ms"]["min"], 2.0)


if __name__ == "__main__":
    unittest.main()
