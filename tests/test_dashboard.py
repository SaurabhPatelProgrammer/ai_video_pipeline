"""HTTP-level tests for the local operator dashboard."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from scoop_ai.dashboard import DashboardServer, ProductManager
from scoop_ai.dashboard.discovery import DiscoveredCamera, parse_probe_matches
from scoop_ai.dashboard.product import PreviewSession
from scoop_ai.inference.checkpoint_manifest import create_checkpoint_manifest
from scoop_ai.storage import EvidenceRecord, EventRecord, EvidenceWriter, SessionRecord, SQLiteEventRepository


NOW = "2026-08-13T08:30:00+00:00"
DISCOVERED = DiscoveredCamera(
    device_id="urn:uuid:camera-123",
    name="Main Counter Camera",
    host="192.168.1.40",
    service_url="http://192.168.1.40/onvif/device_service",
    scopes=("onvif://www.onvif.org/name/Main_Counter_Camera",),
)


class DiscoveryTests(unittest.TestCase):
    def test_probe_match_is_parsed_without_vendor_dependencies(self) -> None:
        payload = b'''<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
 xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"><s:Body><d:ProbeMatches><d:ProbeMatch>
 <a:EndpointReference><a:Address>urn:uuid:camera-123</a:Address></a:EndpointReference>
 <d:Scopes>onvif://www.onvif.org/name/Main_Counter_Camera</d:Scopes>
 <d:XAddrs>http://192.168.1.40/onvif/device_service</d:XAddrs>
 </d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>'''
        cameras = parse_probe_matches(payload)
        self.assertEqual(cameras, [DISCOVERED])

class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database = root / "events.sqlite3"
        self.evidence = root / "evidence"
        with SQLiteEventRepository(self.database) as repository:
            repository.start_session(SessionRecord(session_id="session-1", camera_id="camera-1", started_at=NOW))
            artifact = EvidenceWriter(self.evidence).write_bytes("session-1", "event-1", b"fake-jpeg", extension=".jpg")
            repository.insert_event(EventRecord(
                event_id="event-1", session_id="session-1", camera_id="camera-1",
                event_type="ice_cream_handover_candidate", occurred_at=NOW, confidence=0.42,
                evidence_path=artifact.relative_path, evidence_sha256=artifact.sha256,
                metadata={"route": "pickup_to_customer", "item_track_id": 7},
            ))
            repository.register_evidence(EvidenceRecord(
                evidence_id="event-1", event_id="event-1", relative_path=artifact.relative_path,
                sha256=artifact.sha256, size_bytes=artifact.size_bytes, media_type="image/jpeg",
                created_at=artifact.created_at, retention_deadline="2026-08-27T08:30:00+00:00",
            ))
        self.server = DashboardServer(self.database, self.evidence, port=0, health_url="http://127.0.0.1:1/health").start()
        self.base = f"http://{self.server.address[0]}:{self.server.address[1]}"

    def tearDown(self) -> None:
        self.server.stop()
        self.temporary.cleanup()

    def get_json(self, path: str) -> dict[str, object]:
        with urllib.request.urlopen(self.base + path) as response:
            return json.loads(response.read())

    def post_json(self, path: str, payload: object) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_dashboard_page_and_daily_payload(self) -> None:
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read().decode()
        self.assertIn("Scoop AI Dashboard", page)
        payload = self.get_json("/api/dashboard?date=2026-08-13&state=all")
        self.assertEqual(payload["summary"]["candidate_events"], 1)  # type: ignore[index]
        self.assertEqual(payload["summary"]["pending_review"], 1)  # type: ignore[index]
        self.assertEqual(payload["events"][0]["route"], "pickup_to_customer")  # type: ignore[index]
        self.assertFalse(payload["system"]["online"])  # type: ignore[index]

    def test_review_accepts_corrected_quantity_and_is_append_only(self) -> None:
        status, response = self.post_json("/api/events/event-1/review", {"decision": "accepted", "quantity": 2})
        self.assertEqual(status, 200)
        self.assertEqual(response["event"]["quantity"], 2)  # type: ignore[index]
        payload = self.get_json("/api/dashboard?date=2026-08-13&state=accepted")
        self.assertEqual(payload["summary"]["confirmed_items"], 2)  # type: ignore[index]
        with SQLiteEventRepository(self.database) as repository:
            reviews = repository.list_reviews("event-1")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].metadata["corrected_quantity"], 2)

    def test_invalid_review_and_evidence_traversal_are_rejected(self) -> None:
        status, _ = self.post_json("/api/events/event-1/review", {"decision": "accepted", "quantity": 0})
        self.assertEqual(status, 400)
        with self.assertRaises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(self.base + "/evidence/not-found")
        self.assertEqual(failure.exception.code, 404)

    def test_server_refuses_non_loopback_bind(self) -> None:
        with self.assertRaises(ValueError):
            DashboardServer(self.database, self.evidence, host="0.0.0.0", port=0)

    def test_dns_rebinding_host_is_rejected(self) -> None:
        request = urllib.request.Request(self.base + "/api/dashboard", headers={"Host": "attacker.example"})
        with self.assertRaises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(request)
        self.assertEqual(failure.exception.code, 403)


class ProductSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        bundle = root / "model"
        bundle.mkdir()
        checkpoint = bundle / "model.pth"
        checkpoint.write_bytes(b"small-test-checkpoint")
        self.manifest = bundle / "model-manifest.json"
        create_checkpoint_manifest(
            checkpoint, self.manifest, architecture="nano", dataset_version="dataset-1",
            model_version="model-1", input_resolution=128, classes=("ice_cream_item",),
        )
        self.credentials: dict[str, str] = {}
        self.manager = ProductManager(
            root / "product", self.manifest,
            credential_writer=lambda key, value: self.credentials.__setitem__(key, value),
            discovery_provider=lambda: [DISCOVERED],
        )
        self.manager._previews["preview-token"] = PreviewSession(  # noqa: SLF001
            source="rtsp://user:secret@camera.local/stream",
            frame=np.full((120, 160, 3), 100, dtype=np.uint8),
            created_at=time.monotonic(),
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.temporary.cleanup()

    def test_first_run_setup_writes_valid_secret_free_configuration(self) -> None:
        result = self.manager.save_setup(
            preview_token="preview-token", shop_name="Test Shop", camera_name="Main Counter",
            camera_id="main-counter",
            pickup_zone=[[0.1, 0.55], [0.45, 0.55], [0.45, 0.9], [0.1, 0.9]],
            customer_zone=[[0.55, 0.1], [0.9, 0.1], [0.9, 0.4], [0.55, 0.4]],
        )
        self.assertTrue(result["configured"])
        self.assertEqual(
            self.credentials["scoop-ai/main-counter/rtsp-url"],
            "rtsp://user:secret@camera.local/stream",
        )
        config_text = self.manager.paths.camera_config.read_text(encoding="utf-8")
        self.assertNotIn("secret", config_text)
        self.assertNotIn("rtsp://", config_text)
        self.assertTrue(self.manager.paths.database.is_file())
        self.assertTrue(self.manager.paths.calibration.is_file())

    def test_setup_page_is_default_until_product_is_configured(self) -> None:
        server = DashboardServer(
            self.manager.paths.database, self.manager.paths.evidence,
            port=0, product_manager=self.manager,
        ).start()
        try:
            base = f"http://{server.address[0]}:{server.address[1]}"
            with urllib.request.urlopen(base + "/") as response:
                page = response.read().decode()
            self.assertIn("First-time setup", page)
            with urllib.request.urlopen(base + "/api/product/status") as response:
                status = json.loads(response.read())
            self.assertFalse(status["configured"])
        finally:
            server.stop()

    def test_discovery_endpoint_returns_safe_camera_metadata(self) -> None:
        server = DashboardServer(
            self.manager.paths.database, self.manager.paths.evidence,
            port=0, product_manager=self.manager,
        ).start()
        try:
            base = f"http://{server.address[0]}:{server.address[1]}"
            request = urllib.request.Request(
                base + "/api/setup/discover", data=b"{}",
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request) as response:
                payload = json.loads(response.read())
            self.assertEqual(payload["cameras"][0]["device_id"], DISCOVERED.device_id)
            self.assertEqual(payload["cameras"][0]["host"], DISCOVERED.host)
        finally:
            server.stop()

    def test_stale_discovery_identity_is_rejected_before_camera_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "scan the network again"):
            self.manager.test_camera(
                "rtsp://camera.invalid/stream",
                camera_device_identity="urn:uuid:unknown",
            )

    def test_local_device_and_site_identity_survive_restart(self) -> None:
        initial = self.manager.status()
        restarted = ProductManager(
            self.manager.paths.root, self.manifest,
            credential_writer=lambda key, value: self.credentials.__setitem__(key, value),
            discovery_provider=lambda: [],
        )
        self.assertEqual(restarted.status()["device_id"], initial["device_id"])
        self.assertEqual(restarted.status()["site_id"], initial["site_id"])

    def test_invalid_camera_id_is_rejected_before_secret_is_stored(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.save_setup(
                preview_token="preview-token", shop_name="Test Shop", camera_name="Main Counter",
                camera_id="INVALID ID", pickup_zone=[[0, 0], [0.4, 0], [0.4, 0.4]],
                customer_zone=[[0.6, 0.6], [1, 0.6], [1, 1]],
            )
        self.assertEqual(self.credentials, {})

    def test_overlapping_zones_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.manager.save_setup(
                preview_token="preview-token", shop_name="Test Shop", camera_name="Main Counter",
                camera_id="main-counter",
                pickup_zone=[[0.1, 0.1], [0.7, 0.1], [0.7, 0.7], [0.1, 0.7]],
                customer_zone=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            )
        self.assertEqual(self.credentials, {})


if __name__ == "__main__":
    unittest.main()
