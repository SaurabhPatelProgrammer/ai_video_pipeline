"""A killed client must not leave a monitoring service behind, or start a second."""

from __future__ import annotations

import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.dashboard.product import ProductManager, PreviewSession  # noqa: E402
from scoop_ai.inference.checkpoint_manifest import create_checkpoint_manifest  # noqa: E402


class _FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.terminated = False
        self._running = True

    def poll(self):
        return None if self._running else 0

    def terminate(self) -> None:
        self.terminated = True
        self._running = False

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self._running = False


class MonitorLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        bundle = root / "model"
        bundle.mkdir()
        checkpoint = bundle / "model.pth"
        checkpoint.write_bytes(b"small-test-checkpoint")
        manifest = bundle / "model-manifest.json"
        create_checkpoint_manifest(
            checkpoint, manifest, architecture="nano", dataset_version="dataset-1",
            model_version="model-1", input_resolution=128, classes=("ice_cream_item",),
        )
        self.processes: list[_FakeProcess] = []

        def factory(*_args, **_kwargs):
            process = _FakeProcess()
            self.processes.append(process)
            return process

        self.manager = ProductManager(
            root / "product", manifest,
            credential_writer=lambda _key, _value: None,
            discovery_provider=lambda: [],
            process_factory=factory,
        )
        self.addCleanup(self.manager.close)
        self.manager._previews["token"] = PreviewSession(  # noqa: SLF001
            source="rtsp://user:secret@camera.local/stream",
            frame=np.full((120, 160, 3), 100, dtype=np.uint8),
            created_at=float("inf"),
        )
        self.manager.save_setup(
            preview_token="token", shop_name="Test Shop", camera_name="Main Counter",
            camera_id="main-counter",
            pickup_zone=[[0.1, 0.55], [0.45, 0.55], [0.45, 0.9], [0.1, 0.9]],
            customer_zone=[[0.55, 0.1], [0.9, 0.1], [0.9, 0.4], [0.55, 0.4]],
        )

    def test_monitoring_starts_and_stops_once(self) -> None:
        self.manager.start_monitoring()
        self.assertTrue(self.manager.monitoring)
        self.assertEqual(len(self.processes), 1)

        # A second request must not spawn a duplicate service.
        self.manager.start_monitoring()
        self.assertEqual(len(self.processes), 1)

        self.manager.stop_monitoring()
        self.assertTrue(self.processes[0].terminated)
        self.assertFalse(self.manager.monitoring)

    def test_a_service_surviving_a_crash_blocks_a_second_one(self) -> None:
        """The real cause of duplicate services: an orphan we no longer own."""

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        port = listener.getsockname()[1]

        config = self.manager.paths.service_config
        config.write_text(
            config.read_text(encoding="utf-8").replace("health_port = 8080", f"health_port = {port}"),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as caught:
            self.manager.start_monitoring()
        self.assertIn("already running", str(caught.exception))
        self.assertEqual(self.processes, [])

    def test_a_free_health_port_means_no_service_is_running(self) -> None:
        self.assertFalse(self.manager._service_already_running())  # noqa: SLF001

    def test_the_child_is_adopted_so_the_operating_system_can_clean_up(self) -> None:
        with mock.patch("scoop_ai.dashboard.product._adopt_into_job") as adopt:
            self.manager.start_monitoring()
        adopt.assert_called_once()
        self.assertEqual(adopt.call_args.args[1], self.processes[0].pid)

    def test_setup_cannot_run_while_monitoring_holds_the_camera(self) -> None:
        self.manager.start_monitoring()
        self.manager._previews["second"] = PreviewSession(  # noqa: SLF001
            source="rtsp://user:secret@camera.local/stream2",
            frame=np.full((120, 160, 3), 100, dtype=np.uint8),
            created_at=float("inf"),
        )
        self.manager.save_setup(
            preview_token="second", shop_name="Test Shop", camera_name="Second",
            camera_id="second-counter",
            pickup_zone=[[0.1, 0.55], [0.45, 0.55], [0.45, 0.9], [0.1, 0.9]],
            customer_zone=[[0.55, 0.1], [0.9, 0.1], [0.9, 0.4], [0.55, 0.4]],
        )
        # Reconfiguring must release the previous service, not run two cameras.
        self.assertTrue(self.processes[0].terminated)
        self.assertFalse(self.manager.monitoring)


if __name__ == "__main__":
    unittest.main()
