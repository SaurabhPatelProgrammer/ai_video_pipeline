"""The live view must actually paint, not merely return HTTP 200.

A page can be served perfectly and still show nothing: the live view once used a
blob: URL that the page's own Content-Security-Policy forbade, so the image was
blocked with no error anywhere a request-level check could see. This loads the
real page in the same browser engine the desktop shell embeds and asserts the
frame decoded.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.dashboard.app import DashboardServer  # noqa: E402

PROBE = """
(() => {
  const frame = document.getElementById('frame');
  const state = document.getElementById('live-state');
  const badge = document.getElementById('badge-text');
  const overlay = document.getElementById('overlay');
  return JSON.stringify({
    decoded_width: frame ? frame.naturalWidth : 0,
    decoded_height: frame ? frame.naturalHeight : 0,
    placeholder_hidden: state ? state.classList.contains('hidden') : false,
    badge: badge ? badge.textContent : null,
    overlay_shapes: overlay ? overlay.children.length : 0
  });
})()
"""


def _jpeg(width: int = 320, height: int = 180) -> bytes:
    import cv2

    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


class _StubManager:
    """A configured product whose camera always yields one frame."""

    configured = True
    monitoring = False

    def __init__(self, root: Path) -> None:
        self.paths = mock.Mock()
        self.paths.root = root
        self.frame = _jpeg()

    def status(self) -> dict[str, object]:
        return {"configured": True, "shop_name": "Test Shop", "monitoring": False}

    def camera_settings(self) -> dict[str, object]:
        return {
            "camera_id": "main-counter", "camera_name": "Main Counter",
            "shop_name": "Test Shop", "pipeline": "handover", "device": "auto",
            "analysis_fps": 6.0, "expected_width": 320, "expected_height": 180,
            "model_version": "model-1", "configured_at": None, "calibrated_at": None,
            "source_name": "rtsp://camera.local/stream1", "monitoring": False,
            "pickup_zone": [[0.1, 0.6], [0.4, 0.6], [0.4, 0.9], [0.1, 0.9]],
            "customer_zone": [[0.6, 0.1], [0.9, 0.1], [0.9, 0.4], [0.6, 0.4]],
        }

    def live_snapshot(self) -> tuple[bytes, tuple[int, int]]:
        return self.frame, (320, 180)


class LiveViewRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        except ImportError:  # pragma: no cover - depends on the install
            raise unittest.SkipTest("PySide6 QtWebEngine is not installed")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        (root / "evidence").mkdir()
        self.manager = _StubManager(root)
        self.server = DashboardServer(
            root / "events.sqlite3", root / "evidence",
            port=0, product_manager=self.manager,
        ).start()
        self.addCleanup(self.server.stop)
        self.base = f"http://{self.server.address[0]}:{self.server.address[1]}"

    def _probe(self, path: str, settle_ms: int = 4000) -> dict[str, object]:
        from PySide6.QtCore import QEventLoop, QTimer, QUrl
        from PySide6.QtWebEngineCore import QWebEnginePage
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication

        messages: list[str] = []

        class Page(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
                messages.append(f"{getattr(level, 'name', level)}: {message}")

        application = QApplication.instance() or QApplication(sys.argv[:1])
        view = QWebEngineView()
        page = Page()
        view.setPage(page)
        view.resize(1280, 800)
        view.load(QUrl(self.base + path))

        captured: dict[str, object] = {}
        loop = QEventLoop()

        def run_probe() -> None:
            def received(value: object) -> None:
                try:
                    captured.update(json.loads(value))
                except (TypeError, ValueError):
                    captured["raw"] = value
                loop.quit()

            page.runJavaScript(PROBE, received)

        QTimer.singleShot(settle_ms, run_probe)
        QTimer.singleShot(settle_ms + 6000, loop.quit)
        loop.exec()
        view.deleteLater()
        application.processEvents()
        captured["console"] = messages
        return captured

    def test_the_live_frame_decodes_in_the_page(self) -> None:
        result = self._probe("/camera")
        self.assertEqual(
            (result.get("decoded_width"), result.get("decoded_height")),
            (320, 180),
            f"the frame did not paint; console={result.get('console')}",
        )
        self.assertTrue(result.get("placeholder_hidden"))
        self.assertEqual(result.get("badge"), "Live")

    def test_both_calibrated_zones_are_drawn_over_the_frame(self) -> None:
        self.assertEqual(self._probe("/camera").get("overlay_shapes"), 2)

    def test_the_page_reports_no_console_errors(self) -> None:
        console = self._probe("/camera").get("console") or []
        blocked = [line for line in console if "Content Security Policy" in line]
        self.assertEqual(blocked, [], f"the policy blocked a resource: {blocked}")


if __name__ == "__main__":
    unittest.main()
