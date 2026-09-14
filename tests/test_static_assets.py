"""Every asset a page loads must be one its Content-Security-Policy permits.

A page that violates its own CSP fails silently in the browser: the request is
blocked, no exception reaches the script, and the feature simply never appears.
The live view shipped broken this way once, so the policy and the pages are
checked against each other here.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.dashboard.app import STATIC_ROOT  # noqa: E402

PAGES = sorted(STATIC_ROOT.glob("*.html"))
STYLESHEETS = sorted(STATIC_ROOT.glob("*.css"))
# Mirrors the policy sent by DashboardServer._send.
IMG_SRC = ("'self'", "data:")


class ContentSecurityPolicyTests(unittest.TestCase):
    def test_the_expected_pages_are_present(self) -> None:
        self.assertEqual(
            {path.name for path in PAGES},
            {"index.html", "setup.html", "camera.html", "system.html"},
        )

    def test_no_page_builds_an_image_source_the_policy_blocks(self) -> None:
        """blob: and object URLs are not 'self'; an <img> using one never paints."""

        if "blob:" in IMG_SRC:
            self.skipTest("policy now allows blob: image sources")
        for path in PAGES:
            with self.subTest(page=path.name):
                self.assertNotIn("createObjectURL", path.read_text(encoding="utf-8"))

    def test_nothing_is_loaded_from_a_remote_host(self) -> None:
        """The client must work on a shop computer with no internet access."""

        remote = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)
        css_remote = re.compile(r"""url\(\s*["']?https?://""", re.IGNORECASE)
        for path in PAGES + STYLESHEETS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(asset=path.name):
                self.assertIsNone(remote.search(body))
                self.assertIsNone(css_remote.search(body))

    def test_every_page_uses_the_shared_stylesheet(self) -> None:
        for path in PAGES:
            with self.subTest(page=path.name):
                self.assertIn('href="/app.css"', path.read_text(encoding="utf-8"))


class LiveViewMarkupTests(unittest.TestCase):
    def test_the_frame_is_driven_by_load_and_error_events(self) -> None:
        """Chaining off the image events keeps one request in flight and reports
        a failure the element itself cannot describe."""

        body = (STATIC_ROOT / "camera.html").read_text(encoding="utf-8")
        self.assertIn("/api/camera/snapshot?t=", body)
        self.assertIn("addEventListener('load'", body)
        self.assertIn("addEventListener('error'", body)


if __name__ == "__main__":
    unittest.main()
