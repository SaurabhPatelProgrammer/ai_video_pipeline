"""Tests for immutable served-order profile creation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.training import create_served_order_profile  # noqa: E402


class ServedOrderProfileTests(unittest.TestCase):
    def test_creates_review_only_checksum_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "tub_zone": [[0, 0], [0.4, 0], [0.4, 0.4]],
                        "serving_zone": [[0.6, 0.6], [1, 0.6], [1, 1]],
                        "minimum_transfer_seconds": 0.2,
                        "transfer_timeout_seconds": 5,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "served.json"
            result = create_served_order_profile(
                base,
                output,
                tub_threshold=0.075,
                customer_threshold=0.04,
                minimum_preparation_seconds=20,
                order_timeout_seconds=45,
                minimum_container_frames=3,
                minimum_customer_frames=2,
                cooldown_seconds=20,
                labeled_events=2,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["event_mode"], "served_order")
            self.assertNotIn("transfer_timeout_seconds", payload)
            self.assertFalse(payload["served_order_tuning"]["production_approved"])
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                create_served_order_profile(
                    base,
                    output,
                    tub_threshold=0.075,
                    customer_threshold=0.04,
                    minimum_preparation_seconds=20,
                    order_timeout_seconds=45,
                    minimum_container_frames=3,
                    minimum_customer_frames=2,
                    cooldown_seconds=20,
                    labeled_events=2,
                )


if __name__ == "__main__":
    unittest.main()
