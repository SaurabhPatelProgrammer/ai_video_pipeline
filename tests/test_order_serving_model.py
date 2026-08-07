"""Deterministic tests for the served-order state machine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from order_serving_model import ServedOrderState, ServedOrderStateMachine  # noqa: E402


class ServedOrderStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ServedOrderStateMachine(
            tub_threshold=0.1,
            serving_threshold=0.1,
            minimum_loading_frames=2,
            minimum_serving_frames=2,
            minimum_preparation_seconds=2.0,
            order_timeout_seconds=10.0,
            cooldown_seconds=3.0,
            nominal_fps=10.0,
        )

    def test_emits_one_order_after_preparation_and_handoff(self) -> None:
        self.assertIsNone(self.model.update(0.2, 0.0, 0.0))
        self.assertIsNone(self.model.update(0.2, 0.0, 0.1))
        self.assertIsNone(self.model.update(0.0, 0.2, 1.0))
        self.assertIsNone(self.model.update(0.0, 0.2, 2.1))
        event = self.model.update(0.0, 0.2, 2.2)
        self.assertIsNotNone(event)
        self.assertEqual(self.model.state, ServedOrderState.COOLDOWN)

    def test_early_customer_motion_does_not_complete_order(self) -> None:
        self.model.update(0.2, 0.0, 0.0)
        self.model.update(0.2, 0.0, 0.1)
        self.model.update(0.0, 0.2, 0.5)
        self.model.update(0.0, 0.2, 0.6)
        self.assertEqual(self.model.state, ServedOrderState.PREPARING)

    def test_timeout_discards_unserved_preparation(self) -> None:
        self.model.update(0.2, 0.0, 0.0)
        self.model.update(0.2, 0.0, 0.1)
        self.assertIsNone(self.model.update(0.0, 0.2, 10.2))
        self.assertEqual(self.model.state, ServedOrderState.IDLE)

    def test_cooldown_prevents_duplicate_handoff(self) -> None:
        self.model.update(0.2, 0.0, 0.0)
        self.model.update(0.2, 0.0, 0.1)
        self.model.update(0.0, 0.2, 2.1)
        self.assertIsNotNone(self.model.update(0.0, 0.2, 2.2))
        self.assertIsNone(self.model.update(0.2, 0.2, 2.3))


if __name__ == "__main__":
    unittest.main()
