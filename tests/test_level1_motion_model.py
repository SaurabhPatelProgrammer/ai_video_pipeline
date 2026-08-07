"""Deterministic tests for the Level-1 motion state machine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from level1_motion_model import MotionScoopStateMachine, MotionState  # noqa: E402
from level1_counter import fit_for_display  # noqa: E402


class MotionScoopStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MotionScoopStateMachine(
            tub_threshold=0.1,
            serving_threshold=0.1,
            minimum_loading_frames=2,
            minimum_serving_frames=2,
            minimum_transfer_seconds=0.2,
            transfer_timeout_seconds=2.0,
            cooldown_seconds=0.5,
        )

    def test_tub_then_serving_motion_emits_one_candidate(self) -> None:
        sequence = [
            (0.2, 0.0, 0.0),
            (0.2, 0.0, 0.1),
            (0.0, 0.2, 0.4),
            (0.0, 0.2, 0.5),
        ]
        events = [self.model.update(*values) for values in sequence]
        self.assertEqual(sum(event is not None for event in events), 1)
        self.assertEqual(self.model.state, MotionState.COOLDOWN)

    def test_serving_motion_without_tub_motion_does_not_count(self) -> None:
        for index in range(10):
            self.assertIsNone(self.model.update(0.0, 0.3, index * 0.1))

    def test_cooldown_prevents_duplicate_candidate(self) -> None:
        self.model.update(0.2, 0.0, 0.0)
        self.model.update(0.2, 0.0, 0.1)
        self.model.update(0.0, 0.2, 0.4)
        self.assertIsNotNone(self.model.update(0.0, 0.2, 0.5))
        self.assertIsNone(self.model.update(0.3, 0.3, 0.6))
        self.assertIsNone(self.model.update(0.3, 0.3, 0.7))

    def test_loading_candidate_expires(self) -> None:
        self.model.update(0.2, 0.0, 0.0)
        self.model.update(0.2, 0.0, 0.1)
        self.assertIsNone(self.model.update(0.0, 0.0, 2.2))
        self.assertEqual(self.model.state, MotionState.IDLE)


class DisplayFitTests(unittest.TestCase):
    def test_large_frame_is_fitted_without_changing_aspect_ratio(self) -> None:
        frame = np.zeros((1296, 2304, 3), dtype=np.uint8)
        fitted = fit_for_display(frame, 1280, 800)
        self.assertEqual(fitted.shape, (720, 1280, 3))

    def test_small_frame_is_not_upscaled(self) -> None:
        frame = np.zeros((416, 368, 3), dtype=np.uint8)
        self.assertIs(fit_for_display(frame, 1280, 800), frame)


if __name__ == "__main__":
    unittest.main()
