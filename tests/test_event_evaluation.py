"""Tests for one-to-one event timing and container metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.training import EvaluationEvent, evaluate_events  # noqa: E402


def event(timestamp: float, container: str = "cup-1", session: str = "video-1") -> EvaluationEvent:
    return EvaluationEvent(session, timestamp, container)


class EventEvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_reports_fp_per_hour(self) -> None:
        result = evaluate_events(
            [event(1.1), event(1.2), event(5.0)],
            [event(1.0), event(5.2)],
            tolerance_seconds=0.3,
            observed_duration_seconds=3600,
        )

        self.assertEqual(result.metrics.true_positives, 2)
        self.assertEqual(result.metrics.false_positives, 1)
        self.assertEqual(result.metrics.false_negatives, 0)
        self.assertEqual(result.metrics.false_positives_per_hour, 1.0)
        self.assertEqual(len(result.matches), 2)

    def test_wrong_container_is_fp_fn_and_explicit_assignment_error(self) -> None:
        result = evaluate_events(
            [event(10.1, "cup-2")],
            [event(10.0, "cup-1")],
            tolerance_seconds=0.5,
        )

        self.assertEqual(result.metrics.true_positives, 0)
        self.assertEqual(result.metrics.false_positives, 1)
        self.assertEqual(result.metrics.false_negatives, 1)
        self.assertEqual(result.metrics.wrong_container_assignments, 1)
        self.assertEqual(result.metrics.wrong_container_rate, 1.0)
        self.assertEqual(result.metrics.exact_container_count_accuracy, 0.0)

    def test_events_outside_tolerance_do_not_match(self) -> None:
        result = evaluate_events(
            [event(2.0)],
            [event(1.0)],
            tolerance_seconds=0.25,
        )
        self.assertEqual(result.metrics.true_positives, 0)
        self.assertEqual(result.unmatched_prediction_indices, (0,))
        self.assertEqual(result.unmatched_truth_indices, (0,))

    def test_empty_evaluation_is_perfect_and_has_zero_assignment_error(self) -> None:
        result = evaluate_events([], [])
        self.assertEqual(result.metrics.precision, 1.0)
        self.assertEqual(result.metrics.recall, 1.0)
        self.assertEqual(result.metrics.f1, 1.0)
        self.assertEqual(result.metrics.wrong_container_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
