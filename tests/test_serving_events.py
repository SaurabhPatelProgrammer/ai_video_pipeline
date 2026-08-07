"""Tests for de-duplicated per-container scoop events."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from serving_events import Observation, ServingEventCounter  # noqa: E402


def observation(
    track_id: int,
    class_name: str,
    center_x: float,
    center_y: float,
    confidence: float = 0.9,
) -> Observation:
    return Observation(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        xyxy=(center_x - 5, center_y - 5, center_x + 5, center_y + 5),
    )


class ServingEventCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = ServingEventCounter(association_distance_pixels=30)
        self.container = observation(10, "serving_container", 100, 100)

    def test_loaded_scoop_approach_counts_once(self) -> None:
        far = observation(20, "loaded_scoop", 20, 20)
        near = observation(20, "loaded_scoop", 90, 90)

        self.assertEqual(self.counter.update([self.container, far]), [])
        events = self.counter.update([self.container, near], timestamp=12.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].container_track_id, 10)
        self.assertEqual(self.counter.containers[10].scoop_count, 1)
        self.assertEqual(self.counter.update([self.container, near]), [])

    def test_scoop_must_leave_before_second_count(self) -> None:
        near = observation(20, "loaded_scoop", 90, 90)
        far = observation(20, "loaded_scoop", 20, 20)

        self.counter.update([self.container, near])
        self.counter.update([self.container, far])
        events = self.counter.update([self.container, near])

        self.assertEqual(len(events), 1)
        self.assertEqual(self.counter.total_count, 2)

    def test_empty_scoop_never_counts(self) -> None:
        empty = observation(20, "scoop", 90, 90)
        self.assertEqual(self.counter.update([self.container, empty]), [])
        self.assertEqual(self.counter.total_count, 0)

    def test_short_detection_dropout_does_not_duplicate(self) -> None:
        near = observation(20, "loaded_scoop", 90, 90)
        self.counter.update([self.container, near])
        self.counter.update([self.container])
        events = self.counter.update([self.container, near])
        self.assertEqual(events, [])
        self.assertEqual(self.counter.total_count, 1)

    def test_event_is_assigned_to_nearest_container(self) -> None:
        other = observation(11, "serving_container", 200, 100)
        scoop = observation(20, "loaded_scoop", 190, 100)
        events = self.counter.update([self.container, other, scoop])
        self.assertEqual(events[0].container_track_id, 11)

    def test_low_confidence_detection_is_ignored(self) -> None:
        scoop = observation(20, "loaded_scoop", 90, 90, confidence=0.2)
        self.assertEqual(self.counter.update([self.container, scoop]), [])


if __name__ == "__main__":
    unittest.main()
