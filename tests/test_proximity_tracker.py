"""Tests for deterministic sparse-item proximity tracking."""

from __future__ import annotations

import unittest

from scoop_ai.inference import Detection, ProximityTrackerAdapter


def detection(center_x: float, center_y: float) -> Detection:
    return Detection(
        class_name="ice_cream_item",
        confidence=0.8,
        xyxy=(center_x - 5, center_y - 5, center_x + 5, center_y + 5),
    )


def scored_detection(center_x: float, center_y: float, confidence: float) -> Detection:
    item = detection(center_x, center_y)
    return Detection(
        class_name=item.class_name,
        confidence=confidence,
        xyxy=item.xyxy,
    )


class ProximityTrackerAdapterTests(unittest.TestCase):
    def test_track_is_emitted_after_two_frames_and_survives_fast_motion(self) -> None:
        tracker = ProximityTrackerAdapter(maximum_center_distance_pixels=120)

        self.assertEqual(tracker.update([detection(100, 100)], 0.0), [])
        tracked = tracker.update([detection(170, 120)], 0.2)

        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].track_id, 1)

    def test_far_detection_creates_new_track(self) -> None:
        tracker = ProximityTrackerAdapter(
            maximum_center_distance_pixels=30,
            minimum_consecutive_frames=1,
        )
        first = tracker.update([detection(10, 10)], 0.0)
        second = tracker.update([detection(100, 100)], 0.2)

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 2)

    def test_expired_track_is_not_reused(self) -> None:
        tracker = ProximityTrackerAdapter(
            lost_track_seconds=0.5,
            minimum_consecutive_frames=1,
        )
        first = tracker.update([detection(20, 20)], 0.0)
        tracker.update([], 0.6)
        second = tracker.update([detection(20, 20)], 0.7)

        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_multiple_tracks_are_assigned_one_to_one(self) -> None:
        tracker = ProximityTrackerAdapter(minimum_consecutive_frames=1)
        first = tracker.update([detection(20, 20), detection(200, 20)], 0.0)
        second = tracker.update([detection(205, 20), detection(25, 20)], 0.2)

        self.assertEqual([item.track_id for item in first], [1, 2])
        self.assertEqual([item.track_id for item in second], [2, 1])

    def test_overlapping_duplicate_keeps_highest_confidence_only(self) -> None:
        tracker = ProximityTrackerAdapter(minimum_consecutive_frames=1)
        tracked = tracker.update(
            [scored_detection(50, 50, 0.2), scored_detection(51, 50, 0.8)],
            0.0,
        )

        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
