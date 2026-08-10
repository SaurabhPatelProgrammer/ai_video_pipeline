"""Tests for generic pickup-to-customer handover counting."""

from __future__ import annotations

import unittest

from scoop_ai.domain import (
    BoundingBox,
    HandoverFSMConfig,
    HandoverState,
    IceCreamHandoverFSM,
    ObjectClass,
    Observation,
)


PICKUP = ((0.1, 0.3), (0.9, 0.3), (0.9, 0.9), (0.1, 0.9))
CUSTOMER = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.5), (0.1, 0.5))


def item(timestamp: float, track_id: int, x: float, y: float, confidence: float = 0.9) -> Observation:
    return Observation(
        timestamp=timestamp,
        track_id=track_id,
        class_name=ObjectClass.ICE_CREAM_ITEM,
        confidence=confidence,
        box=BoundingBox(x - 0.02, y - 0.02, x + 0.02, y + 0.02),
    )


class IceCreamHandoverFSMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fsm = IceCreamHandoverFSM(
            HandoverFSMConfig(
                pickup_zone=PICKUP,
                customer_zone=CUSTOMER,
                minimum_transfer_seconds=0.2,
                minimum_customer_dwell_seconds=0.1,
                minimum_customer_observations=2,
                minimum_movement_distance=0.03,
                duplicate_cooldown_seconds=1.5,
                duplicate_distance=0.12,
            )
        )

    def complete_handover(self, track_id: int = 7, offset: float = 0.0):
        self.fsm.update(offset, [item(offset, track_id, 0.5, 0.62)])
        self.fsm.update(offset + 0.2, [item(offset + 0.2, track_id, 0.5, 0.44)])
        self.fsm.update(offset + 0.4, [item(offset + 0.4, track_id, 0.5, 0.34)])
        return self.fsm.update(offset + 1.2, [])

    def test_pickup_then_customer_dwell_counts_once(self) -> None:
        events = self.complete_handover()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].item_track_id, 7)
        self.assertEqual(self.fsm.total_count, 1)
        self.assertEqual(self.fsm.state_for(7), HandoverState.COUNTED)
        self.assertEqual(self.fsm.update(1.4, [item(1.4, 7, 0.5, 0.3)]), [])

    def test_track_starting_in_customer_zone_does_not_count(self) -> None:
        for timestamp in (0.0, 0.2, 0.4):
            self.assertEqual(
                self.fsm.update(timestamp, [item(timestamp, 8, 0.5, 0.2)]),
                [],
            )
        self.assertEqual(self.fsm.total_count, 0)

    def test_moving_customer_only_track_counts_when_it_disappears(self) -> None:
        self.fsm.update(0.0, [item(0.0, 16, 0.5, 0.24)])
        self.fsm.update(0.3, [item(0.3, 16, 0.5, 0.22)])
        self.fsm.update(0.6, [item(0.6, 16, 0.5, 0.19)])
        self.fsm.update(0.9, [])
        events = self.fsm.update(1.5, [])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].route, "customer_only_departure")
        self.assertEqual(events[0].timestamp, 0.6)
        self.assertEqual(self.fsm.total_count, 1)

    def test_static_customer_only_track_does_not_count(self) -> None:
        self.fsm.update(0.0, [item(0.0, 17, 0.5, 0.24)])
        self.fsm.update(0.3, [item(0.3, 17, 0.501, 0.24)])
        self.fsm.update(0.6, [item(0.6, 17, 0.5, 0.239)])
        self.fsm.update(1.5, [])

        self.assertEqual(self.fsm.total_count, 0)

    def test_sustained_static_customer_item_counts_on_departure(self) -> None:
        for timestamp in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
            self.fsm.update(timestamp, [item(timestamp, 18, 0.5, 0.24)])
        events = self.fsm.update(2.4, [])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].timestamp, 1.5)
        self.assertEqual(self.fsm.total_count, 1)

    def test_overlap_requires_meaningful_movement(self) -> None:
        self.fsm.update(0.0, [item(0.0, 9, 0.5, 0.44)])
        self.fsm.update(0.3, [item(0.3, 9, 0.51, 0.43)])
        events = self.fsm.update(0.6, [item(0.6, 9, 0.5, 0.44)])

        self.assertEqual(events, [])
        self.assertEqual(self.fsm.total_count, 0)

    def test_new_track_near_recent_event_is_deduplicated(self) -> None:
        self.assertEqual(len(self.complete_handover(track_id=10)), 1)
        events = self.complete_handover(track_id=11, offset=1.3)

        self.assertEqual(events, [])
        self.assertEqual(self.fsm.total_count, 1)
        self.assertEqual(self.fsm.state_for(11), HandoverState.COUNTED)

    def test_two_close_items_visible_together_both_count(self) -> None:
        self.fsm.update(
            0.0,
            [item(0.0, 20, 0.48, 0.62), item(0.0, 21, 0.52, 0.62)],
        )
        self.fsm.update(
            0.2,
            [item(0.2, 20, 0.48, 0.44), item(0.2, 21, 0.52, 0.44)],
        )
        self.fsm.update(
            0.4,
            [item(0.4, 20, 0.48, 0.34), item(0.4, 21, 0.52, 0.34)],
        )
        events = self.fsm.update(1.2, [])

        self.assertEqual([event.item_track_id for event in events], [20, 21])
        self.assertEqual(self.fsm.total_count, 2)

    def test_later_spatially_distinct_handover_counts(self) -> None:
        self.assertEqual(len(self.complete_handover(track_id=12)), 1)
        self.fsm.update(2.0, [item(2.0, 13, 0.25, 0.62)])
        self.fsm.update(2.2, [item(2.2, 13, 0.25, 0.44)])
        self.fsm.update(2.4, [item(2.4, 13, 0.25, 0.34)])
        events = self.fsm.update(3.2, [])

        self.assertEqual(len(events), 1)
        self.assertEqual(self.fsm.total_count, 2)

    def test_timeout_resets_incomplete_sequence(self) -> None:
        fsm = IceCreamHandoverFSM(
            HandoverFSMConfig(
                pickup_zone=PICKUP,
                customer_zone=CUSTOMER,
                sequence_timeout_seconds=0.5,
            )
        )
        fsm.update(0.0, [item(0.0, 14, 0.5, 0.62)])

        self.assertEqual(fsm.update(0.6, [item(0.6, 14, 0.5, 0.3)]), [])
        self.assertEqual(fsm.state_for(14), HandoverState.PICKED_UP)

    def test_low_confidence_item_is_ignored(self) -> None:
        self.assertEqual(self.fsm.update(0.0, [item(0.0, 15, 0.5, 0.62, 0.2)]), [])
        self.assertEqual(self.fsm.state_for(15), HandoverState.IDLE)


if __name__ == "__main__":
    unittest.main()
