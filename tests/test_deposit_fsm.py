"""Contract tests for normalized, release-and-withdrawal-aware deposits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoop_ai.domain import (  # noqa: E402
    BoundingBox,
    DepositFSMConfig,
    DepositState,
    ObjectClass,
    Observation,
    ReleaseAwareDepositFSM,
)


CONTAINER_BOX = BoundingBox(0.45, 0.45, 0.55, 0.60)
NEAR_BOX = BoundingBox(0.40, 0.42, 0.48, 0.50)
FAR_BOX = BoundingBox(0.05, 0.05, 0.12, 0.12)


def observation(
    timestamp: float,
    track_id: int,
    class_name: ObjectClass,
    box: BoundingBox,
    confidence: float = 0.9,
) -> Observation:
    return Observation(timestamp, track_id, class_name, confidence, box)


def frame(
    timestamp: float,
    scoop_class: ObjectClass | None,
    scoop_box: BoundingBox = NEAR_BOX,
) -> list[Observation]:
    items = [
        observation(
            timestamp,
            10,
            ObjectClass.SERVING_CONTAINER,
            CONTAINER_BOX,
            0.95,
        )
    ]
    if scoop_class is not None:
        items.append(observation(timestamp, 20, scoop_class, scoop_box))
    return items


class ReleaseAwareDepositFSMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fsm = ReleaseAwareDepositFSM(
            DepositFSMConfig(
                approach_distance=0.18,
                exit_distance=0.24,
                minimum_approach_seconds=0.10,
                minimum_release_seconds=0.08,
                missing_tolerance_seconds=0.20,
            )
        )

    def prepare_pending_release(self) -> None:
        self.assertEqual(
            self.fsm.update(0.00, frame(0.00, ObjectClass.LOADED_SCOOP, FAR_BOX)),
            [],
        )
        self.assertEqual(
            self.fsm.update(0.10, frame(0.10, ObjectClass.LOADED_SCOOP)),
            [],
        )
        self.assertEqual(
            self.fsm.update(0.22, frame(0.22, ObjectClass.LOADED_SCOOP)),
            [],
        )
        self.assertEqual(self.fsm.state_for(20), DepositState.ARMED)
        self.assertEqual(
            self.fsm.update(0.30, frame(0.30, ObjectClass.SCOOP)),
            [],
        )
        self.assertEqual(
            self.fsm.update(0.40, frame(0.40, ObjectClass.SCOOP)),
            [],
        )
        self.assertEqual(self.fsm.state_for(20), DepositState.WAITING_FOR_EXIT)

    def test_no_event_is_emitted_before_withdrawal(self) -> None:
        self.prepare_pending_release()

        self.assertEqual(self.fsm.total_count, 0)
        self.assertEqual(
            self.fsm.update(0.45, frame(0.45, ObjectClass.SCOOP)),
            [],
        )
        self.assertEqual(self.fsm.total_count, 0)

    def test_withdrawal_confirms_event_and_retains_release_timestamp(self) -> None:
        self.prepare_pending_release()

        events = self.fsm.update(0.50, frame(0.50, ObjectClass.SCOOP, FAR_BOX))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].container_track_id, 10)
        self.assertEqual(events[0].released_at, 0.40)
        self.assertEqual(events[0].timestamp, 0.40)
        self.assertEqual(events[0].confirmed_at, 0.50)
        self.assertEqual(self.fsm.total_count, 1)

    def test_loaded_scoop_proximity_without_release_never_counts(self) -> None:
        self.fsm.update(0.00, frame(0.00, ObjectClass.LOADED_SCOOP, FAR_BOX))
        for timestamp in (0.10, 0.25, 0.50, 0.80):
            self.assertEqual(
                self.fsm.update(
                    timestamp,
                    frame(timestamp, ObjectClass.LOADED_SCOOP),
                ),
                [],
            )
        self.assertEqual(self.fsm.total_count, 0)

    def test_loaded_state_rebound_before_withdrawal_cancels_pending_release(self) -> None:
        self.prepare_pending_release()

        self.assertEqual(
            self.fsm.update(0.50, frame(0.50, ObjectClass.LOADED_SCOOP, FAR_BOX)),
            [],
        )
        self.assertEqual(self.fsm.state_for(20), DepositState.LOADED)
        self.assertEqual(self.fsm.total_count, 0)

    def test_missing_scoop_after_release_confirms_withdrawal_once(self) -> None:
        self.prepare_pending_release()

        self.assertEqual(self.fsm.update(0.50, frame(0.50, None)), [])
        events = self.fsm.update(0.65, frame(0.65, None))
        self.assertEqual(len(events), 1)
        self.assertEqual(self.fsm.update(1.00, frame(1.00, None)), [])

    def test_timestamps_must_be_monotonic(self) -> None:
        self.fsm.update(1.0, frame(1.0, ObjectClass.LOADED_SCOOP))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            self.fsm.update(0.9, frame(0.9, ObjectClass.LOADED_SCOOP))


class NormalizedDomainTests(unittest.TestCase):
    def test_invalid_normalized_box_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            BoundingBox(-0.01, 0.1, 0.5, 0.5)

    def test_calibrated_fsm_requires_tub_entry_before_loaded_transfer(self) -> None:
        fsm = ReleaseAwareDepositFSM(
            DepositFSMConfig(
                tub_zone=((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)),
            )
        )
        fsm.update(0.0, frame(0.0, ObjectClass.LOADED_SCOOP, NEAR_BOX))
        self.assertEqual(fsm.state_for(20), DepositState.IDLE)
        fsm.update(0.1, frame(0.1, ObjectClass.SCOOP, FAR_BOX))
        self.assertEqual(fsm.state_for(20), DepositState.IN_TUB)
        fsm.update(0.2, frame(0.2, ObjectClass.LOADED_SCOOP, NEAR_BOX))
        self.assertEqual(fsm.state_for(20), DepositState.LOADED)


if __name__ == "__main__":
    unittest.main()
