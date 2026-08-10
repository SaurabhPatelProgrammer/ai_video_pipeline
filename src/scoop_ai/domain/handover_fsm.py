"""Tracked pickup-to-customer handover state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite

from .models import HandoverEvent, ObjectClass, Observation


class HandoverState(str, Enum):
    IDLE = "idle"
    PICKED_UP = "picked_up"
    CUSTOMER_DWELL = "customer_dwell"
    CUSTOMER_VISIBLE = "customer_visible"
    COUNTED = "counted"


@dataclass(frozen=True, slots=True)
class HandoverFSMConfig:
    pickup_zone: tuple[tuple[float, float], ...]
    customer_zone: tuple[tuple[float, float], ...]
    minimum_confidence: float = 0.35
    minimum_transfer_seconds: float = 0.20
    minimum_customer_dwell_seconds: float = 0.12
    minimum_customer_observations: int = 2
    minimum_movement_distance: float = 0.03
    sequence_timeout_seconds: float = 5.0
    missing_tolerance_seconds: float = 0.75
    duplicate_cooldown_seconds: float = 3.5
    duplicate_distance: float = 0.12
    customer_only_minimum_seconds: float = 0.50
    customer_only_minimum_observations: int = 3
    customer_only_minimum_movement: float = 0.025
    customer_only_static_minimum_seconds: float = 1.50
    customer_only_static_minimum_observations: int = 6

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        for name, zone in (("pickup_zone", self.pickup_zone), ("customer_zone", self.customer_zone)):
            if len(zone) < 3:
                raise ValueError(f"{name} must contain at least three points")
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in zone):
                raise ValueError(f"{name} points must be normalized")
        if self.minimum_transfer_seconds < 0 or self.minimum_customer_dwell_seconds < 0:
            raise ValueError("minimum times cannot be negative")
        if self.minimum_customer_observations < 1:
            raise ValueError("minimum_customer_observations must be positive")
        if self.minimum_movement_distance < 0:
            raise ValueError("minimum_movement_distance cannot be negative")
        if self.sequence_timeout_seconds <= 0 or self.missing_tolerance_seconds < 0:
            raise ValueError("timeout values are invalid")
        if self.duplicate_cooldown_seconds < 0 or self.duplicate_distance < 0:
            raise ValueError("duplicate thresholds cannot be negative")
        if self.customer_only_minimum_seconds < 0 or self.customer_only_minimum_movement < 0:
            raise ValueError("customer-only thresholds cannot be negative")
        if self.customer_only_minimum_observations < 2:
            raise ValueError("customer_only_minimum_observations must be at least two")
        if self.customer_only_static_minimum_seconds < self.customer_only_minimum_seconds:
            raise ValueError("static customer-only duration cannot be shorter than moving duration")
        if self.customer_only_static_minimum_observations < self.customer_only_minimum_observations:
            raise ValueError(
                "static customer-only observations cannot be fewer than moving observations"
            )


@dataclass(slots=True)
class _HandoverCycle:
    state: HandoverState = HandoverState.IDLE
    pickup_at: float | None = None
    pickup_center: tuple[float, float] | None = None
    customer_entered_at: float | None = None
    customer_observations: int = 0
    last_seen_at: float | None = None
    confidence: float = 1.0
    last_center: tuple[float, float] | None = None
    maximum_movement: float = 0.0
    route: str = "pickup_to_customer"
    coexisting_track_ids: set[int] = field(default_factory=set)

    def reset(self) -> None:
        self.state = HandoverState.IDLE
        self.pickup_at = None
        self.pickup_center = None
        self.customer_entered_at = None
        self.customer_observations = 0
        self.confidence = 1.0
        self.last_center = None
        self.maximum_movement = 0.0
        self.route = "pickup_to_customer"
        self.coexisting_track_ids.clear()


class IceCreamHandoverFSM:
    """Count one event per tracked item that moves from pickup to customer."""

    def __init__(self, config: HandoverFSMConfig) -> None:
        self.config = config
        self.total_count = 0
        self._next_event_id = 1
        self._last_timestamp: float | None = None
        self._cycles: dict[int, _HandoverCycle] = {}
        self._counted_track_ids: set[int] = set()
        self._recent_events: list[HandoverEvent] = []

    def state_for(self, track_id: int) -> HandoverState:
        if track_id in self._counted_track_ids:
            return HandoverState.COUNTED
        cycle = self._cycles.get(track_id)
        return cycle.state if cycle else HandoverState.IDLE

    def reset(self) -> None:
        self.total_count = 0
        self._next_event_id = 1
        self._last_timestamp = None
        self._cycles.clear()
        self._counted_track_ids.clear()
        self._recent_events.clear()

    def update(self, timestamp: float, observations: list[Observation]) -> list[HandoverEvent]:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp = timestamp
        if any(abs(item.timestamp - timestamp) > 1e-6 for item in observations):
            raise ValueError("all observation timestamps must match the update timestamp")

        items: dict[int, Observation] = {}
        for item in observations:
            if item.class_name is not ObjectClass.ICE_CREAM_ITEM:
                continue
            if item.confidence < self.config.minimum_confidence:
                continue
            previous = items.get(item.track_id)
            if previous is None or item.confidence > previous.confidence:
                items[item.track_id] = item

        events: list[HandoverEvent] = []
        visible_track_ids = set(items)
        for track_id in visible_track_ids:
            cycle = self._cycles.setdefault(track_id, _HandoverCycle())
            cycle.coexisting_track_ids.update(visible_track_ids - {track_id})
        for track_id, item in items.items():
            if track_id in self._counted_track_ids:
                continue
            cycle = self._cycles.setdefault(track_id, _HandoverCycle())
            cycle.last_seen_at = timestamp
            event = self._advance(track_id, cycle, item, timestamp)
            if event is not None:
                events.append(event)

        for track_id, cycle in list(self._cycles.items()):
            if track_id in items or cycle.last_seen_at is None:
                continue
            if timestamp - cycle.last_seen_at > self.config.missing_tolerance_seconds:
                if cycle.state is HandoverState.CUSTOMER_VISIBLE and self._departure_ready(
                    cycle, cycle.last_seen_at
                ):
                    assert cycle.last_center is not None
                    event = self._emit(track_id, cycle, cycle.last_center, timestamp)
                    if event is not None:
                        events.append(event)
                self._cycles.pop(track_id, None)
        return events

    def _advance(
        self,
        track_id: int,
        cycle: _HandoverCycle,
        item: Observation,
        timestamp: float,
    ) -> HandoverEvent | None:
        center = item.box.center
        in_pickup = self._point_in_polygon(center, self.config.pickup_zone)
        in_customer = self._point_in_polygon(center, self.config.customer_zone)

        if cycle.state is HandoverState.IDLE:
            if in_pickup:
                cycle.state = HandoverState.PICKED_UP
                cycle.pickup_at = timestamp
                cycle.pickup_center = center
                cycle.confidence = item.confidence
                cycle.last_center = center
            elif in_customer:
                cycle.state = HandoverState.CUSTOMER_VISIBLE
                cycle.pickup_at = timestamp
                cycle.pickup_center = center
                cycle.customer_entered_at = timestamp
                cycle.customer_observations = 1
                cycle.confidence = item.confidence
                cycle.last_center = center
                cycle.route = "customer_only_departure"
            return None

        assert cycle.pickup_at is not None
        assert cycle.pickup_center is not None
        if timestamp - cycle.pickup_at > self.config.sequence_timeout_seconds:
            cycle.reset()
            if in_pickup:
                cycle.state = HandoverState.PICKED_UP
                cycle.pickup_at = timestamp
                cycle.pickup_center = center
                cycle.confidence = item.confidence
            return None

        cycle.confidence = min(cycle.confidence, item.confidence)
        moved = hypot(center[0] - cycle.pickup_center[0], center[1] - cycle.pickup_center[1])
        cycle.maximum_movement = max(cycle.maximum_movement, moved)
        cycle.last_center = center
        transfer_ready = (
            timestamp - cycle.pickup_at + 1e-9 >= self.config.minimum_transfer_seconds
            and moved >= self.config.minimum_movement_distance
        )

        if cycle.state is HandoverState.PICKED_UP:
            if in_customer and transfer_ready:
                cycle.state = HandoverState.CUSTOMER_DWELL
                cycle.customer_entered_at = timestamp
                cycle.customer_observations = 1
            return None

        if cycle.state is HandoverState.CUSTOMER_DWELL:
            if not in_customer:
                cycle.state = HandoverState.PICKED_UP
                cycle.customer_entered_at = None
                cycle.customer_observations = 0
                return None
            assert cycle.customer_entered_at is not None
            cycle.customer_observations += 1
            dwell_ready = (
                timestamp - cycle.customer_entered_at + 1e-9
                >= self.config.minimum_customer_dwell_seconds
                and cycle.customer_observations >= self.config.minimum_customer_observations
            )
            if dwell_ready:
                cycle.state = HandoverState.CUSTOMER_VISIBLE
                return None
        if cycle.state is HandoverState.CUSTOMER_VISIBLE:
            if in_customer:
                cycle.customer_observations += 1
                return None
            if self._departure_ready(cycle, timestamp):
                return self._emit(track_id, cycle, center, timestamp)
            cycle.reset()
        return None

    def _departure_ready(self, cycle: _HandoverCycle, timestamp: float) -> bool:
        return cycle.route == "pickup_to_customer" or self._customer_only_ready(
            cycle, timestamp
        )

    def _customer_only_ready(self, cycle: _HandoverCycle, timestamp: float) -> bool:
        assert cycle.customer_entered_at is not None
        duration = timestamp - cycle.customer_entered_at + 1e-9
        moving_route = (
            duration >= self.config.customer_only_minimum_seconds
            and cycle.customer_observations
            >= self.config.customer_only_minimum_observations
            and cycle.maximum_movement
            >= self.config.customer_only_minimum_movement
        )
        static_route = (
            duration >= self.config.customer_only_static_minimum_seconds
            and cycle.customer_observations
            >= self.config.customer_only_static_minimum_observations
        )
        return moving_route or static_route

    def _emit(
        self,
        track_id: int,
        cycle: _HandoverCycle,
        center: tuple[float, float],
        confirmed_at: float,
    ) -> HandoverEvent | None:
        assert cycle.pickup_at is not None
        assert cycle.customer_entered_at is not None
        self._counted_track_ids.add(track_id)
        cycle.state = HandoverState.COUNTED
        for previous in reversed(self._recent_events):
            age = confirmed_at - previous.confirmed_at
            if age > self.config.duplicate_cooldown_seconds:
                break
            if previous.item_track_id in cycle.coexisting_track_ids:
                continue
            distance = hypot(center[0] - previous.center_x, center[1] - previous.center_y)
            if distance <= self.config.duplicate_distance:
                return None
        event = HandoverEvent(
            event_id=self._next_event_id,
            timestamp=(
                cycle.last_seen_at
                if cycle.last_seen_at is not None
                else cycle.customer_entered_at
            ),
            item_track_id=track_id,
            confidence=cycle.confidence,
            pickup_at=cycle.pickup_at,
            customer_entered_at=cycle.customer_entered_at,
            confirmed_at=confirmed_at,
            center_x=center[0],
            center_y=center[1],
            route=cycle.route,
        )
        self._next_event_id += 1
        self.total_count += 1
        self._recent_events.append(event)
        cutoff = confirmed_at - self.config.duplicate_cooldown_seconds
        self._recent_events = [item for item in self._recent_events if item.confirmed_at >= cutoff]
        return event

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
    ) -> bool:
        x, y = point
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            crosses = (current_y > y) != (previous_y > y)
            if crosses:
                boundary_x = (previous_x - current_x) * (y - current_y) / (
                    previous_y - current_y
                ) + current_x
                if x < boundary_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside
