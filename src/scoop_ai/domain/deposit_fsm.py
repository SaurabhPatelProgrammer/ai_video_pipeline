"""Release-aware, timestamp-driven scoop deposit state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .models import DepositEvent, ObjectClass, Observation


class DepositState(str, Enum):
    IDLE = "idle"
    IN_TUB = "in_tub"
    LOADED = "loaded"
    APPROACHING = "approaching"
    ARMED = "armed"
    RELEASING = "releasing"
    WAITING_FOR_EXIT = "waiting_for_exit"


@dataclass(frozen=True, slots=True)
class DepositFSMConfig:
    """Temporal and normalized-spatial thresholds for one-camera inference."""

    minimum_confidence: float = 0.35
    approach_distance: float = 0.18
    exit_distance: float = 0.24
    minimum_approach_seconds: float = 0.10
    minimum_release_seconds: float = 0.08
    sequence_timeout_seconds: float = 8.0
    missing_tolerance_seconds: float = 0.35
    tub_zone: tuple[tuple[float, float], ...] | None = None
    serving_zone: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.approach_distance <= 0.0:
            raise ValueError("approach_distance must be positive")
        if self.exit_distance <= self.approach_distance:
            raise ValueError("exit_distance must exceed approach_distance for hysteresis")
        if self.minimum_approach_seconds < 0.0 or self.minimum_release_seconds < 0.0:
            raise ValueError("minimum dwell times cannot be negative")
        if self.sequence_timeout_seconds <= 0.0:
            raise ValueError("sequence_timeout_seconds must be positive")
        if self.missing_tolerance_seconds < 0.0:
            raise ValueError("missing_tolerance_seconds cannot be negative")
        for name, zone in (("tub_zone", self.tub_zone), ("serving_zone", self.serving_zone)):
            if zone is None:
                continue
            if len(zone) < 3:
                raise ValueError(f"{name} must contain at least three points")
            if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in zone):
                raise ValueError(f"{name} points must be normalized")


@dataclass(slots=True)
class _ScoopCycle:
    state: DepositState = DepositState.IDLE
    loaded_at: float | None = None
    approached_at: float | None = None
    release_started_at: float | None = None
    container_track_id: int | None = None
    last_seen_at: float | None = None
    loaded_confidence: float = 1.0
    container_confidence: float = 1.0
    release_confidence: float = 1.0
    released_at: float | None = None

    def reset(self) -> None:
        self.state = DepositState.IDLE
        self.loaded_at = None
        self.approached_at = None
        self.release_started_at = None
        self.container_track_id = None
        self.loaded_confidence = 1.0
        self.container_confidence = 1.0
        self.release_confidence = 1.0
        self.released_at = None


class ReleaseAwareDepositFSM:
    """Confirm deposits only after loaded, approach, release and exit evidence.

    Input timestamps must use the source clock (video PTS for files, a monotonic
    capture clock for live cameras). A loaded scoop merely approaching a cup does
    not count. The same tracked utensil must become ``scoop`` (empty) while still
    associated with the cup for the configured release dwell.
    """

    def __init__(self, config: DepositFSMConfig | None = None) -> None:
        self.config = config or DepositFSMConfig()
        self.total_count = 0
        self._next_event_id = 1
        self._last_timestamp: float | None = None
        self._cycles: dict[int, _ScoopCycle] = {}

    def reset(self) -> None:
        self.total_count = 0
        self._next_event_id = 1
        self._last_timestamp = None
        self._cycles.clear()

    def state_for(self, scoop_track_id: int) -> DepositState:
        cycle = self._cycles.get(scoop_track_id)
        return cycle.state if cycle is not None else DepositState.IDLE

    def update(
        self,
        timestamp: float,
        observations: list[Observation],
    ) -> list[DepositEvent]:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp = timestamp
        if any(abs(item.timestamp - timestamp) > 1e-6 for item in observations):
            raise ValueError("all observation timestamps must match the update timestamp")

        eligible = [
            item
            for item in observations
            if item.confidence >= self.config.minimum_confidence
        ]
        containers = {
            item.track_id: item
            for item in eligible
            if item.class_name is ObjectClass.SERVING_CONTAINER
            and (
                self.config.serving_zone is None
                or self._point_in_polygon(item.box.center, self.config.serving_zone)
            )
        }
        scoops = self._best_scoop_observation(eligible)
        events: list[DepositEvent] = []

        for track_id, observation in scoops.items():
            cycle = self._cycles.setdefault(track_id, _ScoopCycle())
            cycle.last_seen_at = timestamp
            event = self._advance_cycle(cycle, observation, containers, timestamp)
            if event is not None:
                events.append(event)

        for track_id, cycle in list(self._cycles.items()):
            if track_id in scoops or cycle.last_seen_at is None:
                continue
            missing_for = timestamp - cycle.last_seen_at
            if missing_for <= self.config.missing_tolerance_seconds:
                continue
            if cycle.state is DepositState.WAITING_FOR_EXIT:
                events.append(self._emit_pending_event(track_id, cycle, timestamp))
            else:
                cycle.reset()

        return events

    @staticmethod
    def _best_scoop_observation(
        observations: list[Observation],
    ) -> dict[int, Observation]:
        output: dict[int, Observation] = {}
        for item in observations:
            if item.class_name not in (ObjectClass.SCOOP, ObjectClass.LOADED_SCOOP):
                continue
            previous = output.get(item.track_id)
            if previous is None or item.confidence > previous.confidence:
                output[item.track_id] = item
        return output

    def _advance_cycle(
        self,
        cycle: _ScoopCycle,
        scoop: Observation,
        containers: dict[int, Observation],
        timestamp: float,
    ) -> DepositEvent | None:
        if (
            cycle.state is not DepositState.WAITING_FOR_EXIT
            and cycle.loaded_at is not None
            and timestamp - cycle.loaded_at > self.config.sequence_timeout_seconds
        ):
            cycle.reset()

        nearest, distance = self._nearest_container(scoop, containers)
        is_loaded = scoop.class_name is ObjectClass.LOADED_SCOOP

        if cycle.state is DepositState.IDLE:
            if self.config.tub_zone is not None:
                if self._point_in_polygon(scoop.box.center, self.config.tub_zone):
                    cycle.state = DepositState.IN_TUB
                    if is_loaded:
                        cycle.loaded_at = timestamp
                        cycle.loaded_confidence = scoop.confidence
                return None
            if is_loaded:
                cycle.state = DepositState.LOADED
                cycle.loaded_at = timestamp
                cycle.loaded_confidence = scoop.confidence
            return None

        if cycle.state is DepositState.IN_TUB:
            assert self.config.tub_zone is not None
            inside_tub = self._point_in_polygon(scoop.box.center, self.config.tub_zone)
            if inside_tub:
                if is_loaded:
                    cycle.loaded_at = cycle.loaded_at or timestamp
                    cycle.loaded_confidence = min(cycle.loaded_confidence, scoop.confidence)
                return None
            if not is_loaded:
                cycle.reset()
                return None
            cycle.state = DepositState.LOADED
            cycle.loaded_at = cycle.loaded_at or timestamp
            cycle.loaded_confidence = min(cycle.loaded_confidence, scoop.confidence)
            return None

        if cycle.state is DepositState.LOADED:
            if not is_loaded:
                cycle.reset()
                return None
            cycle.loaded_confidence = min(cycle.loaded_confidence, scoop.confidence)
            if nearest is not None and distance <= self.config.approach_distance:
                cycle.state = DepositState.APPROACHING
                cycle.approached_at = timestamp
                cycle.container_track_id = nearest.track_id
                cycle.container_confidence = nearest.confidence
            return None

        if cycle.state is DepositState.APPROACHING:
            container = containers.get(cycle.container_track_id)  # type: ignore[arg-type]
            if container is None:
                return None
            distance = scoop.box.center_distance(container.box)
            if distance >= self.config.exit_distance:
                cycle.state = DepositState.LOADED if is_loaded else DepositState.IDLE
                cycle.approached_at = None
                cycle.container_track_id = None
                return None
            assert cycle.approached_at is not None
            approach_ready = (
                timestamp - cycle.approached_at >= self.config.minimum_approach_seconds
            )
            if not is_loaded:
                if not approach_ready:
                    cycle.reset()
                    return None
                cycle.state = DepositState.RELEASING
                cycle.release_started_at = timestamp
                self._mark_release_if_ready(cycle, scoop, container, timestamp)
                return None
            cycle.container_confidence = min(cycle.container_confidence, container.confidence)
            if approach_ready:
                cycle.state = DepositState.ARMED
            return None

        if cycle.state is DepositState.ARMED:
            container = containers.get(cycle.container_track_id)  # type: ignore[arg-type]
            if container is None:
                return None
            distance = scoop.box.center_distance(container.box)
            if distance >= self.config.exit_distance:
                cycle.state = DepositState.LOADED if is_loaded else DepositState.IDLE
                cycle.approached_at = None
                cycle.container_track_id = None
                return None
            if is_loaded:
                cycle.loaded_confidence = min(cycle.loaded_confidence, scoop.confidence)
                return None
            cycle.state = DepositState.RELEASING
            cycle.release_started_at = timestamp
            self._mark_release_if_ready(cycle, scoop, container, timestamp)
            return None

        if cycle.state is DepositState.RELEASING:
            container = containers.get(cycle.container_track_id)  # type: ignore[arg-type]
            if container is None:
                return None
            if scoop.box.center_distance(container.box) >= self.config.exit_distance:
                cycle.reset()
                return None
            if is_loaded:
                cycle.state = DepositState.ARMED
                cycle.release_started_at = None
                return None
            self._mark_release_if_ready(cycle, scoop, container, timestamp)
            return None

        # WAITING_FOR_EXIT: do not count again until the utensil leaves with
        # hysteresis or disappears beyond the missing tolerance.
        container = containers.get(cycle.container_track_id)  # type: ignore[arg-type]
        if is_loaded:
            # A loaded-state rebound before withdrawal means the apparent empty
            # state was detector flicker or an aborted release, not a deposit.
            outside = (
                container is not None
                and scoop.box.center_distance(container.box) >= self.config.exit_distance
            )
            cycle.state = DepositState.LOADED if outside else DepositState.ARMED
            if outside:
                cycle.approached_at = None
                cycle.container_track_id = None
            cycle.release_started_at = None
            cycle.release_confidence = 1.0
            cycle.released_at = None
            return None
        if container is not None and scoop.box.center_distance(container.box) >= self.config.exit_distance:
            event = self._emit_pending_event(scoop.track_id, cycle, timestamp)
            return event
        return None

    def _mark_release_if_ready(
        self,
        cycle: _ScoopCycle,
        scoop: Observation,
        container: Observation,
        timestamp: float,
    ) -> None:
        assert cycle.release_started_at is not None
        if timestamp - cycle.release_started_at < self.config.minimum_release_seconds:
            return
        cycle.release_confidence = min(scoop.confidence, container.confidence)
        cycle.released_at = timestamp
        cycle.state = DepositState.WAITING_FOR_EXIT

    def _emit_pending_event(
        self,
        scoop_track_id: int,
        cycle: _ScoopCycle,
        confirmed_at: float,
    ) -> DepositEvent:
        assert cycle.loaded_at is not None
        assert cycle.approached_at is not None
        assert cycle.container_track_id is not None
        assert cycle.released_at is not None
        event = DepositEvent(
            event_id=self._next_event_id,
            timestamp=cycle.released_at,
            scoop_track_id=scoop_track_id,
            container_track_id=cycle.container_track_id,
            confidence=min(
                cycle.loaded_confidence,
                cycle.container_confidence,
                cycle.release_confidence,
            ),
            loaded_at=cycle.loaded_at,
            approached_at=cycle.approached_at,
            released_at=cycle.released_at,
            confirmed_at=confirmed_at,
        )
        self._next_event_id += 1
        self.total_count += 1
        cycle.reset()
        return event

    @staticmethod
    def _nearest_container(
        scoop: Observation,
        containers: dict[int, Observation],
    ) -> tuple[Observation | None, float]:
        nearest: Observation | None = None
        nearest_distance = float("inf")
        for container in containers.values():
            distance = scoop.box.center_distance(container.box)
            if distance < nearest_distance:
                nearest = container
                nearest_distance = distance
        return nearest, nearest_distance

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: tuple[tuple[float, float], ...],
    ) -> bool:
        """Ray-casting containment for normalized camera calibration polygons."""

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
