"""Temporal serving-event logic shared by live and offline workbenches.

The detector is responsible for recognizing ``loaded_scoop`` and
``serving_container``.  This module turns their tracked spatial relationship
into de-duplicated scoop-deposit candidates.  It deliberately contains no
camera or model code, which keeps the event rules deterministic and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from time import time


@dataclass(frozen=True)
class Observation:
    track_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class ScoopEvent:
    event_id: int
    timestamp: float
    scoop_track_id: int
    container_track_id: int
    confidence: float
    distance_pixels: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class ContainerCount:
    track_id: int
    scoop_count: int = 0
    last_seen_frame: int = 0


class ServingEventCounter:
    """Count one event when a loaded scoop approaches a tracked container.

    A scoop must leave the association radius before it can trigger again.
    This prevents repeated counts while an employee shapes or releases one
    scoop over the same container.
    """

    def __init__(
        self,
        association_distance_pixels: float = 140.0,
        minimum_confidence: float = 0.35,
        stale_track_frames: int = 90,
        missing_tolerance_frames: int = 3,
    ) -> None:
        if association_distance_pixels <= 0:
            raise ValueError("association_distance_pixels must be positive")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if stale_track_frames < 1:
            raise ValueError("stale_track_frames must be at least 1")
        if missing_tolerance_frames < 0:
            raise ValueError("missing_tolerance_frames cannot be negative")
        self.association_distance_pixels = association_distance_pixels
        self.minimum_confidence = minimum_confidence
        self.stale_track_frames = stale_track_frames
        self.missing_tolerance_frames = missing_tolerance_frames
        self.frame_index = 0
        self.total_count = 0
        self._next_event_id = 1
        self._near_container: dict[int, int] = {}
        self._scoop_last_seen: dict[int, int] = {}
        self.containers: dict[int, ContainerCount] = {}

    def reset(self) -> None:
        self.frame_index = 0
        self.total_count = 0
        self._next_event_id = 1
        self._near_container.clear()
        self._scoop_last_seen.clear()
        self.containers.clear()

    def update(
        self,
        observations: list[Observation],
        timestamp: float | None = None,
    ) -> list[ScoopEvent]:
        self.frame_index += 1
        event_time = time() if timestamp is None else timestamp
        containers = [
            item
            for item in observations
            if item.class_name == "serving_container"
            and item.confidence >= self.minimum_confidence
        ]
        loaded_scoops = [
            item
            for item in observations
            if item.class_name == "loaded_scoop"
            and item.confidence >= self.minimum_confidence
        ]

        for item in containers:
            state = self.containers.setdefault(
                item.track_id,
                ContainerCount(track_id=item.track_id),
            )
            state.last_seen_frame = self.frame_index

        for item in loaded_scoops:
            self._scoop_last_seen[item.track_id] = self.frame_index
        for track_id in list(self._near_container):
            last_seen = self._scoop_last_seen.get(track_id, 0)
            if self.frame_index - last_seen > self.missing_tolerance_frames:
                self._near_container.pop(track_id, None)
                self._scoop_last_seen.pop(track_id, None)

        events: list[ScoopEvent] = []
        for scoop in loaded_scoops:
            nearest, distance = self._nearest_container(scoop, containers)
            previous_container = self._near_container.get(scoop.track_id)
            if nearest is None or distance > self.association_distance_pixels:
                self._near_container.pop(scoop.track_id, None)
                continue

            self._near_container[scoop.track_id] = nearest.track_id
            if previous_container == nearest.track_id:
                continue

            confidence = min(scoop.confidence, nearest.confidence)
            event = ScoopEvent(
                event_id=self._next_event_id,
                timestamp=event_time,
                scoop_track_id=scoop.track_id,
                container_track_id=nearest.track_id,
                confidence=confidence,
                distance_pixels=distance,
            )
            self._next_event_id += 1
            self.total_count += 1
            self.containers[nearest.track_id].scoop_count += 1
            events.append(event)

        cutoff = self.frame_index - self.stale_track_frames
        for track_id, state in list(self.containers.items()):
            if state.last_seen_frame < cutoff:
                self.containers.pop(track_id, None)
        return events

    @staticmethod
    def _nearest_container(
        scoop: Observation,
        containers: list[Observation],
    ) -> tuple[Observation | None, float]:
        nearest: Observation | None = None
        nearest_distance = float("inf")
        scoop_x, scoop_y = scoop.center
        for container in containers:
            container_x, container_y = container.center
            distance = hypot(scoop_x - container_x, scoop_y - container_y)
            if distance < nearest_distance:
                nearest = container
                nearest_distance = distance
        return nearest, nearest_distance
