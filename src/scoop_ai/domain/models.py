"""Normalized, model-independent scoop AI domain objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import hypot, isfinite


class ObjectClass(str, Enum):
    """Canonical classes emitted by the scoop detector/state estimator."""

    SCOOP = "scoop"
    LOADED_SCOOP = "loaded_scoop"
    SERVING_CONTAINER = "serving_container"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An ``xyxy`` bounding box normalized to the inclusive range ``[0, 1]``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in values):
            raise ValueError("bounding-box coordinates must be finite")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized bounding-box coordinates must be between 0 and 1")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def center_distance(self, other: "BoundingBox") -> float:
        x, y = self.center
        other_x, other_y = other.center
        return hypot(x - other_x, y - other_y)


@dataclass(frozen=True, slots=True)
class Observation:
    """A tracked object observation at a monotonic source timestamp."""

    timestamp: float
    track_id: int
    class_name: ObjectClass
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("observation timestamp must be finite and non-negative")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError("track_id must be an integer")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if not isinstance(self.class_name, ObjectClass):
            try:
                object_class = ObjectClass(str(self.class_name))
            except ValueError as exc:
                raise ValueError(f"unsupported object class: {self.class_name!r}") from exc
            object.__setattr__(self, "class_name", object_class)
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("observation confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DepositEvent:
    """A deposit confirmed by a loaded-to-empty release transition."""

    event_id: int
    timestamp: float
    scoop_track_id: int
    container_track_id: int
    confidence: float
    loaded_at: float
    approached_at: float
    released_at: float
    confirmed_at: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
