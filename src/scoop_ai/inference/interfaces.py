"""Adapters that isolate detector-specific output from domain logic."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence, runtime_checkable

from scoop_ai.domain.models import BoundingBox, ObjectClass, Observation


@dataclass(frozen=True, slots=True)
class Detection:
    """A detector/tracker result in source-image pixel coordinates."""

    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


@runtime_checkable
class DetectorAdapter(Protocol):
    """Minimal interface implemented by RF-DETR or future detector adapters."""

    def predict(self, frame: object, timestamp: float) -> Sequence[Detection]:
        """Return detections for one source-timestamped frame."""


@runtime_checkable
class TrackerAdapter(Protocol):
    """Attach stable track IDs before conversion to domain observations."""

    def update(
        self, detections: Sequence[Detection], timestamp: float
    ) -> Sequence[Detection]:
        """Return detections whose ``track_id`` values are populated."""


def observations_from_detections(
    detections: Sequence[Detection],
    *,
    timestamp: float,
    frame_width: int,
    frame_height: int,
    aliases: dict[str, str] | None = None,
) -> list[Observation]:
    """Validate and normalize pixel detections into stable domain objects."""

    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    if not isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and non-negative")
    class_aliases = aliases or {}
    output: list[Observation] = []
    for detection in detections:
        if detection.track_id is None:
            raise ValueError("tracking is required before creating domain observations")
        source_name = detection.class_name.strip().lower()
        canonical_name = class_aliases.get(source_name, source_name)
        try:
            object_class = ObjectClass(canonical_name)
        except ValueError:
            continue
        x1, y1, x2, y2 = detection.xyxy
        if not all(isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("detection coordinates must be finite")
        # Small numerical overflow at image borders is clipped. Boxes wholly
        # outside the source or with reversed coordinates remain invalid.
        clipped = BoundingBox(
            x1=max(0.0, min(1.0, x1 / frame_width)),
            y1=max(0.0, min(1.0, y1 / frame_height)),
            x2=max(0.0, min(1.0, x2 / frame_width)),
            y2=max(0.0, min(1.0, y2 / frame_height)),
        )
        output.append(
            Observation(
                timestamp=timestamp,
                track_id=detection.track_id,
                class_name=object_class,
                confidence=detection.confidence,
                box=clipped,
            )
        )
    return output
