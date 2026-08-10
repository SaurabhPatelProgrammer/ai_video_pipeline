"""Small deterministic proximity tracker for sparse one-class detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from .interfaces import Detection


@dataclass(slots=True)
class _Track:
    track_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    last_seen_at: float
    hits: int = 1


class ProximityTrackerAdapter:
    """Associate sparse detections by center distance and source timestamp."""

    def __init__(
        self,
        *,
        maximum_center_distance_pixels: float = 120.0,
        lost_track_seconds: float = 0.75,
        minimum_consecutive_frames: int = 2,
        duplicate_iou_threshold: float = 0.50,
    ) -> None:
        if maximum_center_distance_pixels <= 0 or lost_track_seconds <= 0:
            raise ValueError("distance and lost_track_seconds must be positive")
        if minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be positive")
        if not 0 < duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be between 0 and 1")
        self.maximum_center_distance_pixels = maximum_center_distance_pixels
        self.lost_track_seconds = lost_track_seconds
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._last_timestamp = None

    def update(
        self, detections: Sequence[Detection], timestamp: float
    ) -> Sequence[Detection]:
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp = timestamp
        detections = self._deduplicate_detections(detections)
        for track_id, track in list(self._tracks.items()):
            if timestamp - track.last_seen_at > self.lost_track_seconds:
                self._tracks.pop(track_id, None)

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if detection.class_name != track.class_name:
                    continue
                distance = self._center_distance(detection.xyxy, track.xyxy)
                if distance <= self.maximum_center_distance_pixels:
                    candidates.append((distance, detection_index, track_id))
        candidates.sort()
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for _distance, detection_index, track_id in candidates:
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assignments[detection_index] = track_id

        output: list[Detection] = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    class_name=detection.class_name,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    last_seen_at=timestamp,
                )
            else:
                track = self._tracks[track_id]
                track.xyxy = detection.xyxy
                track.confidence = detection.confidence
                track.last_seen_at = timestamp
                track.hits += 1
            track = self._tracks[track_id]
            if track.hits >= self.minimum_consecutive_frames:
                output.append(
                    Detection(
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        xyxy=detection.xyxy,
                        track_id=track_id,
                    )
                )
        return output

    def _deduplicate_detections(
        self, detections: Sequence[Detection]
    ) -> list[Detection]:
        kept: list[int] = []
        for index in sorted(
            range(len(detections)),
            key=lambda value: (-detections[value].confidence, value),
        ):
            candidate = detections[index]
            if any(
                candidate.class_name == detections[previous].class_name
                and self._intersection_over_union(
                    candidate.xyxy, detections[previous].xyxy
                )
                >= self.duplicate_iou_threshold
                for previous in kept
            ):
                continue
            kept.append(index)
        return [detections[index] for index in sorted(kept)]

    @staticmethod
    def _intersection_over_union(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_distance(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        first_x = (first[0] + first[2]) / 2
        first_y = (first[1] + first[3]) / 2
        second_x = (second[0] + second[2]) / 2
        second_y = (second[1] + second[3]) / 2
        return hypot(first_x - second_x, first_y - second_y)
