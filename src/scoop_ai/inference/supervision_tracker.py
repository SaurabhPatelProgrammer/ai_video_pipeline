"""Supervision ByteTrack adapter isolated behind the tracker protocol."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from .interfaces import Detection


class SupervisionByteTrackAdapter:
    """Attach stable IDs while preserving checkpoint-aware class names."""

    def __init__(
        self,
        *,
        frame_rate: float = 10.0,
        confidence_threshold: float = 0.35,
        lost_track_seconds: float = 1.5,
        tracker: object | None = None,
    ) -> None:
        if frame_rate <= 0 or lost_track_seconds <= 0:
            raise ValueError("frame_rate and lost_track_seconds must be positive")
        self._class_ids: dict[str, int] = {}
        if tracker is None:
            import supervision as sv

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The `ByteTrack` was deprecated.*")
                tracker = sv.ByteTrack(
                    track_activation_threshold=confidence_threshold,
                    lost_track_buffer=max(2, round(frame_rate * lost_track_seconds)),
                    minimum_matching_threshold=0.8,
                    frame_rate=frame_rate,
                    minimum_consecutive_frames=2,
                )
        self._tracker = tracker

    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float,
    ) -> Sequence[Detection]:
        del timestamp  # ByteTrack is configured with the effective frame rate.
        import supervision as sv

        boxes = np.array([item.xyxy for item in detections], dtype=float).reshape((-1, 4))
        confidences = np.array([item.confidence for item in detections], dtype=float)
        class_ids = np.array(
            [self._class_ids.setdefault(item.class_name, len(self._class_ids)) for item in detections],
            dtype=int,
        )
        names = np.array([item.class_name for item in detections], dtype=object)
        source = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
            data={"class_name": names},
        )
        tracked = self._tracker.update_with_detections(source)
        if tracked.tracker_id is None:
            return []
        tracked_names = tracked.data.get("class_name", ())
        return [
            Detection(
                class_name=str(tracked_names[index]),
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in box),
                track_id=int(track_id),
            )
            for index, (box, confidence, track_id) in enumerate(
                zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)
            )
        ]
