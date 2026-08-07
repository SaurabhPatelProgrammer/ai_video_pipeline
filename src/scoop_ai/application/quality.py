"""Camera-frame quality checks used before expensive model inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQualityAssessment:
    """Result of validating one frame against a fixed-camera profile."""

    acceptable: bool
    blur_variance: float
    changed_fraction: float
    reasons: tuple[str, ...]


class FrameQualityGate:
    """Reject unreadable frames and detect a moved or obstructed camera.

    The reference image is updated slowly only for accepted frames.  This is
    intentionally a health gate, not an activity detector; local foreground
    motion should affect less of the image than a moved/covered camera.
    """

    def __init__(
        self,
        minimum_blur_variance: float = 20.0,
        pixel_change_threshold: int = 35,
        maximum_changed_fraction: float = 0.65,
        reference_alpha: float = 0.01,
        analysis_width: int = 320,
    ) -> None:
        if minimum_blur_variance < 0:
            raise ValueError("minimum_blur_variance cannot be negative")
        if not 0 <= pixel_change_threshold <= 255:
            raise ValueError("pixel_change_threshold must be between 0 and 255")
        if not 0 < maximum_changed_fraction <= 1:
            raise ValueError("maximum_changed_fraction must be in (0, 1]")
        if not 0 < reference_alpha <= 1:
            raise ValueError("reference_alpha must be in (0, 1]")
        if analysis_width < 32:
            raise ValueError("analysis_width must be at least 32")
        self.minimum_blur_variance = minimum_blur_variance
        self.pixel_change_threshold = pixel_change_threshold
        self.maximum_changed_fraction = maximum_changed_fraction
        self.reference_alpha = reference_alpha
        self.analysis_width = analysis_width
        self._reference: np.ndarray | None = None

    def reset(self) -> None:
        self._reference = None

    def assess(self, frame: np.ndarray) -> FrameQualityAssessment:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        scale = min(1.0, self.analysis_width / frame.shape[1])
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (self.analysis_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        changed_fraction = 0.0
        if self._reference is None:
            self._reference = gray.astype(np.float32)
        else:
            reference_u8 = cv2.convertScaleAbs(self._reference)
            difference = cv2.absdiff(gray, reference_u8)
            changed_fraction = float(
                np.count_nonzero(difference >= self.pixel_change_threshold)
            ) / float(difference.size)

        reasons: list[str] = []
        if blur_variance < self.minimum_blur_variance:
            reasons.append("blurred")
        if changed_fraction > self.maximum_changed_fraction:
            reasons.append("camera_moved_or_obstructed")
        acceptable = not reasons
        if acceptable and self._reference is not None:
            cv2.accumulateWeighted(gray, self._reference, self.reference_alpha)
        return FrameQualityAssessment(
            acceptable=acceptable,
            blur_variance=blur_variance,
            changed_fraction=changed_fraction,
            reasons=tuple(reasons),
        )
