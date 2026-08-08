"""Static approved-view fingerprints and sustained scene-drift protection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence

import cv2
import numpy as np

NormalizedPoint = tuple[float, float]


FINGERPRINT_SCHEMA_VERSION = 1
_GRAY_SIZE = (160, 90)
_EDGE_SIZE = (32, 18)


def _gray_frame(frame: np.ndarray, size: tuple[int, int] = _GRAY_SIZE) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("reference frame must be a non-empty BGR image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def create_reference_fingerprint(frame: np.ndarray) -> dict[str, object]:
    """Create a compact static fingerprint that tolerates small lighting shifts."""

    gray = _gray_frame(frame)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.resize(edges, _EDGE_SIZE, interpolation=cv2.INTER_AREA)
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "algorithm": "gray-edge-v1",
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "gray_width": _GRAY_SIZE[0],
        "gray_height": _GRAY_SIZE[1],
        "gray": gray.astype(int).ravel().tolist(),
        "edge_width": _EDGE_SIZE[0],
        "edge_height": _EDGE_SIZE[1],
        "edge": edges.astype(int).ravel().tolist(),
    }


def _reference_arrays(reference: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if reference.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        raise ValueError("unsupported calibration fingerprint schema")
    if reference.get("algorithm") != "gray-edge-v1":
        raise ValueError("unsupported calibration fingerprint algorithm")
    gray_width = int(reference.get("gray_width", 0))
    gray_height = int(reference.get("gray_height", 0))
    edge_width = int(reference.get("edge_width", 0))
    edge_height = int(reference.get("edge_height", 0))
    gray_values = reference.get("gray")
    edge_values = reference.get("edge")
    if (
        gray_width < 1
        or gray_height < 1
        or edge_width < 1
        or edge_height < 1
        or not isinstance(gray_values, list)
        or not isinstance(edge_values, list)
        or len(gray_values) != gray_width * gray_height
        or len(edge_values) != edge_width * edge_height
    ):
        raise ValueError("calibration fingerprint arrays are invalid")
    gray = np.asarray(gray_values, dtype=np.float32).reshape((gray_height, gray_width))
    edge = np.asarray(edge_values, dtype=np.float32).reshape((edge_height, edge_width))
    if np.any(~np.isfinite(gray)) or np.any(~np.isfinite(edge)):
        raise ValueError("calibration fingerprint contains non-finite values")
    return gray, edge


def load_reference_fingerprint(path: str | Path) -> dict[str, object]:
    profile = Path(path).resolve()
    try:
        payload = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read calibration profile {profile}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration profile must be a JSON object")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or not isinstance(calibration.get("reference_fingerprint"), dict):
        raise ValueError("calibration profile has no approved reference fingerprint")
    _reference_arrays(calibration["reference_fingerprint"])
    return calibration["reference_fingerprint"]


def _zone_mask(zone: Sequence[NormalizedPoint], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    points = np.asarray(
        [[round(x * (width - 1)), round(y * (height - 1))] for x, y in zone],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


@dataclass(frozen=True, slots=True)
class SceneQualityAssessment:
    acceptable: bool
    drift_score: float
    tub_changed_fraction: float
    serving_changed_fraction: float
    reasons: tuple[str, ...]


class SceneQualityGuard:
    """Compare every frame to an immutable approved view and suppress on drift."""

    def __init__(
        self,
        reference: dict[str, object],
        *,
        tub_zone: Iterable[Sequence[float]],
        serving_zone: Iterable[Sequence[float]],
        drift_threshold: float = 0.20,
        zone_change_threshold: float = 0.45,
        obstruction_seconds: float = 1.0,
        pixel_change_threshold: int = 25,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not 0 < drift_threshold <= 1 or not 0 < zone_change_threshold <= 1:
            raise ValueError("scene quality thresholds must be in (0, 1]")
        if obstruction_seconds < 0:
            raise ValueError("obstruction_seconds cannot be negative")
        if not 0 <= pixel_change_threshold <= 255:
            raise ValueError("pixel_change_threshold must be between 0 and 255")
        self._reference_gray, self._reference_edges = _reference_arrays(reference)
        self._tub_mask = _zone_mask(tuple((float(x), float(y)) for x, y in tub_zone), self._reference_gray.shape)
        self._serving_mask = _zone_mask(tuple((float(x), float(y)) for x, y in serving_zone), self._reference_gray.shape)
        self.drift_threshold = drift_threshold
        self.zone_change_threshold = zone_change_threshold
        self.obstruction_seconds = obstruction_seconds
        self.pixel_change_threshold = pixel_change_threshold
        self._monotonic = monotonic or time.monotonic
        self._bad_since: float | None = None

    def reset(self) -> None:
        self._bad_since = None

    def assess(self, frame: np.ndarray, *, timestamp_seconds: float | None = None) -> SceneQualityAssessment:
        gray = _gray_frame(frame, (self._reference_gray.shape[1], self._reference_gray.shape[0]))
        edges = cv2.resize(cv2.Canny(gray, 50, 150), (self._reference_edges.shape[1], self._reference_edges.shape[0]), interpolation=cv2.INTER_AREA)
        drift_score = float(np.mean(np.abs(gray.astype(np.float32) - self._reference_gray)) / 255.0)
        edge_score = float(np.mean(np.abs(edges.astype(np.float32) - self._reference_edges)) / 255.0)
        combined_drift = max(drift_score, edge_score)
        changed = np.abs(gray.astype(np.int16) - self._reference_gray.astype(np.int16)) >= self.pixel_change_threshold
        tub_fraction = float(np.mean(changed[self._tub_mask])) if np.any(self._tub_mask) else 0.0
        serving_fraction = float(np.mean(changed[self._serving_mask])) if np.any(self._serving_mask) else 0.0
        reasons: list[str] = []
        if combined_drift > self.drift_threshold:
            reasons.append("calibration_drift")
        if tub_fraction > self.zone_change_threshold:
            reasons.append("tub_zone_obstructed")
        if serving_fraction > self.zone_change_threshold:
            reasons.append("serving_zone_obstructed")
        now = self._monotonic() if timestamp_seconds is None else timestamp_seconds
        if reasons:
            self._bad_since = self._bad_since if self._bad_since is not None else now
        else:
            self._bad_since = None
        active = bool(reasons) and self._bad_since is not None and now - self._bad_since >= self.obstruction_seconds
        return SceneQualityAssessment(
            acceptable=not active,
            drift_score=combined_drift,
            tub_changed_fraction=tub_fraction,
            serving_changed_fraction=serving_fraction,
            reasons=tuple(reasons) if active else (),
        )
