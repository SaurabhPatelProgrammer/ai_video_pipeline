"""Train a fixed-camera motion profile from traceable extracted frames."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class SessionMotionSummary:
    session_id: str
    source_sha256: str
    frames: int
    transitions: int
    tub_median: float
    tub_p90: float
    tub_p95: float
    serving_median: float
    serving_p90: float
    serving_p95: float


@dataclass(frozen=True)
class MotionBaselineTrainingResult:
    profile_path: Path
    manifest_path: Path
    profile_sha256: str
    tub_motion_threshold: float
    serving_motion_threshold: float
    sessions: tuple[SessionMotionSummary, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc


def _validate_points(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} must contain at least three normalized points")
    points: list[list[float]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(not isinstance(item, (int, float)) for item in point)
            or any(not math.isfinite(float(item)) or not 0 <= float(item) <= 1 for item in point)
        ):
            raise ValueError(f"{name} contains an invalid normalized point")
        points.append([float(point[0]), float(point[1])])
    return points


def _mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    polygon = np.array(
        [
            [
                min(width - 1, max(0, int(round(x * width)))),
                min(height - 1, max(0, int(round(y * height)))),
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, [polygon], 255)
    if cv2.countNonZero(output) == 0:
        raise ValueError("training zone has zero area")
    return output


def _prepare_gray(image: np.ndarray, analysis_width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("training image must be BGR")
    scale = min(1.0, analysis_width / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _motion_ratio(binary: np.ndarray, mask: np.ndarray) -> float:
    area = cv2.countNonZero(mask)
    return cv2.countNonZero(cv2.bitwise_and(binary, mask)) / area


def _percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("motion distribution is empty or non-finite")
    return float(np.quantile(array, quantile))


def _threshold(values: list[float], activation_quantile: float) -> float:
    # A non-zero floor avoids learning camera compression noise as activity.
    # The p95 cap prevents a single busy session from making the detector inert.
    learned = min(
        _percentile(values, activation_quantile),
        _percentile(values, 0.95),
    )
    return round(max(0.005, learned), 6)


def train_motion_baseline(
    captures_root: str | Path,
    base_profile_path: str | Path,
    output_path: str | Path,
    *,
    activation_quantile: float = 0.80,
) -> MotionBaselineTrainingResult:
    """Learn compatible motion thresholds and write an auditable profile."""

    if not 0.5 <= activation_quantile <= 0.95:
        raise ValueError("activation_quantile must be between 0.5 and 0.95")
    captures = Path(captures_root).resolve()
    profile_path = Path(base_profile_path).resolve()
    output = Path(output_path).resolve()
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    for generated in (output, manifest_path):
        if generated.exists():
            raise ValueError(f"refusing to overwrite immutable model artifact: {generated}")

    profile_data = _load_json(profile_path)
    if not isinstance(profile_data, dict):
        raise ValueError("base profile must be a JSON object")
    working_points = _validate_points(profile_data.get("working_zone"), "working_zone")
    tub_points = _validate_points(profile_data.get("tub_zone"), "tub_zone")
    serving_points = _validate_points(profile_data.get("serving_zone"), "serving_zone")
    analysis_width = int(profile_data.get("analysis_width", 0))
    pixel_threshold = int(profile_data.get("pixel_difference_threshold", 0))
    if not 64 <= analysis_width <= 4096:
        raise ValueError("analysis_width must be between 64 and 4096")
    if not 1 <= pixel_threshold <= 255:
        raise ValueError("pixel_difference_threshold must be between 1 and 255")

    sources_data = _load_json(captures / "sources.json")
    if not isinstance(sources_data, list) or len(sources_data) < 2:
        raise ValueError("training requires at least two source sessions")
    source_hashes: dict[str, str] = {}
    for source in sources_data:
        if not isinstance(source, dict):
            raise ValueError("sources.json records must be objects")
        session_id = str(source.get("source_session", "")).strip()
        digest = str(source.get("source_sha256", "")).lower()
        if not session_id or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source record has an invalid session or SHA-256")
        if session_id in source_hashes:
            raise ValueError(f"duplicate source session: {session_id}")
        source_hashes[session_id] = digest

    records_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    legacy_records_without_digest = 0
    manifest = captures / "manifest.jsonl"
    try:
        manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read {manifest}: {exc}") from exc
    for line_number, line in enumerate(manifest_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        session_id = str(record.get("source_session", ""))
        if session_id not in source_hashes:
            raise ValueError(f"manifest line {line_number} references an unknown session")
        record_digest = record.get("source_sha256")
        if record_digest is None:
            legacy_records_without_digest += 1
        elif str(record_digest).lower() != source_hashes[session_id]:
            raise ValueError(f"manifest line {line_number} source SHA-256 mismatch")
        records_by_session[session_id].append(record)

    tub_values: list[float] = []
    serving_values: list[float] = []
    summaries: list[SessionMotionSummary] = []
    morphology_kernel = np.ones((3, 3), dtype=np.uint8)
    for session_id in sorted(source_hashes):
        records = sorted(
            records_by_session.get(session_id, []),
            key=lambda item: float(item.get("timestamp_seconds", -1)),
        )
        if len(records) < 3:
            raise ValueError(f"session {session_id} has fewer than three frames")
        session_tub: list[float] = []
        session_serving: list[float] = []
        previous: np.ndarray | None = None
        working_mask = tub_mask = serving_mask = None
        previous_timestamp = -1.0
        for record in records:
            timestamp = float(record.get("timestamp_seconds", -1))
            if not math.isfinite(timestamp) or timestamp <= previous_timestamp:
                raise ValueError(f"session {session_id} timestamps are not strictly increasing")
            previous_timestamp = timestamp
            relative = Path(str(record.get("image", "")))
            image_path = (captures / relative).resolve()
            if not image_path.is_relative_to(captures) or not image_path.is_file():
                raise ValueError(f"invalid or missing training image: {relative}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode training image: {relative}")
            gray = _prepare_gray(image, analysis_width)
            if working_mask is None:
                height, width = gray.shape
                working_mask = _mask(working_points, width, height)
                tub_mask = _mask(tub_points, width, height)
                serving_mask = _mask(serving_points, width, height)
            elif previous is not None and gray.shape != previous.shape:
                raise ValueError(f"session {session_id} changes resolution")
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                _, binary = cv2.threshold(
                    difference, pixel_threshold, 255, cv2.THRESH_BINARY
                )
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morphology_kernel)
                binary = cv2.bitwise_and(binary, working_mask)
                assert tub_mask is not None and serving_mask is not None
                session_tub.append(_motion_ratio(binary, tub_mask))
                session_serving.append(_motion_ratio(binary, serving_mask))
            previous = gray
        tub_values.extend(session_tub)
        serving_values.extend(session_serving)
        summaries.append(
            SessionMotionSummary(
                session_id=session_id,
                source_sha256=source_hashes[session_id],
                frames=len(records),
                transitions=len(session_tub),
                tub_median=round(_percentile(session_tub, 0.50), 6),
                tub_p90=round(_percentile(session_tub, 0.90), 6),
                tub_p95=round(_percentile(session_tub, 0.95), 6),
                serving_median=round(_percentile(session_serving, 0.50), 6),
                serving_p90=round(_percentile(session_serving, 0.90), 6),
                serving_p95=round(_percentile(session_serving, 0.95), 6),
            )
        )

    tub_threshold = _threshold(tub_values, activation_quantile)
    serving_threshold = _threshold(serving_values, activation_quantile)
    trained_profile = dict(profile_data)
    trained_profile.update(
        {
            "profile_name": "supplied-two-video-weak-motion-v1",
            "tub_motion_threshold": tub_threshold,
            "serving_motion_threshold": serving_threshold,
            "training": {
                "method": "unsupervised_temporal_motion_quantile",
                "activation_quantile": activation_quantile,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_profile_sha256": _sha256(profile_path),
                "frames": sum(item.frames for item in summaries),
                "transitions": sum(item.transitions for item in summaries),
                "sessions": [asdict(item) for item in summaries],
                "label_status": "unlabeled_weak_baseline",
                "provenance_status": (
                    "legacy_source_level_sha256"
                    if legacy_records_without_digest
                    else "per_frame_source_sha256"
                ),
                "legacy_records_without_source_sha256": legacy_records_without_digest,
                "production_approved": False,
                "limitations": [
                    "No human bounding boxes or event labels were available.",
                    "This profile detects tub-to-serving motion candidates, not confirmed scoops.",
                    "It must remain outside billing until reviewed pilot metrics pass.",
                ],
            },
        }
    )
    _atomic_json(output, trained_profile)
    profile_digest = _sha256(output)
    _atomic_json(
        manifest_path,
        {
            "artifact": output.name,
            "artifact_sha256": profile_digest,
            "artifact_type": "level1_motion_profile",
            "production_approved": False,
            "source_sessions": [asdict(item) for item in summaries],
        },
    )
    return MotionBaselineTrainingResult(
        profile_path=output,
        manifest_path=manifest_path,
        profile_sha256=profile_digest,
        tub_motion_threshold=tub_threshold,
        serving_motion_threshold=serving_threshold,
        sessions=tuple(summaries),
    )


def train_motion_baseline_from_videos(
    video_paths: Iterable[str | Path],
    base_profile_path: str | Path,
    output_path: str | Path,
    *,
    activation_quantile: float = 0.80,
) -> MotionBaselineTrainingResult:
    """Learn runtime-compatible thresholds from consecutive video frames."""

    if not 0.5 <= activation_quantile <= 0.95:
        raise ValueError("activation_quantile must be between 0.5 and 0.95")
    videos = tuple(Path(path).resolve() for path in video_paths)
    if len(videos) < 2:
        raise ValueError("video training requires at least two videos")
    if any(not path.is_file() for path in videos):
        missing = [str(path) for path in videos if not path.is_file()]
        raise ValueError(f"training video not found: {', '.join(missing)}")
    digests = [_sha256(path) for path in videos]
    if len(set(digests)) != len(digests):
        raise ValueError("training videos must have distinct SHA-256 digests")

    profile_path = Path(base_profile_path).resolve()
    output = Path(output_path).resolve()
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    for generated in (output, manifest_path):
        if generated.exists():
            raise ValueError(f"refusing to overwrite immutable model artifact: {generated}")
    profile_data = _load_json(profile_path)
    if not isinstance(profile_data, dict):
        raise ValueError("base profile must be a JSON object")
    working_points = _validate_points(profile_data.get("working_zone"), "working_zone")
    tub_points = _validate_points(profile_data.get("tub_zone"), "tub_zone")
    serving_points = _validate_points(profile_data.get("serving_zone"), "serving_zone")
    analysis_width = int(profile_data.get("analysis_width", 0))
    pixel_threshold = int(profile_data.get("pixel_difference_threshold", 0))
    if not 64 <= analysis_width <= 4096:
        raise ValueError("analysis_width must be between 64 and 4096")
    if not 1 <= pixel_threshold <= 255:
        raise ValueError("pixel_difference_threshold must be between 1 and 255")

    tub_values: list[float] = []
    serving_values: list[float] = []
    summaries: list[SessionMotionSummary] = []
    morphology_kernel = np.ones((3, 3), dtype=np.uint8)
    for video, digest in zip(videos, digests):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"could not open training video: {video}")
        session_tub: list[float] = []
        session_serving: list[float] = []
        previous: np.ndarray | None = None
        working_mask = tub_mask = serving_mask = None
        frames = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                frames += 1
                gray = _prepare_gray(image, analysis_width)
                if working_mask is None:
                    height, width = gray.shape
                    working_mask = _mask(working_points, width, height)
                    tub_mask = _mask(tub_points, width, height)
                    serving_mask = _mask(serving_points, width, height)
                elif previous is not None and gray.shape != previous.shape:
                    raise ValueError(f"training video changes resolution: {video}")
                if previous is not None:
                    difference = cv2.absdiff(gray, previous)
                    _, binary = cv2.threshold(
                        difference, pixel_threshold, 255, cv2.THRESH_BINARY
                    )
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morphology_kernel)
                    binary = cv2.bitwise_and(binary, working_mask)
                    assert tub_mask is not None and serving_mask is not None
                    session_tub.append(_motion_ratio(binary, tub_mask))
                    session_serving.append(_motion_ratio(binary, serving_mask))
                previous = gray
        finally:
            capture.release()
        if frames < 3:
            raise ValueError(f"training video has fewer than three readable frames: {video}")
        tub_values.extend(session_tub)
        serving_values.extend(session_serving)
        summaries.append(
            SessionMotionSummary(
                session_id=f"{video.stem[:48]}-{digest[:12]}",
                source_sha256=digest,
                frames=frames,
                transitions=len(session_tub),
                tub_median=round(_percentile(session_tub, 0.50), 6),
                tub_p90=round(_percentile(session_tub, 0.90), 6),
                tub_p95=round(_percentile(session_tub, 0.95), 6),
                serving_median=round(_percentile(session_serving, 0.50), 6),
                serving_p90=round(_percentile(session_serving, 0.90), 6),
                serving_p95=round(_percentile(session_serving, 0.95), 6),
            )
        )

    tub_threshold = _threshold(tub_values, activation_quantile)
    serving_threshold = _threshold(serving_values, activation_quantile)
    trained_profile = dict(profile_data)
    trained_profile.update(
        {
            "profile_name": "supplied-two-video-consecutive-motion-v1",
            "tub_motion_threshold": tub_threshold,
            "serving_motion_threshold": serving_threshold,
            "training": {
                "method": "unsupervised_consecutive_video_motion_quantile",
                "activation_quantile": activation_quantile,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_profile_sha256": _sha256(profile_path),
                "frames": sum(item.frames for item in summaries),
                "transitions": sum(item.transitions for item in summaries),
                "sessions": [asdict(item) for item in summaries],
                "label_status": "unlabeled_weak_baseline",
                "provenance_status": "video_sha256",
                "production_approved": False,
                "limitations": [
                    "Inputs are Level-1 annotated video copies because raw originals were unavailable.",
                    "No human bounding boxes or event labels were available.",
                    "This profile detects tub-to-serving motion candidates, not confirmed scoops.",
                    "It must remain outside billing until reviewed pilot metrics pass.",
                ],
            },
        }
    )
    _atomic_json(output, trained_profile)
    profile_digest = _sha256(output)
    _atomic_json(
        manifest_path,
        {
            "artifact": output.name,
            "artifact_sha256": profile_digest,
            "artifact_type": "level1_motion_profile",
            "production_approved": False,
            "source_sessions": [asdict(item) for item in summaries],
        },
    )
    return MotionBaselineTrainingResult(
        profile_path=output,
        manifest_path=manifest_path,
        profile_sha256=profile_digest,
        tub_motion_threshold=tub_threshold,
        serving_motion_threshold=serving_threshold,
        sessions=tuple(summaries),
    )
