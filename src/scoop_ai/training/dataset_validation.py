"""Strong COCO and source-provenance validation before GPU training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


REQUIRED_CLASSES = ("scoop", "loaded_scoop", "serving_container")
REQUIRED_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True, slots=True)
class DatasetValidationOptions:
    require_provenance: bool = True
    require_timestamp: bool = True
    require_source_sha256: bool = True
    hash_images: bool = True
    require_all_classes_per_split: bool = True
    required_classes: tuple[str, ...] = REQUIRED_CLASSES


@dataclass(frozen=True, slots=True)
class SplitSummary:
    images: int
    annotations: int
    negative_images: int
    class_counts: dict[str, int]
    sessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    root: Path
    fingerprint_sha256: str
    splits: dict[str, SplitSummary]


class DatasetValidationError(ValueError):
    """One or more actionable dataset integrity violations."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("Dataset validation failed:\n- " + "\n- ".join(issues))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_image_path(split_directory: Path, file_name: object) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string")
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image path: {file_name!r}")
    root = split_directory.resolve()
    image_path = (root / relative).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes split directory: {file_name!r}") from exc
    return image_path


def _load_coco(annotation_path: Path, issues: list[str], split: str) -> dict[str, Any] | None:
    if not annotation_path.is_file():
        issues.append(f"{split}: missing {annotation_path.name}")
        return None
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"{split}: invalid COCO JSON: {exc}")
        return None
    if not isinstance(data, dict):
        issues.append(f"{split}: COCO root must be an object")
        return None
    return data


def _unique_integer_id_map(
    records: object,
    *,
    kind: str,
    split: str,
    issues: list[str],
) -> dict[int, dict[str, Any]]:
    if not isinstance(records, list):
        issues.append(f"{split}: {kind} must be a list")
        return {}
    output: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(f"{split}: {kind}[{index}] must be an object")
            continue
        value = record.get("id")
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(f"{split}: {kind}[{index}] has a non-integer id")
            continue
        if value in output:
            issues.append(f"{split}: duplicate {kind} id {value}")
            continue
        output[value] = record
    return output


def _validate_categories(
    data: dict[str, Any],
    split: str,
    issues: list[str],
    required_classes: tuple[str, ...],
) -> dict[int, str]:
    records = _unique_integer_id_map(
        data.get("categories"), kind="categories", split=split, issues=issues
    )
    names: dict[int, str] = {}
    seen_names: set[str] = set()
    for category_id, record in records.items():
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(f"{split}: category {category_id} has an empty name")
            continue
        normalized = name.strip()
        if normalized in seen_names:
            issues.append(f"{split}: duplicate category name {normalized!r}")
        seen_names.add(normalized)
        names[category_id] = normalized
    if set(names.values()) != set(required_classes):
        issues.append(
            f"{split}: classes must be exactly {list(required_classes)!r}; "
            f"received {sorted(names.values())!r}"
        )
    return names


def _validate_image_record(
    image_id: int,
    record: dict[str, Any],
    *,
    split: str,
    split_directory: Path,
    options: DatasetValidationOptions,
    issues: list[str],
) -> tuple[Path | None, str | None, float | None, str | None]:
    width = record.get("width")
    height = record.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        issues.append(f"{split}: image {image_id} width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        issues.append(f"{split}: image {image_id} height must be a positive integer")
    try:
        image_path = _safe_image_path(split_directory, record.get("file_name"))
    except ValueError as exc:
        issues.append(f"{split}: image {image_id}: {exc}")
        image_path = None
    if image_path is not None and not image_path.is_file():
        issues.append(f"{split}: image {image_id} file not found: {record.get('file_name')}")

    session = record.get("source_session")
    if not isinstance(session, str) or not session.strip():
        if options.require_provenance:
            issues.append(f"{split}: image {image_id} is missing source_session")
        normalized_session = None
    else:
        normalized_session = session.strip()

    timestamp_value = record.get("timestamp_seconds")
    timestamp: float | None = None
    if timestamp_value is None:
        if options.require_timestamp:
            issues.append(f"{split}: image {image_id} is missing timestamp_seconds")
    else:
        try:
            timestamp = float(timestamp_value)
        except (TypeError, ValueError):
            issues.append(f"{split}: image {image_id} timestamp_seconds is not numeric")
        else:
            if not isfinite(timestamp) or timestamp < 0:
                issues.append(
                    f"{split}: image {image_id} timestamp_seconds must be finite and non-negative"
                )

    source_sha = record.get("source_sha256")
    normalized_sha: str | None = None
    if source_sha is None:
        if options.require_source_sha256:
            issues.append(f"{split}: image {image_id} is missing source_sha256")
    else:
        normalized_sha = str(source_sha).strip().lower()
        if len(normalized_sha) != 64 or any(ch not in "0123456789abcdef" for ch in normalized_sha):
            issues.append(f"{split}: image {image_id} source_sha256 is invalid")
            normalized_sha = None
    return image_path, normalized_session, timestamp, normalized_sha


def _validate_source_segment(
    image_id: int,
    record: dict[str, Any],
    *,
    split: str,
    timestamp: float | None,
    issues: list[str],
) -> tuple[float, float] | None:
    start_value = record.get("source_segment_start_seconds")
    end_value = record.get("source_segment_end_seconds")
    if start_value is None and end_value is None:
        return None
    if start_value is None or end_value is None:
        issues.append(f"{split}: image {image_id} source segment requires both start and end")
        return None
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        issues.append(f"{split}: image {image_id} source segment bounds must be numeric")
        return None
    if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
        issues.append(f"{split}: image {image_id} source segment bounds are invalid")
        return None
    if timestamp is not None and not (start <= timestamp <= end):
        issues.append(f"{split}: image {image_id} timestamp falls outside its source segment")
        return None
    return start, end


def _valid_bbox(
    annotation: dict[str, Any], image: dict[str, Any]
) -> tuple[bool, str | None]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "bbox must contain [x, y, width, height]"
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False, "bbox coordinates must be numeric"
    if not all(isfinite(value) for value in (x, y, width, height)):
        return False, "bbox coordinates must be finite"
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return False, "bbox must have non-negative origin and positive size"
    image_width = image.get("width")
    image_height = image.get("height")
    if isinstance(image_width, int) and isinstance(image_height, int):
        epsilon = 1e-6
        if x + width > image_width + epsilon or y + height > image_height + epsilon:
            return False, "bbox extends outside image bounds"
    area = annotation.get("area")
    if area is not None:
        try:
            numeric_area = float(area)
        except (TypeError, ValueError):
            return False, "area must be numeric"
        if not isfinite(numeric_area) or numeric_area <= 0:
            return False, "area must be finite and positive"
    return True, None


def validate_dataset(
    dataset_root: Path,
    *,
    options: DatasetValidationOptions | None = None,
) -> DatasetValidationReport:
    """Validate all COCO splits, provenance, leakage, and duplicate content."""

    config = options or DatasetValidationOptions()
    if not config.required_classes or len(config.required_classes) != len(set(config.required_classes)):
        raise DatasetValidationError(["required_classes must contain unique class names"])
    root = Path(dataset_root)
    issues: list[str] = []
    if not root.is_dir():
        raise DatasetValidationError([f"dataset directory not found: {root}"])

    summaries: dict[str, SplitSummary] = {}
    session_owner: dict[str, str] = {}
    session_source_hash: dict[str, str] = {}
    source_hash_sessions: dict[
        str, dict[str, tuple[str, tuple[float, float] | None]]
    ] = {}
    content_owner: dict[str, tuple[str, str]] = {}
    fingerprint = hashlib.sha256()

    for split in REQUIRED_SPLITS:
        split_directory = root / split
        data = _load_coco(split_directory / "_annotations.coco.json", issues, split)
        if data is None:
            continue
        fingerprint.update(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        categories = _validate_categories(data, split, issues, config.required_classes)
        images = _unique_integer_id_map(
            data.get("images"), kind="images", split=split, issues=issues
        )
        annotations = _unique_integer_id_map(
            data.get("annotations"), kind="annotations", split=split, issues=issues
        )
        if not images:
            issues.append(f"{split}: split has no images")

        sessions: set[str] = set()
        file_names: set[str] = set()
        annotated_image_ids: set[int] = set()
        class_counts: Counter[str] = Counter()
        for image_id, image in images.items():
            file_name = image.get("file_name")
            if isinstance(file_name, str):
                normalized_name = Path(file_name).as_posix().casefold()
                if normalized_name in file_names:
                    issues.append(f"{split}: duplicate image file_name {file_name!r}")
                file_names.add(normalized_name)
            image_path, session, timestamp, source_sha = _validate_image_record(
                image_id,
                image,
                split=split,
                split_directory=split_directory,
                options=config,
                issues=issues,
            )
            segment = _validate_source_segment(
                image_id,
                image,
                split=split,
                timestamp=timestamp,
                issues=issues,
            )
            if session is not None:
                sessions.add(session)
                previous_split = session_owner.setdefault(session, split)
                if previous_split != split:
                    issues.append(
                        f"source session {session!r} leaks across {previous_split} and {split}"
                    )
                if source_sha is not None:
                    previous_sha = session_source_hash.setdefault(session, source_sha)
                    if previous_sha != source_sha:
                        issues.append(
                            f"source session {session!r} has inconsistent source_sha256 values"
                        )
                    source_sessions = source_hash_sessions.setdefault(source_sha, {})
                    previous_source = source_sessions.setdefault(session, (split, segment))
                    if previous_source != (split, segment):
                        issues.append(
                            f"source session {session!r} has inconsistent segment metadata"
                        )
            if config.hash_images and image_path is not None and image_path.is_file():
                content_hash = _file_sha256(image_path)
                fingerprint.update(content_hash.encode("ascii"))
                previous = content_owner.setdefault(
                    content_hash, (split, str(image.get("file_name")))
                )
                if previous != (split, str(image.get("file_name"))):
                    issues.append(
                        f"duplicate image content: {previous[0]}/{previous[1]} and "
                        f"{split}/{image.get('file_name')}"
                    )

        for annotation_id, annotation in annotations.items():
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id not in images:
                issues.append(
                    f"{split}: annotation {annotation_id} references unknown image_id {image_id!r}"
                )
                continue
            if (
                isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or category_id not in categories
            ):
                issues.append(
                    f"{split}: annotation {annotation_id} references unknown category_id "
                    f"{category_id!r}"
                )
            else:
                class_counts[categories[category_id]] += 1
            valid, reason = _valid_bbox(annotation, images[image_id])
            if not valid:
                issues.append(f"{split}: annotation {annotation_id}: {reason}")
            annotated_image_ids.add(image_id)
            iscrowd = annotation.get("iscrowd", 0)
            if iscrowd not in (0, 1):
                issues.append(f"{split}: annotation {annotation_id} iscrowd must be 0 or 1")

        if config.require_all_classes_per_split:
            missing_classes = set(config.required_classes) - set(class_counts)
            if missing_classes:
                issues.append(
                    f"{split}: no annotations for classes {sorted(missing_classes)!r}"
                )
        summaries[split] = SplitSummary(
            images=len(images),
            annotations=len(annotations),
            negative_images=len(set(images) - annotated_image_ids),
            class_counts=dict(sorted(class_counts.items())),
            sessions=tuple(sorted(sessions)),
        )

    for source_sha, sessions in source_hash_sessions.items():
        if len(sessions) < 2:
            continue
        if any(details[1] is None for details in sessions.values()):
            names = ", ".join(sorted(repr(name) for name in sessions))
            issues.append(
                "source_sha256 is assigned to multiple sessions without auditable "
                f"non-overlapping source segments: {source_sha[:12]}... ({names})"
            )
            continue
        ordered_segments = sorted(
            (details[1][0], details[1][1], session, details[0])
            for session, details in sessions.items()
            if details[1] is not None
        )
        for previous, current in zip(ordered_segments, ordered_segments[1:]):
            if current[0] <= previous[1]:
                issues.append(
                    "source segments overlap across sessions: "
                    f"{previous[2]!r} ({previous[3]}) and {current[2]!r} ({current[3]})"
                )

    if issues:
        raise DatasetValidationError(issues)
    return DatasetValidationReport(
        root=root.resolve(),
        fingerprint_sha256=fingerprint.hexdigest(),
        splits=summaries,
    )
