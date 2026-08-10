"""Build a leakage-safe COCO dataset from reviewed ice-cream-item labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAME = "ice_cream_item"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument(
        "--split-mode",
        choices=("source-session", "positive-cluster"),
        default="source-session",
        help="Split whole videos or temporally isolated positive event clusters.",
    )
    parser.add_argument(
        "--event-gap",
        type=float,
        default=3.0,
        help="Maximum gap in seconds between positive frames in one event cluster.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _session_assignments(
    records: list[dict[str, object]], items: dict[str, dict[str, object]]
) -> dict[str, str]:
    positives = Counter(
        str(item["source_session"])
        for item in items.values()
        if item.get("status") == "annotated"
    )
    sessions = sorted({str(record["source_session"]) for record in records})
    sessions_with_positives = [session for session in sessions if positives[session] > 0]
    if len(sessions_with_positives) < 3:
        raise ValueError("At least three source sessions with positive labels are required")
    ranked = sorted(sessions_with_positives, key=lambda value: (-positives[value], value))
    assignments = {ranked[0]: "train", ranked[1]: "valid", ranked[2]: "test"}
    for session in ranked[3:]:
        assignments[session] = "train"
    return assignments


def _positive_cluster_records(
    records: list[dict[str, object]],
    items: dict[str, dict[str, object]],
    event_gap: float,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Turn separated positive moments into leakage-safe logical sessions."""
    if event_gap <= 0:
        raise ValueError("event_gap must be positive")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_session"])].append(record)

    event_records: list[dict[str, object]] = []
    assignments: dict[str, str] = {}
    for source_session, source_records in sorted(grouped.items()):
        ordered = sorted(source_records, key=lambda item: float(item["timestamp_seconds"]))
        positives = [
            record
            for record in ordered
            if items[str(record["image"])]["status"] == "annotated"
        ]
        clusters: list[list[dict[str, object]]] = []
        for record in positives:
            if (
                not clusters
                or float(record["timestamp_seconds"])
                - float(clusters[-1][-1]["timestamp_seconds"])
                > event_gap
            ):
                clusters.append([record])
            else:
                clusters[-1].append(record)
        if not clusters:
            continue

        cluster_splits = ["train"] * len(clusters)
        if len(clusters) >= 2:
            cluster_splits[-1] = "valid"
        if len(clusters) >= 3:
            cluster_splits[-2] = "test"

        cluster_centers = [
            sum(float(record["timestamp_seconds"]) for record in cluster) / len(cluster)
            for cluster in clusters
        ]
        records_by_cluster: dict[int, list[dict[str, object]]] = defaultdict(list)
        for record in ordered:
            cluster_index = min(
                range(len(clusters)),
                key=lambda index: abs(float(record["timestamp_seconds"]) - cluster_centers[index]),
            )
            records_by_cluster[cluster_index].append(record)

        for index, cluster in enumerate(clusters):
            start_ms = round(float(cluster[0]["timestamp_seconds"]) * 1000)
            logical_session = f"{source_session}--event-{start_ms:09d}ms"
            assignments[logical_session] = cluster_splits[index]
            positive_paths = {str(record["image"]) for record in cluster}
            candidates = records_by_cluster[index]
            segment_start = min(float(record["timestamp_seconds"]) for record in candidates)
            segment_end = max(float(record["timestamp_seconds"]) for record in candidates)
            selected_paths = positive_paths | {
                str(record["image"])
                for record in candidates
                if items[str(record["image"])]["status"] == "negative"
            }
            for record in candidates:
                if str(record["image"]) not in selected_paths:
                    continue
                logical_record = dict(record)
                logical_record["source_capture_session"] = source_session
                logical_record["source_session"] = logical_session
                logical_record["source_segment_start_seconds"] = segment_start
                logical_record["source_segment_end_seconds"] = segment_end
                logical_record["assigned_split"] = cluster_splits[index]
                event_records.append(logical_record)
    return event_records, assignments


def _sample_negatives(
    negatives: list[dict[str, object]],
    positive_times: list[float],
    limit: int,
) -> list[dict[str, object]]:
    if len(negatives) <= limit:
        return negatives
    ordered = sorted(negatives, key=lambda item: float(item["timestamp_seconds"]))
    nearest = sorted(
        ordered,
        key=lambda item: (
            min(abs(float(item["timestamp_seconds"]) - value) for value in positive_times),
            hashlib.sha256(str(item["image"]).encode("utf-8")).hexdigest(),
        ),
    )
    hard_count = limit // 2
    selected = {str(item["image"]): item for item in nearest[:hard_count]}
    remaining = limit - len(selected)
    if remaining > 0:
        step = len(ordered) / remaining
        for index in range(remaining):
            candidate = ordered[min(len(ordered) - 1, int((index + 0.5) * step))]
            selected.setdefault(str(candidate["image"]), candidate)
    if len(selected) < limit:
        for candidate in nearest:
            selected.setdefault(str(candidate["image"]), candidate)
            if len(selected) == limit:
                break
    return sorted(selected.values(), key=lambda item: float(item["timestamp_seconds"]))


def build_dataset(
    captures: Path,
    annotation_path: Path,
    output: Path,
    *,
    negative_ratio: float,
    split_mode: str = "source-session",
    event_gap: float = 3.0,
) -> dict[str, object]:
    captures = captures.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output dataset is immutable and already exists: {output}")
    if negative_ratio < 0:
        raise ValueError("negative_ratio cannot be negative")
    records = _load_jsonl(captures / "manifest.jsonl")
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation_data.get("class_names") != [CLASS_NAME]:
        raise ValueError(f"Annotations must contain only {CLASS_NAME!r}")
    items: dict[str, dict[str, object]] = annotation_data["items"]
    record_by_path = {str(record["image"]): record for record in records}
    if set(items) != set(record_by_path):
        raise ValueError("Annotation paths do not exactly match the capture manifest")

    if split_mode == "positive-cluster":
        records, assignments = _positive_cluster_records(records, items, event_gap)
    elif split_mode == "source-session":
        assignments = _session_assignments(records, items)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    selected_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for session, split in assignments.items():
        session_records = [record for record in records if str(record["source_session"]) == session]
        positive_records = [
            record for record in session_records if items[str(record["image"])]["status"] == "annotated"
        ]
        negative_records = [
            record for record in session_records if items[str(record["image"])]["status"] == "negative"
        ]
        positive_times = [float(record["timestamp_seconds"]) for record in positive_records]
        negative_limit = math.ceil(len(positive_records) * negative_ratio)
        selected_negatives = _sample_negatives(negative_records, positive_times, negative_limit)
        selected_by_split[split].extend(positive_records + selected_negatives)

    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, object] = {
        "class_names": [CLASS_NAME],
        "negative_ratio": negative_ratio,
        "split_mode": split_mode,
        "event_gap_seconds": event_gap if split_mode == "positive-cluster" else None,
        "session_assignments": assignments,
        "splits": {},
    }
    annotation_id = 1
    for split in SPLITS:
        split_directory = output / split
        split_directory.mkdir()
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        split_records = sorted(
            selected_by_split[split],
            key=lambda item: (str(item["source_session"]), float(item["timestamp_seconds"])),
        )
        positives = 0
        for image_id, record in enumerate(split_records, start=1):
            key = str(record["image"])
            source = captures / Path(key)
            destination_name = source.name
            shutil.copy2(source, split_directory / destination_name)
            images.append(
                {
                    "id": image_id,
                    "file_name": destination_name,
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "source_session": record["source_session"],
                    "source_capture_session": record.get(
                        "source_capture_session", record["source_session"]
                    ),
                    "source_video": record["source_video"],
                    "source_sha256": record["source_sha256"],
                    "timestamp_seconds": record["timestamp_seconds"],
                    **(
                        {
                            "source_segment_start_seconds": record[
                                "source_segment_start_seconds"
                            ],
                            "source_segment_end_seconds": record[
                                "source_segment_end_seconds"
                            ],
                        }
                        if "source_segment_start_seconds" in record
                        else {}
                    ),
                }
            )
            item = items[key]
            if item["status"] == "annotated":
                positives += 1
                for box in item["boxes"]:
                    x, y, width, height = (int(value) for value in box)
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x, y, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
        coco = {
            "info": {"description": "Ice cream handover item detector", "version": "1.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "product"}],
        }
        (split_directory / "_annotations.coco.json").write_text(
            json.dumps(coco, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["splits"][split] = {
            "images": len(images),
            "positive_images": positives,
            "negative_images": len(images) - positives,
            "annotations": len(annotations),
            "sessions": sorted({str(record["source_session"]) for record in split_records}),
        }
    (output / "build-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = build_dataset(
        args.captures,
        args.annotations,
        args.output,
        negative_ratio=args.negative_ratio,
        split_mode=args.split_mode,
        event_gap=args.event_gap,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
