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

    assignments = _session_assignments(records, items)
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
                    "source_video": record["source_video"],
                    "source_sha256": record["source_sha256"],
                    "timestamp_seconds": record["timestamp_seconds"],
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
