"""Validate and fine-tune RF-DETR Nano on a versioned scoop dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rfdetr import RFDETRNano
from scoop_ai.inference import create_checkpoint_manifest
from scoop_ai.training import DatasetValidationOptions
from scoop_ai.training import validate_dataset as validate_production_dataset

REQUIRED_CLASSES = {"scoop", "loaded_scoop", "serving_container"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom scoop detector.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/scoop-rfdetr-nano"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--classes",
        default=",".join(sorted(REQUIRED_CLASSES)),
        help="Comma-separated detector classes; defaults to the legacy scoop classes.",
    )
    return parser.parse_args()


def inspect_split(
    dataset: Path, split: str, required_classes: set[str] | None = None
) -> dict[str, object]:
    expected = required_classes or REQUIRED_CLASSES
    split_directory = dataset / split
    annotation_path = split_directory / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise ValueError(f"Missing {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    category_names = set(categories.values())
    if category_names != expected:
        raise ValueError(
            f"{split} classes must be exactly {sorted(expected)}; "
            f"received {sorted(category_names)}"
        )
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not images:
        raise ValueError(f"{split} has no images")
    missing_images = [
        item["file_name"]
        for item in images
        if not (split_directory / item["file_name"]).is_file()
    ]
    if missing_images:
        raise ValueError(f"{split} references {len(missing_images)} missing image(s)")
    class_counts = Counter(categories.get(int(item["category_id"]), "unknown") for item in annotations)
    empty_classes = expected - set(class_counts)
    if empty_classes:
        raise ValueError(f"{split} has no annotations for: {sorted(empty_classes)}")
    sessions = {
        str(item["source_session"])
        for item in images
        if item.get("source_session") is not None
    }
    return {
        "images": len(images),
        "annotations": len(annotations),
        "class_counts": dict(sorted(class_counts.items())),
        "sessions": sessions,
    }


def validate_dataset(
    dataset: Path, required_classes: set[str] | None = None
) -> dict[str, dict[str, object]]:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset}")
    summary = {
        split: inspect_split(dataset, split, required_classes)
        for split in ("train", "valid", "test")
    }
    session_owners: dict[str, str] = {}
    for split, details in summary.items():
        for session in details["sessions"]:
            previous = session_owners.setdefault(session, split)
            if previous != split:
                raise ValueError(
                    f"Source session {session!r} leaks across {previous} and {split}"
                )
    return summary


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    args = parse_args()
    classes = tuple(dict.fromkeys(value.strip() for value in args.classes.split(",") if value.strip()))
    if not classes:
        raise SystemExit("--classes must contain at least one class")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise SystemExit("epochs, batch-size, and grad-accum-steps must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.resume and not args.resume.is_file():
        raise SystemExit(f"Resume checkpoint not found: {args.resume}")
    try:
        validation_report = validate_production_dataset(
            args.dataset,
            options=DatasetValidationOptions(required_classes=classes),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc
    printable = {
        split: asdict(details)
        for split, details in validation_report.splits.items()
    }
    printable["dataset_fingerprint_sha256"] = validation_report.fingerprint_sha256
    print(json.dumps(printable, indent=2, sort_keys=True))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(printable, handle, indent=2, sort_keys=True)

    model = RFDETRNano(resolution=args.resolution)
    train_args: dict[str, object] = {
        "dataset_dir": str(args.dataset.resolve()),
        "output_dir": str(args.output.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.learning_rate,
        "seed": 42,
        "num_workers": 0,
        "tensorboard": False,
    }
    if args.resume:
        train_args["resume"] = str(args.resume.resolve())
    model.train(**train_args)
    checkpoints = sorted(
        args.output.glob("*.pth"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not checkpoints:
        raise SystemExit("Training finished without a .pth checkpoint")
    manifest = create_checkpoint_manifest(
        checkpoints[0],
        args.output / "model-manifest.json",
        architecture="nano",
        dataset_version=validation_report.fingerprint_sha256,
        model_version=f"scoop-rfdetr-nano-{time.strftime('%Y%m%d-%H%M%S')}",
        input_resolution=args.resolution,
        classes=classes,
    )
    print(f"Verified model manifest: {args.output / 'model-manifest.json'}")
    print(f"Checkpoint SHA-256: {manifest.checkpoint_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
