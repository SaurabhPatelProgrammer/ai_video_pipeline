"""Deterministically extract traceable frames from one or more source videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import cv2


def safe_session_name(path: Path, source_sha256: str | None = None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-").lower()
    digest = (source_sha256 or file_sha256(path))[:12]
    return f"{stem[:48]}-{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract timestamped frames while preserving source-video identity."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/scoop-mvp/captures"),
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--minimum-change",
        type=float,
        default=2.0,
        help="Minimum mean grayscale difference from the last saved frame.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--dataset-version",
        default="",
        help="Immutable version directory name; defaults to a UTC run ID.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = [str(path) for path in args.videos if not path.is_file()]
    if missing:
        raise SystemExit("Video file(s) not found: " + ", ".join(missing))
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.minimum_change < 0:
        raise SystemExit("--minimum-change cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")


def extract_video(
    video_path: Path,
    output_root: Path,
    interval: float,
    minimum_change: float,
    jpeg_quality: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_sha256 = file_sha256(video_path)
    session_id = safe_session_name(video_path, source_sha256)
    image_directory = output_root / "images" / session_id
    image_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    records: list[dict[str, object]] = []
    previous_gray = None
    timestamp = 0.0
    saved = 0
    try:
        while timestamp <= duration + 1e-6:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            change_score = (
                float(cv2.absdiff(gray, previous_gray).mean())
                if previous_gray is not None
                else 255.0
            )
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if previous_gray is None or change_score >= minimum_change:
                filename = f"{session_id}-{int(round(timestamp * 1000)):09d}ms.jpg"
                relative_path = Path("images") / session_id / filename
                destination = output_root / relative_path
                if not cv2.imwrite(
                    str(destination),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                ):
                    raise RuntimeError(f"Could not write {destination}")
                records.append(
                    {
                        "image": relative_path.as_posix(),
                        "source_session": session_id,
                        "source_video": video_path.name,
                        "source_sha256": source_sha256,
                        "timestamp_seconds": round(timestamp, 3),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "change_score": round(change_score, 3),
                        "blur_score": round(blur_score, 3),
                    }
                )
                previous_gray = gray
                saved += 1
            timestamp += interval
    finally:
        capture.release()
    source = {
        "source_session": session_id,
        "source_video": video_path.name,
        "source_sha256": source_sha256,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "saved_frames": saved,
    }
    return records, source


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_version = args.dataset_version.strip() or time.strftime("v-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", dataset_version):
        raise SystemExit("--dataset-version must be 3-64 safe filename characters")
    output_root = args.output / dataset_version
    if output_root.exists():
        raise SystemExit(f"Dataset version already exists and is immutable: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    for path in args.videos:
        records, source = extract_video(
            path.resolve(),
            output_root,
            args.interval,
            args.minimum_change,
            args.jpeg_quality,
        )
        manifest_records.extend(records)
        source_records.append(source)
        print(
            f"{path.name}: {source['saved_frames']} frames from "
            f"{source['duration_seconds']:.1f}s"
        )
    for record in manifest_records:
        record["dataset_version"] = dataset_version
    with (output_root / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_root / "sources.json").open("x", encoding="utf-8") as handle:
        json.dump(source_records, handle, indent=2, sort_keys=True)
    print(f"Saved {len(manifest_records)} traceable frames in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
