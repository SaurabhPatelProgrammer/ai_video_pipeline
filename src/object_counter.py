"""Count tracked objects that cross a virtual line in a camera stream."""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from dotenv import load_dotenv

from app import (
    MODEL_CLASSES,
    as_bool,
    detection_class_name,
    resize_for_display,
)
from video_source import LatestFrameSource, parse_source, safe_source_name

LOGGER = logging.getLogger("object-counter")


def parse_line(
    value: str,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Parse ``x1,y1,x2,y2`` or return a centered vertical line."""
    if not value.strip():
        center_x = frame_width // 2
        return center_x, 0, center_x, frame_height - 1

    try:
        coordinates = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("line must contain four integer coordinates") from exc

    if len(coordinates) != 4:
        raise ValueError("line must use x1,y1,x2,y2 format")

    x1, y1, x2, y2 = coordinates
    if not (
        0 <= x1 < frame_width
        and 0 <= x2 < frame_width
        and 0 <= y1 < frame_height
        and 0 <= y2 < frame_height
    ):
        raise ValueError(
            f"line must stay inside the {frame_width}x{frame_height} source frame"
        )
    if (x1, y1) == (x2, y2):
        raise ValueError("line start and end cannot be the same point")
    return x1, y1, x2, y2


def filter_detections(
    detections: sv.Detections,
    model: object,
    class_filter: set[str],
) -> sv.Detections:
    """Keep only configured class names while preserving detection metadata."""
    if not class_filter or len(detections) == 0:
        return detections
    keep = np.array(
        [
            detection_class_name(detections, index, model, int(class_id)).lower()
            in class_filter
            for index, class_id in enumerate(detections.class_id)
        ],
        dtype=bool,
    )
    return detections[keep]


def crossing_total(line_zone: sv.LineZone) -> int:
    return int(line_zone.in_count + line_zone.out_count)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Track recognized objects and count virtual-line crossings."
    )
    parser.add_argument("--source", default=os.getenv("CAMERA_URL", "0"))
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CLASSES),
        default=os.getenv("MODEL_SIZE", "nano").lower(),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(os.getenv("CONFIDENCE", "0.40")),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=float(os.getenv("TARGET_FPS", "10")),
    )
    parser.add_argument(
        "--classes",
        default=os.getenv("COUNT_CLASSES", os.getenv("CLASS_FILTER", "")),
        help="Comma-separated class names; empty counts every recognized class.",
    )
    parser.add_argument(
        "--line",
        default=os.getenv("COUNT_LINE", ""),
        help="x1,y1,x2,y2 in source pixels; empty uses a centered vertical line.",
    )
    parser.add_argument(
        "--minimum-crossing-frames",
        type=int,
        default=int(os.getenv("MINIMUM_CROSSING_FRAMES", "2")),
        help="Stable frames required on each side before a crossing is counted.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0, help="0 means run until Q.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model not in MODEL_CLASSES:
        raise SystemExit(
            f"--model must be one of: {', '.join(sorted(MODEL_CLASSES))}."
        )
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1.")
    if args.target_fps < 0:
        raise SystemExit("--target-fps cannot be negative.")
    if args.seconds < 0:
        raise SystemExit("--seconds cannot be negative.")
    if args.minimum_crossing_frames < 1:
        raise SystemExit("--minimum-crossing-frames must be at least 1.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    source = parse_source(args.source)
    recorded_file = isinstance(source, str) and Path(source).is_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_filter = {
        item.strip().lower() for item in args.classes.split(",") if item.strip()
    }
    show_window = as_bool(os.getenv("SHOW_WINDOW", "true")) and not args.headless
    display_width = int(os.getenv("DISPLAY_WIDTH", "1280"))
    reconnect_seconds = float(os.getenv("RECONNECT_SECONDS", "3"))
    snapshot_dir = Path(os.getenv("SNAPSHOT_DIR", "artifacts/snapshots"))

    LOGGER.info("Loading RF-DETR %s on %s.", args.model, device)
    model = MODEL_CLASSES[args.model](device=device)

    tracker_fps = args.target_fps if args.target_fps > 0 else 30.0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The `ByteTrack` was deprecated.*",
            category=FutureWarning,
        )
        tracker = sv.ByteTrack(
            track_activation_threshold=args.confidence,
            lost_track_buffer=max(15, int(tracker_fps * 1.5)),
            minimum_matching_threshold=0.8,
            frame_rate=tracker_fps,
            minimum_consecutive_frames=2,
        )

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    line_annotator = sv.LineZoneAnnotator(
        custom_in_text="IN",
        custom_out_text="OUT",
    )

    camera = LatestFrameSource(
        source,
        reconnect_seconds=reconnect_seconds,
        rtsp_transport=os.getenv("RTSP_TRANSPORT", "tcp"),
        open_timeout_ms=int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000")),
        read_timeout_ms=int(os.getenv("CAMERA_READ_TIMEOUT_MS", "5000")),
    ).start()
    LOGGER.info("Opening %s", safe_source_name(source))

    started = time.monotonic()
    last_sequence = -1
    last_inference_at = 0.0
    line_zone: sv.LineZone | None = None
    line_coordinates: tuple[int, int, int, int] | None = None

    try:
        while True:
            sequence, frame = camera.read(last_sequence, timeout=2.0)
            if frame is None:
                if recorded_file and camera.recorded_eof:
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                continue
            last_sequence = sequence

            now = time.monotonic()
            minimum_interval = (
                1.0 / args.target_fps
                if args.target_fps and not recorded_file
                else 0.0
            )
            if minimum_interval and now - last_inference_at < minimum_interval:
                continue
            last_inference_at = now

            if line_zone is None:
                try:
                    line_coordinates = parse_line(
                        args.line,
                        frame.shape[1],
                        frame.shape[0],
                    )
                except ValueError as exc:
                    raise SystemExit(f"Invalid --line: {exc}") from exc
                x1, y1, x2, y2 = line_coordinates
                line_zone = sv.LineZone(
                    start=sv.Point(x1, y1),
                    end=sv.Point(x2, y2),
                    minimum_crossing_threshold=args.minimum_crossing_frames,
                )
                LOGGER.info("Counting line: (%d,%d) -> (%d,%d)", x1, y1, x2, y2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = model.predict(
                rgb,
                threshold=args.confidence,
                include_source_image=False,
            )
            detections = filter_detections(detections, model, class_filter)
            tracked = tracker.update_with_detections(detections)
            crossed_in, crossed_out = line_zone.trigger(tracked)

            names = [
                detection_class_name(tracked, index, model, int(class_id))
                for index, class_id in enumerate(tracked.class_id)
            ]
            labels = [
                f"{name} #{int(tracker_id)} {float(confidence):.0%}"
                for name, tracker_id, confidence in zip(
                    names,
                    tracked.tracker_id,
                    tracked.confidence,
                )
            ]

            for index in np.flatnonzero(crossed_in):
                LOGGER.info(
                    "COUNT IN | total=%d | class=%s | track=%d",
                    crossing_total(line_zone),
                    names[index],
                    int(tracked.tracker_id[index]),
                )
            for index in np.flatnonzero(crossed_out):
                LOGGER.info(
                    "COUNT OUT | total=%d | class=%s | track=%d",
                    crossing_total(line_zone),
                    names[index],
                    int(tracked.tracker_id[index]),
                )

            annotated = frame.copy()
            annotated = box_annotator.annotate(annotated, tracked)
            annotated = label_annotator.annotate(annotated, tracked, labels=labels)
            annotated = line_annotator.annotate(annotated, line_zone)
            cv2.putText(
                annotated,
                f"TOTAL CROSSINGS: {crossing_total(line_zone)}",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            if show_window:
                cv2.imshow(
                    "Object crossing counter - Q quit, S snapshot",
                    resize_for_display(annotated, display_width),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    filename = snapshot_dir / time.strftime(
                        "object-count-%Y%m%d-%H%M%S.jpg"
                    )
                    if cv2.imwrite(str(filename), annotated):
                        LOGGER.info("Snapshot saved: %s", filename)
                    else:
                        LOGGER.error("Snapshot could not be saved: %s", filename)

            elapsed = (
                camera.media_timestamp_seconds
                if recorded_file
                else now - started
            )
            if args.seconds and elapsed is not None and elapsed >= args.seconds:
                break
    except KeyboardInterrupt:
        LOGGER.info("Stopping.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if line_zone is None:
        LOGGER.error("No readable camera frame was received.")
        return 1

    LOGGER.info(
        "Final count | IN=%d | OUT=%d | TOTAL=%d",
        line_zone.in_count,
        line_zone.out_count,
        crossing_total(line_zone),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
