"""Live RF-DETR object detection for a webcam, IP camera, or video file."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from video_source import LatestFrameSource, parse_source, safe_source_name

LOGGER = logging.getLogger("camera-ai")
MODEL_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run RF-DETR on a live camera stream.")
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
        default=os.getenv("CLASS_FILTER", ""),
        help="Comma-separated class names; empty means all.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0, help="0 means run until Q.")
    return parser.parse_args()


def class_name(model: object, class_id: int) -> str:
    """Resolve a zero-based class ID for custom/legacy model outputs."""
    names = getattr(model, "class_names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def detection_class_name(
    detections: object,
    index: int,
    model: object,
    class_id: int,
) -> str:
    """Use RF-DETR's checkpoint-aware class name, with a legacy fallback.

    Pretrained COCO checkpoints return sparse category IDs (for example,
    person=1 and bus=6), while fine-tuned checkpoints normally return
    zero-based IDs. Recent RF-DETR versions resolve that distinction and attach
    the correct name to each detection in ``data["class_name"]``.
    """
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        names = data.get("class_name")
        if names is not None:
            try:
                resolved = names[index]
            except (IndexError, KeyError, TypeError):
                pass
            else:
                if resolved is not None and str(resolved).strip():
                    return str(resolved)

    return class_name(model, class_id)


def color_for(class_id: int) -> tuple[int, int, int]:
    # Stable, bright BGR color without maintaining a global palette.
    return (
        80 + (class_id * 67) % 176,
        80 + (class_id * 97) % 176,
        80 + (class_id * 37) % 176,
    )


def annotate(
    frame: np.ndarray,
    detections: object,
    model: object,
    class_filter: set[str],
    max_detections: int,
) -> tuple[np.ndarray, Counter[str]]:
    output = frame.copy()
    counts: Counter[str] = Counter()
    boxes = getattr(detections, "xyxy", np.empty((0, 4)))
    confidences = getattr(detections, "confidence", np.empty(0))
    class_ids = getattr(detections, "class_id", np.empty(0, dtype=int))

    order = np.argsort(confidences)[::-1][:max_detections]
    for index in order:
        class_id = int(class_ids[index])
        name = detection_class_name(detections, int(index), model, class_id)
        if class_filter and name.lower() not in class_filter:
            continue

        confidence = float(confidences[index])
        x1, y1, x2, y2 = (int(value) for value in boxes[index])
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))
        color = color_for(class_id)
        counts[name] += 1

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {confidence:.0%}"
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_top = max(0, y1 - text_height - 10)
        cv2.rectangle(
            output,
            (x1, label_top),
            (x1 + text_width + 8, y1),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
    return output, counts


def put_status(
    frame: np.ndarray,
    model_size: str,
    inference_ms: float,
    display_fps: float,
    counts: Counter[str],
) -> None:
    summary = ", ".join(f"{name}:{count}" for name, count in counts.most_common(5))
    lines = [
        f"RF-DETR {model_size} | inference {inference_ms:.0f} ms | {display_fps:.1f} FPS",
        summary or "No objects above threshold",
        "Q/Esc: quit | S: snapshot",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 760), 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 34 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def resize_for_display(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] <= width:
        return frame
    ratio = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * ratio)))


def main() -> int:
    args = parse_args()
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1.")
    if args.target_fps < 0:
        raise SystemExit("--target-fps cannot be negative.")

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
    max_detections = int(os.getenv("MAX_DETECTIONS", "50"))
    reconnect_seconds = float(os.getenv("RECONNECT_SECONDS", "3"))
    snapshot_dir = Path(os.getenv("SNAPSHOT_DIR", "artifacts/snapshots"))

    LOGGER.info("Device: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    LOGGER.info("Loading RF-DETR %s; first run downloads pretrained weights.", args.model)
    model = MODEL_CLASSES[args.model](device=device)
    LOGGER.info("Model ready. Classes: %d", len(getattr(model, "class_names", {})))

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
    timings: deque[float] = deque(maxlen=30)
    last_log_at = 0.0

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inference_started = time.perf_counter()
            detections = model.predict(
                rgb,
                threshold=args.confidence,
                include_source_image=False,
            )
            inference_seconds = time.perf_counter() - inference_started
            timings.append(inference_seconds)
            annotated, counts = annotate(
                frame,
                detections,
                model,
                class_filter,
                max_detections,
            )
            average_seconds = sum(timings) / len(timings)
            display_fps = 1.0 / average_seconds if average_seconds else 0.0
            put_status(
                annotated,
                args.model,
                inference_seconds * 1000,
                display_fps,
                counts,
            )

            if now - last_log_at >= 2.0:
                LOGGER.info(
                    "Inference %.0f ms | objects: %s",
                    inference_seconds * 1000,
                    dict(counts),
                )
                last_log_at = now

            if show_window:
                cv2.imshow(
                    "IP Camera AI MVP",
                    resize_for_display(annotated, display_width),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    filename = snapshot_dir / time.strftime("detection-%Y%m%d-%H%M%S.jpg")
                    cv2.imwrite(str(filename), annotated)
                    LOGGER.info("Snapshot saved: %s", filename)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
