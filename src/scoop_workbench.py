"""Visual development workbench for custom scoop-counting models.

This is intentionally a development tool: it shows tracked model detections,
per-container candidate counts, a human ground-truth count, and persists every
AI/manual event for later error analysis.
"""

from __future__ import annotations

import argparse
import json
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

from app import MODEL_CLASSES, detection_class_name, resize_for_display
from serving_events import Observation, ScoopEvent, ServingEventCounter
from video_source import LatestFrameSource, parse_source, safe_source_name

LOGGER = logging.getLogger("scoop-workbench")
CANONICAL_CLASSES = {"scoop", "loaded_scoop", "serving_container"}
CLASS_ALIASES = {
    "scoop": "scoop",
    "scoop_utensil": "scoop",
    "ice_cream_scoop": "scoop",
    "loaded_scoop": "loaded_scoop",
    "scoop_loaded": "loaded_scoop",
    "serving_container": "serving_container",
    "cup": "serving_container",
    "cone": "serving_container",
}
CLASS_COLORS = {
    "scoop": (80, 210, 255),
    "loaded_scoop": (40, 80, 255),
    "serving_container": (80, 255, 120),
}


def canonical_class(name: str) -> str | None:
    return CLASS_ALIASES.get(name.strip().lower())


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Visually test a scoop detector and record ground truth."
    )
    parser.add_argument("--source", default=os.getenv("CAMERA_URL", "0"))
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CLASSES),
        default=os.getenv("MODEL_SIZE", "nano").lower(),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("SCOOP_CHECKPOINT", ""),
        help="Fine-tuned RF-DETR checkpoint. Empty runs a COCO preview only.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(os.getenv("SCOOP_CONFIDENCE", "0.35")),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=float(os.getenv("TARGET_FPS", "10")),
    )
    parser.add_argument(
        "--association-distance",
        type=float,
        default=float(os.getenv("SCOOP_ASSOCIATION_DISTANCE", "140")),
        help="Maximum loaded-scoop/container center distance in source pixels.",
    )
    parser.add_argument(
        "--session-dir",
        default=os.getenv("WORKBENCH_SESSION_DIR", "artifacts/workbench"),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many inference frames; 0 means unlimited.",
    )
    parser.add_argument("--seconds", type=float, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.target_fps < 0:
        raise SystemExit("--target-fps cannot be negative")
    if args.association_distance <= 0:
        raise SystemExit("--association-distance must be positive")
    if args.seconds < 0:
        raise SystemExit("--seconds cannot be negative")
    if args.max_frames < 0:
        raise SystemExit("--max-frames cannot be negative")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")


class WorkbenchSource:
    """Preserve video-file order, but use latest-frame semantics for cameras."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        self.sequence = -1
        self.timestamp_seconds = 0.0
        self.recorded = False
        self._fps = 0.0
        self.file_capture: cv2.VideoCapture | None = None
        self.live_source: LatestFrameSource | None = None
        if isinstance(source, str) and Path(source).is_file():
            self.recorded = True
            self.file_capture = cv2.VideoCapture(source)
            if not self.file_capture.isOpened():
                raise RuntimeError(f"Could not open video file: {source}")
            self._fps = float(self.file_capture.get(cv2.CAP_PROP_FPS))
        else:
            self.live_source = LatestFrameSource(
                source,
                reconnect_seconds=float(os.getenv("RECONNECT_SECONDS", "3")),
                rtsp_transport=os.getenv("RTSP_TRANSPORT", "tcp"),
                open_timeout_ms=int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000")),
                read_timeout_ms=int(os.getenv("CAMERA_READ_TIMEOUT_MS", "5000")),
            ).start()

    def read(self, after_sequence: int) -> tuple[int, np.ndarray | None]:
        if self.file_capture is not None:
            ok, frame = self.file_capture.read()
            if not ok or frame is None:
                return after_sequence, None
            self.sequence += 1
            timestamp = float(self.file_capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= self.timestamp_seconds and self.sequence > 0 and self._fps > 0:
                timestamp = self.sequence / self._fps
            self.timestamp_seconds = max(0.0, timestamp)
            return self.sequence, frame
        assert self.live_source is not None
        sequence, frame = self.live_source.read(after_sequence, timeout=2.0)
        if frame is not None:
            self.timestamp_seconds = time.time()
        return sequence, frame

    def stop(self) -> None:
        if self.file_capture is not None:
            self.file_capture.release()
        if self.live_source is not None:
            self.live_source.stop()


class EventJournal:
    def __init__(self, session_root: Path) -> None:
        session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        self.directory = session_root / session_id
        self.evidence_directory = self.directory / "evidence"
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"

    def append(self, record: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def snapshot(self, frame: np.ndarray, prefix: str) -> str | None:
        filename = self.evidence_directory / (
            f"{prefix}-{time.strftime('%H%M%S')}-{time.time_ns() % 1_000_000:06d}.jpg"
        )
        if cv2.imwrite(str(filename), frame):
            return str(filename)
        return None


def tracked_observations(detections: sv.Detections, model: object) -> list[Observation]:
    if len(detections) == 0 or detections.tracker_id is None:
        return []
    output: list[Observation] = []
    for index, (box, confidence, class_id, tracker_id) in enumerate(
        zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
            detections.tracker_id,
        )
    ):
        source_name = detection_class_name(detections, index, model, int(class_id))
        class_name = canonical_class(source_name)
        if class_name is None:
            continue
        output.append(
            Observation(
                track_id=int(tracker_id),
                class_name=class_name,
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in box),
            )
        )
    return output


def draw_observations(frame: np.ndarray, observations: list[Observation]) -> None:
    for item in observations:
        x1, y1, x2, y2 = (int(value) for value in item.xyxy)
        color = CLASS_COLORS[item.class_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{item.class_name} #{item.track_id} {item.confidence:.0%}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_dashboard(
    frame: np.ndarray,
    counter: ServingEventCounter,
    ground_truth_total: int,
    checkpoint_loaded: bool,
    last_message: str,
) -> None:
    panel_width = min(frame.shape[1], 760)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    mode = "CUSTOM CHECKPOINT" if checkpoint_loaded else "COCO PREVIEW - AI COUNT DISABLED"
    lines = [
        f"{mode}",
        f"AI scoop candidates: {counter.total_count} | Ground truth: {ground_truth_total}",
        "Containers: "
        + (
            ", ".join(
                f"#{state.track_id}={state.scoop_count}"
                for state in sorted(counter.containers.values(), key=lambda x: x.track_id)
            )
            or "none"
        ),
        "G: ground-truth +1 | U: undo | R: reset | S: snapshot | Q/Esc: quit",
        last_message,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, 26 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def record_ai_event(
    journal: EventJournal,
    event: ScoopEvent,
    frame: np.ndarray,
    timestamp_domain: str,
) -> None:
    evidence = journal.snapshot(frame, f"ai-event-{event.event_id:04d}")
    journal.append(
        {
            "type": "ai_scoop_candidate",
            **event.to_dict(),
            "timestamp_domain": timestamp_domain,
            "evidence": evidence,
        }
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    source = parse_source(args.source)
    checkpoint_loaded = bool(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict[str, object] = {"device": device}
    if checkpoint_loaded:
        model_kwargs["pretrain_weights"] = args.checkpoint
    LOGGER.info("Loading RF-DETR %s on %s", args.model, device)
    model = MODEL_CLASSES[args.model](**model_kwargs)

    tracker_fps = args.target_fps if args.target_fps else 30.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The `ByteTrack` was deprecated.*")
        tracker = sv.ByteTrack(
            track_activation_threshold=args.confidence,
            lost_track_buffer=max(15, int(tracker_fps * 1.5)),
            minimum_matching_threshold=0.8,
            frame_rate=tracker_fps,
            minimum_consecutive_frames=2,
        )
    counter = ServingEventCounter(
        association_distance_pixels=args.association_distance,
        minimum_confidence=args.confidence,
    )
    journal = EventJournal(Path(args.session_dir))
    journal.append(
        {
            "type": "session_started",
            "timestamp": time.time(),
            "source": safe_source_name(source),
            "checkpoint": str(Path(args.checkpoint).name) if args.checkpoint else None,
            "model": args.model,
        }
    )
    camera = WorkbenchSource(source)
    LOGGER.info("Workbench session: %s", journal.directory)

    started = time.monotonic()
    last_sequence = -1
    last_inference_at = 0.0
    ground_truth_total = 0
    last_message = "Waiting for detections"
    received_frame = False
    processed_frames = 0
    try:
        while True:
            sequence, frame = camera.read(last_sequence)
            if frame is None:
                if isinstance(source, str) and Path(source).is_file():
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                continue
            received_frame = True
            last_sequence = sequence
            now = time.monotonic()
            interval = (
                1.0 / args.target_fps
                if args.target_fps and not camera.recorded
                else 0.0
            )
            if interval and now - last_inference_at < interval:
                continue
            last_inference_at = now

            detections = model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                threshold=args.confidence,
                include_source_image=False,
            )
            tracked = tracker.update_with_detections(detections)
            processed_frames += 1
            observations = tracked_observations(tracked, model)
            event_timestamp = camera.timestamp_seconds
            events = counter.update(observations, timestamp=event_timestamp) if checkpoint_loaded else []
            annotated = frame.copy()
            draw_observations(annotated, observations)
            for event in events:
                last_message = (
                    f"SCOOP +1 -> container #{event.container_track_id} "
                    f"({event.confidence:.0%})"
                )
                record_ai_event(
                    journal,
                    event,
                    annotated,
                    "media_time" if camera.recorded else "utc_epoch",
                )
                LOGGER.info("%s", last_message)
            draw_dashboard(
                annotated,
                counter,
                ground_truth_total,
                checkpoint_loaded,
                last_message,
            )
            key = 255
            if not args.headless:
                cv2.imshow(
                    "Scoop AI development workbench",
                    resize_for_display(annotated, int(os.getenv("DISPLAY_WIDTH", "1280"))),
                )
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                ground_truth_total += 1
                evidence = journal.snapshot(annotated, "ground-truth")
                journal.append(
                    {
                        "type": "ground_truth_scoop",
                        "timestamp": event_timestamp,
                        "timestamp_domain": "media_time" if camera.recorded else "utc_epoch",
                        "ground_truth_total": ground_truth_total,
                        "ai_total": counter.total_count,
                        "evidence": evidence,
                    }
                )
                last_message = "Ground truth +1 recorded"
            elif key == ord("u"):
                ground_truth_total = max(0, ground_truth_total - 1)
                journal.append(
                    {
                        "type": "ground_truth_undo",
                        "timestamp": event_timestamp,
                        "timestamp_domain": "media_time" if camera.recorded else "utc_epoch",
                        "ground_truth_total": ground_truth_total,
                    }
                )
                last_message = "Ground truth count corrected"
            elif key == ord("r"):
                counter.reset()
                ground_truth_total = 0
                journal.append(
                    {
                        "type": "counts_reset",
                        "timestamp": event_timestamp,
                        "timestamp_domain": "media_time" if camera.recorded else "utc_epoch",
                    }
                )
                last_message = "Counts reset"
            elif key == ord("s"):
                path = journal.snapshot(annotated, "manual-snapshot")
                last_message = f"Snapshot: {Path(path).name if path else 'failed'}"

            elapsed = camera.timestamp_seconds if camera.recorded else now - started
            if args.seconds and elapsed >= args.seconds:
                break
            if args.max_frames and processed_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        LOGGER.info("Stopping")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        journal.append(
            {
                "type": "session_finished",
                "timestamp": time.time(),
                "ai_total": counter.total_count,
                "ground_truth_total": ground_truth_total,
                "received_frame": received_frame,
            }
        )
    if not received_frame:
        LOGGER.error("No readable frame was received from %s", safe_source_name(source))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
