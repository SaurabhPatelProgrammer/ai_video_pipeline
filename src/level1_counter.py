"""Run the fixed-camera Level-1 temporal motion baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from level1_motion_model import MotionCandidate, MotionScoopStateMachine
from order_serving_model import ServedOrderStateMachine
from video_source import parse_source, safe_source_name

LOGGER = logging.getLogger("level1-counter")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run an experimental tub-to-serving motion baseline."
    )
    parser.add_argument("--source", default=os.getenv("CAMERA_URL", "0"))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/level1_shop_camera.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/level1-runs"),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-output-video", action="store_true")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Recorded-video playback speed in the visual window.",
    )
    parser.add_argument("--seconds", type=float, default=0)
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Profile not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    required = {
        "working_zone",
        "tub_zone",
        "serving_zone",
        "analysis_width",
        "pixel_difference_threshold",
        "tub_motion_threshold",
        "serving_motion_threshold",
    }
    missing = required - set(profile)
    if missing:
        raise SystemExit(f"Profile is missing fields: {sorted(missing)}")
    for field in ("working_zone", "tub_zone", "serving_zone"):
        points = profile[field]
        if not isinstance(points, list) or len(points) < 3:
            raise SystemExit(f"{field} must contain at least three normalized points")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point)
                or not all(0 <= value <= 1 for value in point)
            ):
                raise SystemExit(f"{field} points must be normalized [x,y] pairs")
    return profile


def open_capture(source: int | str) -> cv2.VideoCapture:
    source_text = str(source).lower()
    if source_text.startswith(("rtsp://", "rtsps://")):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{os.getenv('RTSP_TRANSPORT', 'tcp')}"
        )
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000")),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(os.getenv("CAMERA_READ_TIMEOUT_MS", "5000")),
            ],
        )
    return cv2.VideoCapture(source)


def polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [
                min(width - 1, max(0, int(round(x * width)))),
                min(height - 1, max(0, int(round(y * height)))),
            ]
            for x, y in points
        ],
        dtype=np.int32,
    )


def polygon_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def motion_ratio(binary_motion: np.ndarray, mask: np.ndarray) -> float:
    area = cv2.countNonZero(mask)
    if area == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(binary_motion, mask)) / area


def prepare_gray(frame: np.ndarray, analysis_width: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, analysis_width / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0), scale


def fit_for_display(
    frame: np.ndarray,
    maximum_width: int,
    maximum_height: int,
) -> np.ndarray:
    """Fit the complete frame inside the screen without cropping it."""
    if maximum_width <= 0 or maximum_height <= 0:
        return frame
    scale = min(
        1.0,
        maximum_width / frame.shape[1],
        maximum_height / frame.shape[0],
    )
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (
            max(1, int(round(frame.shape[1] * scale))),
            max(1, int(round(frame.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def draw_zone(
    frame: np.ndarray,
    normalized_points: list[list[float]],
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = polygon(normalized_points, frame.shape[1], frame.shape[0])
    cv2.polylines(frame, [points], True, color, 2)
    anchor = tuple(points[0])
    cv2.putText(
        frame,
        label,
        (int(anchor[0]), max(22, int(anchor[1]) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(
    frame: np.ndarray,
    model: MotionScoopStateMachine | ServedOrderStateMachine,
    candidate_count: int,
    ground_truth_count: int,
    tub_motion: float,
    serving_motion: float,
    message: str,
    event_mode: str = "scoop_transfer",
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(820, frame.shape[1]), 142), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    served_order = event_mode == "served_order"
    lines = [
        "LEVEL-1 SERVED-ORDER BASELINE" if served_order else "LEVEL-1 EXPERIMENTAL MOTION BASELINE",
        f"Candidates: {candidate_count} | Ground truth: {ground_truth_count} | State: {model.state.value}",
        f"Tub motion: {tub_motion:.3f}/{model.tub_threshold:.3f} | Serving motion: {serving_motion:.3f}/{model.serving_threshold:.3f}",
        (
            "G: completed order served +1 | U: undo | R: reset | Q/Esc: quit"
            if served_order
            else "G: true scoop +1 | U: undo | R: reset | Q/Esc: quit"
        ),
        message,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, 25 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def save_evidence(directory: Path, frame: np.ndarray, index: int) -> str:
    path = directory / f"candidate-{index:04d}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not write {path}")
    return path.name


def main() -> int:
    args = parse_args()
    if args.seconds < 0:
        raise SystemExit("--seconds cannot be negative")
    if args.playback_speed <= 0:
        raise SystemExit("--playback-speed must be positive")
    profile = load_profile(args.profile)
    source = parse_source(args.source)
    recorded_file = isinstance(source, str) and Path(source).is_file()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    capture = open_capture(source)
    if not capture.isOpened():
        LOGGER.error("Could not open %s", safe_source_name(source))
        return 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 12.0
    session = args.output_dir / (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
    )
    evidence_directory = session / "evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)

    event_mode = str(profile.get("event_mode", "scoop_transfer"))
    if event_mode == "served_order":
        state_machine = ServedOrderStateMachine(
            tub_threshold=float(profile["tub_motion_threshold"]),
            serving_threshold=float(profile["serving_motion_threshold"]),
            minimum_loading_frames=int(profile.get("minimum_loading_frames", 3)),
            minimum_serving_frames=int(profile.get("minimum_serving_frames", 2)),
            minimum_preparation_seconds=float(profile.get("minimum_preparation_seconds", 20.0)),
            order_timeout_seconds=float(profile.get("order_timeout_seconds", 45.0)),
            cooldown_seconds=float(profile.get("cooldown_seconds", 20.0)),
            nominal_fps=fps,
        )
    elif event_mode == "scoop_transfer":
        state_machine = MotionScoopStateMachine(
            tub_threshold=float(profile["tub_motion_threshold"]),
            serving_threshold=float(profile["serving_motion_threshold"]),
            minimum_loading_frames=int(profile.get("minimum_loading_frames", 3)),
            minimum_serving_frames=int(profile.get("minimum_serving_frames", 3)),
            minimum_transfer_seconds=float(profile.get("minimum_transfer_seconds", 0.25)),
            transfer_timeout_seconds=float(profile.get("transfer_timeout_seconds", 5.0)),
            cooldown_seconds=float(profile.get("cooldown_seconds", 1.25)),
        )
    else:
        raise SystemExit(f"Unsupported event_mode: {event_mode}")
    writer: cv2.VideoWriter | None = None
    previous_gray: np.ndarray | None = None
    working_mask: np.ndarray | None = None
    tub_mask: np.ndarray | None = None
    serving_mask: np.ndarray | None = None
    events: list[dict[str, object]] = []
    ground_truth_events: list[float] = []
    ground_truth_count = 0
    candidate_count = 0
    frame_index = 0
    started = time.monotonic()
    message = (
        "Waiting for order preparation then customer handoff"
        if event_mode == "served_order"
        else "Waiting for a tub-to-serving motion sequence"
    )
    received_frame = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            received_frame = True
            frame_index += 1
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= 0:
                timestamp = time.monotonic() - started
            gray, scale = prepare_gray(frame, int(profile["analysis_width"]))
            height, width = gray.shape
            if working_mask is None:
                working_mask = polygon_mask(
                    polygon(profile["working_zone"], width, height), width, height
                )
                tub_mask = polygon_mask(
                    polygon(profile["tub_zone"], width, height), width, height
                )
                serving_mask = polygon_mask(
                    polygon(profile["serving_zone"], width, height), width, height
                )
                if not args.no_output_video:
                    writer = cv2.VideoWriter(
                        str(session / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create annotated output video")

            tub_motion = 0.0
            serving_motion = 0.0
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                _, binary = cv2.threshold(
                    difference,
                    int(profile["pixel_difference_threshold"]),
                    255,
                    cv2.THRESH_BINARY,
                )
                binary = cv2.morphologyEx(
                    binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
                )
                binary = cv2.bitwise_and(binary, working_mask)
                assert tub_mask is not None and serving_mask is not None
                tub_motion = motion_ratio(binary, tub_mask)
                serving_motion = motion_ratio(binary, serving_mask)
                candidate = state_machine.update(tub_motion, serving_motion, timestamp)
            else:
                candidate = None
            previous_gray = gray

            annotated = frame.copy()
            draw_zone(
                annotated,
                profile["tub_zone"],
                (20, 190, 255),
                "CONTAINER / PREP" if event_mode == "served_order" else "TUB/LOAD ZONE",
            )
            draw_zone(
                annotated,
                profile["serving_zone"],
                (60, 255, 80),
                "CUSTOMER / HANDOFF" if event_mode == "served_order" else "SERVING ZONE",
            )
            if candidate is not None:
                candidate_count += 1
                message = (
                    f"SERVED ORDER CANDIDATE +1 at {candidate.timestamp:.2f}s"
                    if event_mode == "served_order"
                    else f"SCOOP CANDIDATE +1 at {candidate.timestamp:.2f}s"
                )
                evidence = save_evidence(evidence_directory, annotated, candidate_count)
                record = asdict(candidate)
                record.update(
                    {
                        "event_id": candidate_count,
                        "type": (
                            "level1_served_order_candidate"
                            if event_mode == "served_order"
                            else "level1_scoop_candidate"
                        ),
                        "evidence": evidence,
                    }
                )
                events.append(record)
                LOGGER.info("%s", message)
            draw_status(
                annotated,
                state_machine,
                candidate_count,
                ground_truth_count,
                tub_motion,
                serving_motion,
                message,
                event_mode,
            )
            if writer is not None:
                writer.write(annotated)

            key = 255
            if not args.headless:
                cv2.imshow(
                    "Level-1 scoop candidate counter",
                    fit_for_display(
                        annotated,
                        int(os.getenv("DISPLAY_WIDTH", "1280")),
                        int(os.getenv("DISPLAY_HEIGHT", "800")),
                    ),
                )
                delay_ms = (
                    max(1, int(round(1000.0 / fps / args.playback_speed)))
                    if recorded_file
                    else 1
                )
                key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                ground_truth_count += 1
                ground_truth_events.append(timestamp)
                message = (
                    "Ground truth served order +1"
                    if event_mode == "served_order"
                    else "Ground truth scoop +1"
                )
            elif key == ord("u"):
                ground_truth_count = max(0, ground_truth_count - 1)
                if ground_truth_events:
                    ground_truth_events.pop()
                message = "Ground truth corrected"
            elif key == ord("r"):
                candidate_count = 0
                ground_truth_count = 0
                events.clear()
                ground_truth_events.clear()
                state_machine.reset()
                message = "Scenario reset"

            if args.seconds and time.monotonic() - started >= args.seconds:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    report = {
        "level": (
            "level1_served_order_motion_baseline"
            if event_mode == "served_order"
            else "level1_experimental_motion_baseline"
        ),
        "event_mode": event_mode,
        "profile": profile.get("profile_name", args.profile.name),
        "source": safe_source_name(source),
        "received_frame": received_frame,
        "frames_processed": frame_index,
        "fps": fps,
        "candidate_count": candidate_count,
        "ground_truth_count": ground_truth_count,
        "ground_truth_timestamps": ground_truth_events,
        "events": events,
        "limitations": (
            [
                "Counts preparation-to-handoff motion candidates, not visually confirmed orders.",
                "Two labeled serves are insufficient for production accuracy.",
                "Profile is calibrated only for the supplied fixed camera view.",
            ]
            if event_mode == "served_order"
            else [
                "Counts motion candidates, not visually confirmed loaded-scoop deposits.",
                "Does not assign candidates to individual cups or cones.",
                "Profile is calibrated only for the supplied fixed camera view.",
            ]
        ),
    }
    with (session / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    LOGGER.info("Level-1 report: %s", session / "report.json")
    if not received_frame:
        LOGGER.error("No readable frame was received")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
