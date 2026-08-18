"""Offline generic ice-cream handover replay with annotated evidence."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..capture import RecordedFrameSource
from ..domain import HandoverFSMConfig, IceCreamHandoverFSM
from ..inference import ProximityTrackerAdapter, RFDETRLocalAdapter, observations_from_detections
from ..security import sha256_file


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_zones(profile_path: Path) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    try:
        pickup = tuple((float(x), float(y)) for x, y in data["tub_zone"])
        customer = tuple((float(x), float(y)) for x, y in data["serving_zone"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("zone profile must contain numeric tub_zone and serving_zone polygons") from exc
    return pickup, customer


def _pixel_polygon(
    zone: tuple[tuple[float, float], ...], width: int, height: int
) -> np.ndarray:
    return np.array(
        [[round(x * width), round(y * height)] for x, y in zone],
        dtype=np.int32,
    )


def _draw_zone(
    frame: np.ndarray,
    zone: tuple[tuple[float, float], ...],
    color: tuple[int, int, int],
    label: str,
) -> None:
    points = _pixel_polygon(zone, frame.shape[1], frame.shape[0])
    overlay = frame.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.polylines(frame, [points], True, color, 2)
    x, y = (int(value) for value in points[0])
    cv2.putText(
        frame,
        label,
        (max(3, x), max(18, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_status(
    frame: np.ndarray,
    *,
    timestamp: float,
    count: int,
    processed: int,
    event_flash: bool,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(frame.shape[1], 360), 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    color = (60, 255, 80) if event_flash else (255, 255, 255)
    cv2.putText(
        frame,
        f"HANDOVERS: {count}  TIME: {timestamp:.1f}s",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"processed frames: {processed}",
        (8, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def run_handover_replay(
    video_path: str | Path,
    checkpoint_manifest_path: str | Path,
    zone_profile_path: str | Path,
    output_directory: str | Path,
    *,
    analysis_fps: float = 6.0,
    minimum_confidence: float | None = None,
    minimum_transfer_seconds: float = 0.20,
    minimum_customer_dwell_seconds: float = 0.12,
    minimum_customer_observations: int = 2,
    minimum_movement_distance: float = 0.03,
    sequence_timeout_seconds: float = 5.0,
    missing_tolerance_seconds: float = 1.0,
    duplicate_cooldown_seconds: float = 3.5,
    duplicate_distance: float = 0.12,
    maximum_center_distance_pixels: float = 120.0,
    lost_track_seconds: float | None = None,
    minimum_consecutive_frames: int = 1,
    duplicate_iou_threshold: float = 0.50,
    customer_only_minimum_seconds: float = 0.50,
    customer_only_minimum_observations: int = 3,
    customer_only_minimum_movement: float = 0.025,
    customer_only_static_minimum_seconds: float = 1.50,
    customer_only_static_minimum_observations: int = 6,
    max_frames: int | None = None,
    device: str | None = None,
    detector_factory: Callable[[Path, float | None], object] | None = None,
) -> dict[str, object]:
    """Replay one video and write annotated video, evidence images, and JSON report.

    Every tracker and state-machine threshold the live service reads from camera
    TOML is a parameter here too, with the same default, so a replay accuracy
    figure describes the configuration the service will actually run. ``device``
    selects the inference device; ``None`` lets the adapter use CUDA when it is
    available and fall back to CPU otherwise. ``detector_factory`` is injectable
    so offline tests can exercise this pipeline without a checkpoint.
    """

    video = Path(video_path).resolve()
    manifest_path = Path(checkpoint_manifest_path).resolve()
    profile_path = Path(zone_profile_path).resolve()
    output = Path(output_directory).resolve()
    if not video.is_file() or not manifest_path.is_file() or not profile_path.is_file():
        raise ValueError("video, checkpoint manifest, and zone profile must exist")
    if analysis_fps <= 0:
        raise ValueError("analysis_fps must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if output.exists():
        raise FileExistsError(f"replay output already exists: {output}")
    output.mkdir(parents=True)
    evidence_directory = output / "events"
    evidence_directory.mkdir()

    pickup_zone, customer_zone = _load_zones(profile_path)
    detector = (
        detector_factory(manifest_path, minimum_confidence)
        if detector_factory is not None
        else RFDETRLocalAdapter(
            manifest_path,
            device=device,
            confidence_threshold=minimum_confidence,
            expected_classes=("ice_cream_item",),
        )
    )
    confidence = float(detector.confidence_threshold)
    # An unset lost_track_seconds keeps the historical behaviour of tying track
    # expiry to the state machine's missing tolerance.
    resolved_lost_track_seconds = (
        missing_tolerance_seconds if lost_track_seconds is None else lost_track_seconds
    )
    tracker = ProximityTrackerAdapter(
        maximum_center_distance_pixels=maximum_center_distance_pixels,
        lost_track_seconds=resolved_lost_track_seconds,
        minimum_consecutive_frames=minimum_consecutive_frames,
        duplicate_iou_threshold=duplicate_iou_threshold,
    )
    fsm = IceCreamHandoverFSM(
        HandoverFSMConfig(
            pickup_zone=pickup_zone,
            customer_zone=customer_zone,
            minimum_confidence=confidence,
            minimum_transfer_seconds=minimum_transfer_seconds,
            minimum_customer_dwell_seconds=minimum_customer_dwell_seconds,
            minimum_customer_observations=minimum_customer_observations,
            minimum_movement_distance=minimum_movement_distance,
            sequence_timeout_seconds=sequence_timeout_seconds,
            missing_tolerance_seconds=missing_tolerance_seconds,
            duplicate_cooldown_seconds=duplicate_cooldown_seconds,
            duplicate_distance=duplicate_distance,
            customer_only_minimum_seconds=customer_only_minimum_seconds,
            customer_only_minimum_observations=customer_only_minimum_observations,
            customer_only_minimum_movement=customer_only_minimum_movement,
            customer_only_static_minimum_seconds=customer_only_static_minimum_seconds,
            customer_only_static_minimum_observations=(
                customer_only_static_minimum_observations
            ),
        )
    )

    events: list[dict[str, int | float | str]] = []
    frames_read = 0
    frames_processed = 0
    latencies: list[float] = []
    last_tracked = []
    pending_evidence: list[str] = []
    event_flash_until = -1.0
    annotated_path = output / "annotated.mp4"
    writer = None
    started = time.perf_counter()
    with RecordedFrameSource(video, source_id=video.stem) as reader:
        next_sample = 0.0
        while max_frames is None or frames_read < max_frames:
            packet = reader.read()
            if packet is None:
                break
            frames_read += 1
            frame = packet.frame.copy()
            if writer is None:
                source_fps = reader.fps if reader.fps > 0 else analysis_fps
                writer = cv2.VideoWriter(
                    str(annotated_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not create annotated video: {annotated_path}")

            if packet.timestamp_seconds + 1e-9 >= next_sample:
                next_sample = packet.timestamp_seconds + 1.0 / analysis_fps
                frame_started = time.perf_counter()
                detections = detector.predict(
                    cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB),
                    packet.timestamp_seconds,
                )
                last_tracked = list(tracker.update(detections, packet.timestamp_seconds))
                observations = observations_from_detections(
                    last_tracked,
                    timestamp=packet.timestamp_seconds,
                    frame_width=packet.frame.shape[1],
                    frame_height=packet.frame.shape[0],
                )
                emitted = fsm.update(packet.timestamp_seconds, observations)
                frames_processed += 1
                latencies.append(time.perf_counter() - frame_started)
                for event in emitted:
                    item = event.to_dict()
                    item["timestamp_hash"] = _canonical_hash(item)
                    evidence_name = f"event-{event.event_id:04d}-{event.timestamp:.3f}s.jpg"
                    item["evidence_file"] = f"events/{evidence_name}"
                    events.append(item)
                    # Every emitted event gets its own image from the frame that
                    # confirmed it; several events can share one frame.
                    pending_evidence.append(evidence_name)
                    event_flash_until = packet.timestamp_seconds + 1.0

            _draw_zone(frame, pickup_zone, (0, 190, 255), "PICKUP")
            _draw_zone(frame, customer_zone, (60, 255, 80), "CUSTOMER")
            for detection in last_tracked:
                x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
                track_id = int(detection.track_id) if detection.track_id is not None else -1
                state = fsm.state_for(track_id).value
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
                cv2.putText(
                    frame,
                    f"item #{track_id} {detection.confidence:.2f} {state}",
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            _draw_status(
                frame,
                timestamp=packet.timestamp_seconds,
                count=fsm.total_count,
                processed=frames_processed,
                event_flash=packet.timestamp_seconds <= event_flash_until,
            )
            for evidence_name in pending_evidence:
                evidence_path = evidence_directory / evidence_name
                if not evidence_path.exists() and not cv2.imwrite(str(evidence_path), frame):
                    raise RuntimeError(f"could not write event evidence: {evidence_path}")
            pending_evidence.clear()
            writer.write(frame)
    if writer is not None:
        writer.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    sorted_latencies = sorted(latencies)
    percentile = lambda fraction: (
        sorted_latencies[min(len(sorted_latencies) - 1, int((len(sorted_latencies) - 1) * fraction))]
        if sorted_latencies
        else 0.0
    )
    # Every threshold that can change the event list belongs here: the report
    # checksum is only meaningful if it covers the whole configuration.
    settings = {
        "analysis_fps": analysis_fps,
        "minimum_confidence": confidence,
        "minimum_transfer_seconds": minimum_transfer_seconds,
        "minimum_customer_dwell_seconds": minimum_customer_dwell_seconds,
        "minimum_customer_observations": minimum_customer_observations,
        "minimum_movement_distance": minimum_movement_distance,
        "sequence_timeout_seconds": sequence_timeout_seconds,
        "missing_tolerance_seconds": missing_tolerance_seconds,
        "duplicate_cooldown_seconds": duplicate_cooldown_seconds,
        "duplicate_distance": duplicate_distance,
        "maximum_center_distance_pixels": maximum_center_distance_pixels,
        "lost_track_seconds": resolved_lost_track_seconds,
        "minimum_consecutive_frames": minimum_consecutive_frames,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "customer_only_minimum_seconds": customer_only_minimum_seconds,
        "customer_only_minimum_observations": customer_only_minimum_observations,
        "customer_only_minimum_movement": customer_only_minimum_movement,
        "customer_only_static_minimum_seconds": customer_only_static_minimum_seconds,
        "customer_only_static_minimum_observations": (
            customer_only_static_minimum_observations
        ),
        "max_frames": max_frames,
    }
    identity = {
        "video_sha256": sha256_file(video),
        "manifest_sha256": sha256_file(manifest_path),
        "zone_profile_sha256": sha256_file(profile_path),
        "events_sha256": _canonical_hash(events),
        "settings": settings,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "video": str(video),
        "video_sha256": identity["video_sha256"],
        "checkpoint_manifest": str(manifest_path),
        "model_version": detector.manifest.model_version,
        "zone_profile": str(profile_path),
        "analysis_fps": analysis_fps,
        "minimum_confidence": confidence,
        "settings": settings,
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "event_count": len(events),
        "events": events,
        "events_sha256": identity["events_sha256"],
        "replay_checksum": _canonical_hash(identity),
        "annotated_video": str(annotated_path),
        "performance": {
            "wall_seconds": elapsed,
            "processed_fps": frames_processed / elapsed,
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency_seconds": percentile(0.50),
            "p95_latency_seconds": percentile(0.95),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
