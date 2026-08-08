"""Deterministic, auditable replay of recorded camera sessions."""

from __future__ import annotations

import hashlib
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

from ..application.quality import FrameQualityGate
from ..capture import RecordedFrameSource
from ..capture._opencv import CaptureFactory
from ..config import load_camera_config, load_service_config
from ..domain import DepositFSMConfig, ReleaseAwareDepositFSM
from ..inference import (
    RFDETRLocalAdapter,
    SupervisionByteTrackAdapter,
    observations_from_detections,
)
from ..security import sha256_file


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _event_timestamp_hash(event: dict[str, int | float]) -> str:
    return _canonical_hash(
        {key: event[key] for key in (
            "timestamp", "loaded_at", "approached_at", "released_at", "confirmed_at",
        )}
    )


def run_replay(
    video_path: str | Path,
    service_config_path: str | Path,
    camera_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 42,
    max_frames: int | None = None,
    detector_factory: Callable[[Path], object] | None = None,
    tracker_factory: Callable[[float, float], object] | None = None,
    capture_factory: CaptureFactory | None = None,
) -> dict[str, object]:
    """Replay a video and return a report whose event checksum is reproducible.

    Factories are intentionally injectable for offline integration tests; the
    default path constructs the exact production detector and ByteTrack adapters.
    """
    video = Path(video_path).resolve()
    manifest = Path(checkpoint_manifest_path).resolve()
    # Loaded for validation only; the replay identity hashes the file itself.
    load_service_config(service_config_path)
    camera_config = load_camera_config(camera_config_path)
    if not video.is_file():
        raise ValueError(f"replay video not found: {video}")
    if camera_config.tub_zone is None or camera_config.serving_zone is None:
        raise ValueError("replay camera config requires tub and serving zones")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    _seed_everything(seed)
    detector = detector_factory(manifest) if detector_factory else RFDETRLocalAdapter(manifest)
    confidence = float(getattr(getattr(detector, "manifest", None), "confidence_threshold", camera_config.event.minimum_confidence))
    tracker = (
        tracker_factory(camera_config.analysis_fps, confidence)
        if tracker_factory
        else SupervisionByteTrackAdapter(
            frame_rate=camera_config.analysis_fps,
            confidence_threshold=confidence,
        )
    )
    fsm = ReleaseAwareDepositFSM(DepositFSMConfig(
        minimum_confidence=camera_config.event.minimum_confidence,
        approach_distance=camera_config.event.approach_distance,
        exit_distance=camera_config.event.exit_distance,
        minimum_approach_seconds=camera_config.event.minimum_approach_seconds,
        minimum_release_seconds=camera_config.event.minimum_release_seconds,
        sequence_timeout_seconds=camera_config.event.sequence_timeout_seconds,
        missing_tolerance_seconds=camera_config.event.missing_tolerance_seconds,
        tub_zone=camera_config.tub_zone,
        serving_zone=camera_config.serving_zone,
    ))
    quality = FrameQualityGate(
        minimum_blur_variance=camera_config.quality.minimum_blur_variance,
        pixel_change_threshold=camera_config.quality.pixel_change_threshold,
        maximum_changed_fraction=camera_config.quality.maximum_changed_fraction,
        analysis_width=camera_config.quality.analysis_width,
    )
    source_id = camera_config.camera_id
    events: list[dict[str, object]] = []
    latencies: list[float] = []
    frames_read = 0
    frames_processed = 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    tracemalloc.start()
    with RecordedFrameSource(video, source_id=source_id, capture_factory=capture_factory) as reader:
        next_sample = 0.0
        while max_frames is None or frames_read < max_frames:
            packet = reader.read()
            if packet is None:
                break
            frames_read += 1
            if packet.timestamp_seconds + 1e-9 < next_sample:
                continue
            next_sample = packet.timestamp_seconds + 1.0 / camera_config.analysis_fps
            if not quality.assess(packet.frame).acceptable:
                continue
            started_frame = time.perf_counter()
            rgb = packet.frame[:, :, ::-1]
            detections = detector.predict(rgb, packet.timestamp_seconds)
            tracked = tracker.update(detections, packet.timestamp_seconds)
            observations = observations_from_detections(
                tracked,
                timestamp=packet.timestamp_seconds,
                frame_width=packet.frame.shape[1],
                frame_height=packet.frame.shape[0],
            )
            for event in fsm.update(packet.timestamp_seconds, observations):
                item = event.to_dict()
                item["timestamp_hash"] = _event_timestamp_hash(item)
                events.append(item)
            frames_processed += 1
            latencies.append(time.perf_counter() - started_frame)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = max(time.perf_counter() - started, 1e-9)
    sorted_latencies = sorted(latencies)
    def percentile(fraction: float) -> float:
        if not sorted_latencies:
            return 0.0
        return sorted_latencies[min(len(sorted_latencies) - 1, int((len(sorted_latencies) - 1) * fraction))]

    event_checksum = _canonical_hash(events)
    identity = {
        "input_sha256": sha256_file(video),
        "model_manifest_sha256": sha256_file(manifest),
        "config_sha256": {
            "service": sha256_file(service_config_path),
            "camera": sha256_file(camera_config_path),
        },
        "seed": seed,
        "events_sha256": event_checksum,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "input": {"path": str(video), "sha256": identity["input_sha256"]},
        "model_manifest_sha256": identity["model_manifest_sha256"],
        "config_sha256": identity["config_sha256"],
        "seed": seed,
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "event_count": len(events),
        "events_sha256": event_checksum,
        "events": events,
        "replay_checksum": _canonical_hash(identity),
        "performance": {
            "wall_seconds": elapsed,
            "cpu_seconds": time.process_time() - cpu_started,
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency_seconds": percentile(0.50),
            "p95_latency_seconds": percentile(0.95),
            "peak_python_memory_bytes": peak_memory,
            "gpu_peak_memory_bytes": None,
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return report
