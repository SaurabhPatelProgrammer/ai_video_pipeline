"""Unified command-line interface for production and diagnostic workflows."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .capture import LiveFrameSource, RecordedFrameSource
from .config import ConfigurationError, load_camera_config
from .security import resolve_credential, store_credential


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scoop-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera = subparsers.add_parser("camera-check", help="Validate a configured source")
    camera.add_argument("--camera-config", type=Path, required=True)
    camera.add_argument("--frames", type=int, default=30)
    camera.add_argument("--timeout", type=float, default=15.0)
    camera.set_defaults(handler=_camera_check)

    database = subparsers.add_parser("database-check", help="Verify the event store")
    database.add_argument("--database", type=Path, required=True)
    database.set_defaults(handler=_database_check)

    credential = subparsers.add_parser(
        "credential-set", help="Provision a camera source in the OS credential store"
    )
    credential.add_argument("--key", required=True)
    credential.set_defaults(handler=_credential_set)

    review = subparsers.add_parser("review", help="Open the local review application")
    review.add_argument("--database", type=Path, required=True)
    review.add_argument("--evidence-root", type=Path)
    review.set_defaults(handler=_review)

    service = subparsers.add_parser("service", help="Run the headless edge service")
    service.add_argument("--service-config", type=Path, required=True)
    service.add_argument("--camera-config", type=Path, required=True)
    service.add_argument("--checkpoint-manifest", type=Path, required=True)
    service.set_defaults(handler=_service)

    dataset = subparsers.add_parser("dataset", help="Validate or split versioned datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.set_defaults(handler=_dataset_validate)
    split = dataset_commands.add_parser("split")
    split.add_argument("--sessions", type=Path, required=True, help="JSON array of session IDs")
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(handler=_dataset_split)

    motion = subparsers.add_parser(
        "motion-train",
        help="Train an auditable weak motion baseline from extracted frames",
    )
    motion.add_argument("--captures", type=Path, required=True)
    motion.add_argument("--base-profile", type=Path, required=True)
    motion.add_argument("--output", type=Path, required=True)
    motion.add_argument("--activation-quantile", type=float, default=0.80)
    motion.set_defaults(handler=_motion_train)

    motion_video = subparsers.add_parser(
        "motion-video-train",
        help="Train a weak motion baseline from consecutive frames in videos",
    )
    motion_video.add_argument("--video", action="append", type=Path, required=True)
    motion_video.add_argument("--base-profile", type=Path, required=True)
    motion_video.add_argument("--output", type=Path, required=True)
    motion_video.add_argument("--activation-quantile", type=float, default=0.80)
    motion_video.set_defaults(handler=_motion_video_train)

    calibrate = subparsers.add_parser(
        "zone-calibrate",
        help="Draw container and customer polygons on a camera/video frame",
    )
    calibrate.add_argument("--source", required=True)
    calibrate.add_argument("--base-profile", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--seek-seconds", type=float, default=0.0)
    calibrate.add_argument("--force", action="store_true")
    calibrate.set_defaults(handler=_zone_calibrate)

    order_profile = subparsers.add_parser(
        "order-profile",
        help="Create an immutable served-order profile from a calibrated baseline",
    )
    order_profile.add_argument("--base-profile", type=Path, required=True)
    order_profile.add_argument("--output", type=Path, required=True)
    order_profile.add_argument("--tub-threshold", type=float, required=True)
    order_profile.add_argument("--customer-threshold", type=float, required=True)
    order_profile.add_argument("--minimum-preparation", type=float, required=True)
    order_profile.add_argument("--order-timeout", type=float, required=True)
    order_profile.add_argument("--container-frames", type=int, default=3)
    order_profile.add_argument("--customer-frames", type=int, default=2)
    order_profile.add_argument("--cooldown", type=float, default=20.0)
    order_profile.add_argument("--labeled-events", type=int, required=True)
    order_profile.set_defaults(handler=_order_profile)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate event JSON files")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument("--tolerance", type=float, default=0.75)
    evaluate.add_argument("--duration", type=float)
    evaluate.set_defaults(handler=_evaluate_events)

    manifest = subparsers.add_parser("model-manifest", help="Create a verified checkpoint bundle manifest")
    manifest.add_argument("--checkpoint", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--architecture", choices=["nano", "small", "medium", "large"], required=True)
    manifest.add_argument("--dataset-version", required=True)
    manifest.add_argument("--model-version", required=True)
    manifest.add_argument("--resolution", type=int, default=576)
    manifest.add_argument("--confidence", type=float, default=0.35)
    manifest.set_defaults(handler=_model_manifest)
    return parser


def _camera_check(args: argparse.Namespace) -> int:
    if args.frames < 1:
        raise ConfigurationError("--frames must be at least 1")
    if args.timeout <= 0:
        raise ConfigurationError("--timeout must be positive")
    config = load_camera_config(args.camera_config)
    source = config.resolve_source(credential_resolver=resolve_credential)
    received = 0
    started = time.monotonic()
    if config.mode == "recorded":
        reader = RecordedFrameSource(source, source_id=config.camera_id).start()
        try:
            while received < args.frames and time.monotonic() - started < args.timeout:
                packet = reader.read()
                if packet is None:
                    break
                received += 1
        finally:
            reader.stop()
    else:
        reader = LiveFrameSource(
            source,
            source_id=config.camera_id,
            reconnect_seconds=config.capture.reconnect_seconds,
            reconnect_max_seconds=config.capture.reconnect_max_seconds,
            reconnect_jitter_ratio=config.capture.reconnect_jitter_ratio,
            rtsp_transport=config.capture.rtsp_transport,
            open_timeout_ms=config.capture.open_timeout_ms,
            read_timeout_ms=config.capture.read_timeout_ms,
        ).start()
        sequence = -1
        try:
            while received < args.frames and time.monotonic() - started < args.timeout:
                packet = reader.read(sequence, timeout=config.capture.read_wait_seconds)
                if packet is None:
                    continue
                sequence = packet.sequence
                received += 1
        finally:
            reader.stop()
    health = reader.health
    print(
        json.dumps(
            {
                "camera_id": config.camera_id,
                "frames_received": received,
                "state": health.state.value,
                "detail": health.detail,
                "reconnect_attempts": health.reconnect_attempts,
            },
            sort_keys=True,
        )
    )
    return 0 if received >= args.frames else 1


def _database_check(args: argparse.Namespace) -> int:
    from .storage import SQLiteEventRepository

    with SQLiteEventRepository(args.database) as repository:
        healthy = repository.integrity_check()
        print(
            json.dumps(
                {
                    "database": str(args.database),
                    "integrity": "ok" if healthy else "failed",
                    "journal_mode": repository.journal_mode(),
                    "schema_version": repository.schema_version,
                },
                sort_keys=True,
            )
        )
    return 0 if healthy else 1


def _credential_set(args: argparse.Namespace) -> int:
    first = getpass.getpass("Camera source/RTSP URL: ")
    second = getpass.getpass("Confirm camera source: ")
    if first != second:
        raise ConfigurationError("credential confirmation does not match")
    store_credential(args.key, first)
    print(f"Credential {args.key!r} stored in the OS credential backend.")
    return 0


def _review(args: argparse.Namespace) -> int:
    from .review.app import run_review_app

    return run_review_app(args.database, args.evidence_root)


def _service(args: argparse.Namespace) -> int:
    from .application.service import run_service

    return run_service(
        service_config_path=args.service_config,
        camera_config_path=args.camera_config,
        checkpoint_manifest_path=args.checkpoint_manifest,
    )


def _dataset_validate(args: argparse.Namespace) -> int:
    from .training import validate_dataset

    report = validate_dataset(args.dataset)
    print(json.dumps(asdict(report), default=str, indent=2, sort_keys=True))
    return 0


def _dataset_split(args: argparse.Namespace) -> int:
    from .training import deterministic_session_split

    data = json.loads(args.sessions.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ConfigurationError("--sessions must contain a JSON array")
    assignments = deterministic_session_split(data, seed=args.seed)
    print(json.dumps(assignments, indent=2, sort_keys=True))
    return 0


def _motion_train(args: argparse.Namespace) -> int:
    from .training import train_motion_baseline

    result = train_motion_baseline(
        args.captures,
        args.base_profile,
        args.output,
        activation_quantile=args.activation_quantile,
    )
    print(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))
    return 0


def _motion_video_train(args: argparse.Namespace) -> int:
    from .training import train_motion_baseline_from_videos

    result = train_motion_baseline_from_videos(
        args.video,
        args.base_profile,
        args.output,
        activation_quantile=args.activation_quantile,
    )
    print(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))
    return 0


def _zone_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_zone_calibrator

    return run_zone_calibrator(
        source=args.source,
        base_profile_path=args.base_profile,
        output_path=args.output,
        seek_seconds=args.seek_seconds,
        overwrite=args.force,
    )


def _order_profile(args: argparse.Namespace) -> int:
    from .training import create_served_order_profile

    result = create_served_order_profile(
        args.base_profile,
        args.output,
        tub_threshold=args.tub_threshold,
        customer_threshold=args.customer_threshold,
        minimum_preparation_seconds=args.minimum_preparation,
        order_timeout_seconds=args.order_timeout,
        minimum_container_frames=args.container_frames,
        minimum_customer_frames=args.customer_frames,
        cooldown_seconds=args.cooldown,
        labeled_events=args.labeled_events,
    )
    print(json.dumps(asdict(result), default=str, indent=2, sort_keys=True))
    return 0


def _load_evaluation_events(path: Path):
    from .training import EvaluationEvent

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ConfigurationError(f"{path} must contain a JSON array")
    return [
        EvaluationEvent(
            session_id=str(item["session_id"]),
            timestamp=float(item["timestamp"]),
            container_id=item["container_id"],
        )
        for item in data
    ]


def _evaluate_events(args: argparse.Namespace) -> int:
    from .training import evaluate_events

    result = evaluate_events(
        _load_evaluation_events(args.predictions),
        _load_evaluation_events(args.truth),
        tolerance_seconds=args.tolerance,
        observed_duration_seconds=args.duration,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    metrics = result.metrics
    passed = (
        metrics.precision >= 0.95
        and metrics.recall >= 0.95
        and metrics.exact_container_count_accuracy >= 0.95
        and metrics.wrong_container_rate <= 0.02
    )
    return 0 if passed else 1


def _model_manifest(args: argparse.Namespace) -> int:
    from .inference import create_checkpoint_manifest

    manifest = create_checkpoint_manifest(
        args.checkpoint,
        args.output,
        architecture=args.architecture,
        dataset_version=args.dataset_version,
        model_version=args.model_version,
        input_resolution=args.resolution,
        confidence_threshold=args.confidence,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        args = _parser().parse_args(argv)
        return int(args.handler(args))
    except (ConfigurationError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
