"""Unified command-line interface for production and diagnostic workflows."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .capture import LiveFrameSource, RecordedFrameSource
from .config import ConfigurationError, load_camera_config
from .security import resolve_credential, store_credential
from .storage import (
    AuditLogRecord,
    BackupError,
    ModelVersionRecord,
    SQLiteEventRepository,
    export_redacted_image,
    utc_now_iso,
)


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
    credential.add_argument("--database", type=Path, required=True, help="audit database")
    credential.set_defaults(handler=_credential_set)

    review = subparsers.add_parser("review", help="Open the local review application")
    review.add_argument("--database", type=Path, required=True)
    review.add_argument("--evidence-root", type=Path)
    review.set_defaults(handler=_review)

    dashboard = subparsers.add_parser("dashboard", help="Open the local operator dashboard")
    dashboard.add_argument("--database", type=Path, required=True)
    dashboard.add_argument("--evidence-root", type=Path, required=True)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8090)
    dashboard.add_argument("--no-browser", action="store_true")
    dashboard.set_defaults(handler=_dashboard)

    product = subparsers.add_parser("product", help="Run the complete local Scoop AI client")
    product.add_argument("--product-root", type=Path, required=True)
    product.add_argument("--checkpoint-manifest", type=Path, required=True)
    product.add_argument("--host", default="127.0.0.1")
    product.add_argument("--port", type=int, default=8090)
    product.add_argument("--no-browser", action="store_true")
    product.set_defaults(handler=_product)

    service = subparsers.add_parser("service", help="Run the headless edge service")
    service.add_argument("--service-config", type=Path, required=True)
    service.add_argument("--camera-config", type=Path, required=True)
    service.add_argument("--checkpoint-manifest", type=Path, required=True)
    service.set_defaults(handler=_service)

    consistency = subparsers.add_parser(
        "consistency-check",
        help="Verify evidence file integrity against the database",
    )
    consistency.add_argument("--database", type=Path, required=True)
    consistency.add_argument("--evidence-root", type=Path, required=True)
    consistency.set_defaults(handler=_consistency_check)

    backup = subparsers.add_parser(
        "database-backup", help="Create a WAL-safe SQLite backup"
    )
    backup.add_argument("--database", type=Path, required=True)
    backup.add_argument("--output-dir", type=Path, required=True)
    backup.add_argument("--retain-generations", type=int, default=5)
    backup.set_defaults(handler=_database_backup)

    restore = subparsers.add_parser(
        "database-restore", help="Restore a verified backup to a new directory"
    )
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    restore.add_argument("--database-name", default="events.sqlite3")
    restore.add_argument("--actor", default="local-operator")
    restore.set_defaults(handler=_database_restore)

    retention = subparsers.add_parser("retention-update", help="Record a retention policy change")
    retention.add_argument("--database", type=Path, required=True)
    retention.add_argument("--actor", required=True)
    retention.add_argument("--policy", required=True)
    retention.set_defaults(handler=_retention_update)

    deletion = subparsers.add_parser("evidence-delete", help="Manually delete one evidence artifact")
    deletion.add_argument("--database", type=Path, required=True)
    deletion.add_argument("--evidence-root", type=Path, required=True)
    deletion.add_argument("--relative-path", required=True)
    deletion.add_argument("--actor", required=True)
    deletion.add_argument("--confirm", required=True, choices=["DELETE"])
    deletion.set_defaults(handler=_evidence_delete)

    evidence_export = subparsers.add_parser("evidence-export", help="Export blurred evidence imagery")
    evidence_export.add_argument("--input", type=Path, required=True)
    evidence_export.add_argument("--output", type=Path, required=True)
    evidence_export.add_argument("--box", action="append", default=[], help="normalized x1,y1,x2,y2")
    evidence_export.add_argument("--zone", action="append", default=[], help="normalized polygon JSON")
    evidence_export.add_argument("--blur-kernel", type=int, default=51)
    evidence_export.set_defaults(handler=_evidence_export)

    pilot = subparsers.add_parser(
        "generate-pilot-report", help="Generate a daily silent-pilot telemetry report"
    )
    pilot.add_argument("--database", type=Path, required=True)
    pilot.add_argument("--camera-id")
    pilot.add_argument("--output", type=Path)
    pilot.set_defaults(handler=_generate_pilot_report)

    outbox_retry = subparsers.add_parser(
        "outbox-retry", help="Re-queue failed or dead-lettered reviewed exports"
    )
    outbox_retry.add_argument("--database", type=Path, required=True)
    outbox_retry.add_argument("--event-id")
    outbox_retry.set_defaults(handler=_outbox_retry)

    readiness = subparsers.add_parser(
        "production-readiness-check", help="Evaluate final production-gate evidence"
    )
    readiness.add_argument("--service-config", type=Path, required=True)
    readiness.add_argument("--camera-config", type=Path, required=True)
    readiness.add_argument("--database", type=Path, required=True)
    readiness.add_argument("--dataset", type=Path)
    readiness.add_argument("--backup", type=Path)
    readiness.add_argument("--downstream-approved", action="store_true")
    readiness.add_argument("--output", type=Path)
    readiness.set_defaults(handler=_production_readiness)

    dataset = subparsers.add_parser("dataset", help="Validate or split versioned datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument(
        "--classes",
        default=",".join(("scoop", "loaded_scoop", "serving_container")),
        help="Comma-separated expected class names.",
    )
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
        aliases=["recalibrate"],
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
    manifest.add_argument("--approved-by")
    manifest.add_argument("--reviewer-signature")
    manifest.set_defaults(handler=_model_manifest)

    register = subparsers.add_parser("model-register", help="Register an approved model manifest")
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--database", type=Path, required=True)
    register.add_argument("--approved-reviewer", required=True)
    register.set_defaults(handler=_model_register)

    promote = subparsers.add_parser("model-promote", help="Promote an approved model for a camera")
    promote.add_argument("--manifest", type=Path, required=True)
    promote.add_argument("--database", type=Path, required=True)
    promote.add_argument("--camera-id", required=True)
    promote.add_argument("--approved-reviewer", required=True)
    promote.set_defaults(handler=_model_promote)

    rollback = subparsers.add_parser("model-rollback", help="Rollback a camera to its prior model")
    rollback.add_argument("--database", type=Path, required=True)
    rollback.add_argument("--camera-id", required=True)
    rollback.add_argument("--approved-reviewer", required=True)
    rollback.set_defaults(handler=_model_rollback)

    replay = subparsers.add_parser(
        "replay", help="Replay a recorded session and write a deterministic report"
    )
    replay.add_argument("--video", type=Path, required=True)
    replay.add_argument("--service-config", type=Path, required=True)
    replay.add_argument("--camera-config", type=Path, required=True)
    replay.add_argument("--checkpoint-manifest", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--seed", type=int, default=42)
    replay.add_argument("--max-frames", type=int)
    replay.set_defaults(handler=_replay)

    handover_replay = subparsers.add_parser(
        "handover-replay", help="Replay generic ice-cream pickup-to-customer handovers"
    )
    handover_replay.add_argument("--video", type=Path, required=True)
    handover_replay.add_argument("--checkpoint-manifest", type=Path, required=True)
    handover_replay.add_argument("--zone-profile", type=Path, required=True)
    handover_replay.add_argument("--output-dir", type=Path, required=True)
    handover_replay.add_argument("--analysis-fps", type=float, default=6.0)
    handover_replay.add_argument("--minimum-confidence", type=float)
    handover_replay.add_argument("--minimum-transfer-seconds", type=float, default=0.20)
    handover_replay.add_argument("--minimum-customer-dwell-seconds", type=float, default=0.12)
    handover_replay.add_argument("--minimum-customer-observations", type=int, default=2)
    handover_replay.add_argument("--minimum-movement-distance", type=float, default=0.03)
    handover_replay.add_argument("--sequence-timeout-seconds", type=float, default=5.0)
    handover_replay.add_argument("--missing-tolerance-seconds", type=float, default=1.0)
    handover_replay.add_argument("--duplicate-cooldown-seconds", type=float, default=3.5)
    handover_replay.add_argument("--duplicate-distance", type=float, default=0.12)
    # Tracker and customer-only thresholds. Defaults match the camera-config
    # defaults so a replay reproduces what the live service would count.
    handover_replay.add_argument("--maximum-center-distance-pixels", type=float, default=120.0)
    handover_replay.add_argument(
        "--lost-track-seconds",
        type=float,
        help="Track expiry; defaults to --missing-tolerance-seconds.",
    )
    handover_replay.add_argument("--minimum-consecutive-frames", type=int, default=1)
    handover_replay.add_argument("--duplicate-iou-threshold", type=float, default=0.50)
    handover_replay.add_argument("--customer-only-minimum-seconds", type=float, default=0.50)
    handover_replay.add_argument("--customer-only-minimum-observations", type=int, default=3)
    handover_replay.add_argument("--customer-only-minimum-movement", type=float, default=0.025)
    handover_replay.add_argument("--customer-only-static-minimum-seconds", type=float, default=1.50)
    handover_replay.add_argument(
        "--customer-only-static-minimum-observations", type=int, default=6
    )
    handover_replay.add_argument("--max-frames", type=int)
    handover_replay.add_argument(
        "--device",
        help="Inference device such as 'cuda' or 'cpu'; defaults to CUDA when available.",
    )
    handover_replay.set_defaults(handler=_handover_replay)

    truth_template = subparsers.add_parser(
        "handover-truth-template",
        help="Create a reviewer worksheet pre-filled with detected handovers",
    )
    truth_template.add_argument("--report", action="append", type=Path, required=True)
    truth_template.add_argument("--output", type=Path, required=True)
    truth_template.add_argument("--force", action="store_true")
    truth_template.set_defaults(handler=_handover_truth_template)

    handover_evaluate = subparsers.add_parser(
        "handover-evaluate",
        help="Score handover replay reports against reviewed ground truth",
    )
    handover_evaluate.add_argument("--report", action="append", type=Path, required=True)
    handover_evaluate.add_argument("--truth", type=Path, required=True)
    handover_evaluate.add_argument("--tolerance", type=float, default=7.0)
    handover_evaluate.add_argument("--output", type=Path)
    handover_evaluate.add_argument(
        "--require-precision",
        type=float,
        help="Exit non-zero when overall precision is below this value.",
    )
    handover_evaluate.add_argument(
        "--require-recall",
        type=float,
        help="Exit non-zero when overall recall is below this value.",
    )
    handover_evaluate.add_argument("--json", action="store_true", help="Print the full JSON report.")
    handover_evaluate.set_defaults(handler=_handover_evaluate)
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
    with SQLiteEventRepository(args.database) as repository:
        repository.record_audit(AuditLogRecord(
            audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(),
            actor=getpass.getuser() or "local-operator", action="credential_change",
            target=args.key, details={"operation": "set"},
        ))
    print(f"Credential {args.key!r} stored in the OS credential backend.")
    return 0


def _review(args: argparse.Namespace) -> int:
    from .review.app import run_review_app

    return run_review_app(args.database, args.evidence_root)


def _dashboard(args: argparse.Namespace) -> int:
    from .dashboard import run_dashboard

    return run_dashboard(
        args.database,
        args.evidence_root,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def _product(args: argparse.Namespace) -> int:
    from .dashboard import run_product

    return run_product(
        args.product_root,
        args.checkpoint_manifest,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def _service(args: argparse.Namespace) -> int:
    from .application.service import run_service

    return run_service(
        service_config_path=args.service_config,
        camera_config_path=args.camera_config,
        checkpoint_manifest_path=args.checkpoint_manifest,
    )


def _consistency_check(args: argparse.Namespace) -> int:
    """Verify evidence files on disk vs. database records."""
    from .storage import SQLiteEventRepository

    evidence_root = args.evidence_root.resolve()
    orphans: list[str] = []
    missing: list[str] = []
    corrupt: list[str] = []

    with SQLiteEventRepository(args.database) as repository:
        db_evidence = repository.list_all_evidence()
        db_paths = {rec.relative_path: rec for rec in db_evidence}

        # Scan disk for orphan files
        if evidence_root.is_dir():
            for path in sorted(evidence_root.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    rel = path.relative_to(evidence_root).as_posix()
                    if rel not in db_paths:
                        orphans.append(rel)

        # Check each DB record against disk
        for rec in db_evidence:
            abs_path = (evidence_root / rec.relative_path).resolve()
            if not abs_path.is_relative_to(evidence_root):
                corrupt.append(rec.relative_path)
                continue
            if not abs_path.is_file():
                missing.append(rec.relative_path)
                continue
            # Verify size + SHA-256
            digest = hashlib.sha256()
            size = 0
            with abs_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != rec.size_bytes or digest.hexdigest() != rec.sha256:
                corrupt.append(rec.relative_path)

    status = "ok" if not (orphans or missing or corrupt) else "failed"
    report = {
        "status": status,
        "orphans": orphans,
        "missing": missing,
        "corrupt": corrupt,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


def _database_backup(args: argparse.Namespace) -> int:
    from .storage import create_backup

    result = create_backup(
        args.database,
        args.output_dir,
        retain_generations=args.retain_generations,
    )
    with SQLiteEventRepository(args.database) as repository:
        repository.record_audit(AuditLogRecord(
            audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(),
            actor=getpass.getuser() or "local-operator", action="database_backup",
            target=str(result.backup_path), details={"metadata": str(result.metadata_path)},
        ))
    print(
        json.dumps(
            {
                "backup": str(result.backup_path),
                "metadata": str(result.metadata_path),
                "details": result.metadata,
                "deleted_generations": list(result.deleted_generations),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _database_restore(args: argparse.Namespace) -> int:
    from .storage import restore_backup

    result = restore_backup(
        args.backup,
        args.output_dir,
        database_name=args.database_name,
    )
    restored_database = Path(result["restored_database"])
    with SQLiteEventRepository(restored_database) as repository:
        repository.record_audit(AuditLogRecord(
            audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(), actor=args.actor,
            action="database_restore", target=str(args.backup), details={"output": str(restored_database)},
        ))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _retention_update(args: argparse.Namespace) -> int:
    with SQLiteEventRepository(args.database) as repository:
        repository.record_audit(AuditLogRecord(
            audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(), actor=args.actor,
            action="retention_rule_update", target="retention_policy", details={"policy": args.policy},
        ))
    return 0


def _evidence_delete(args: argparse.Namespace) -> int:
    root = args.evidence_root.resolve()
    target = (root / args.relative_path).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
        raise ValueError("evidence path is invalid or missing")
    with SQLiteEventRepository(args.database) as repository:
        records = [item for item in repository.list_all_evidence(include_deleted=False)
                   if item.relative_path == target.relative_to(root).as_posix()]
        if len(records) != 1:
            raise ValueError("evidence path is not registered in the database")
        target.unlink()
        repository.mark_evidence_deleted(records[0].evidence_id, reason="manual_deletion")
        repository.record_audit(AuditLogRecord(
            audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(), actor=args.actor,
            action="manual_evidence_deletion", target=records[0].relative_path,
            details={"evidence_id": records[0].evidence_id},
        ))
    return 0


def _evidence_export(args: argparse.Namespace) -> int:
    boxes = []
    for value in args.box:
        parts = tuple(float(item.strip()) for item in value.split(","))
        if len(parts) != 4:
            raise ValueError("--box must be x1,y1,x2,y2")
        boxes.append(parts)
    zones = [tuple(tuple(float(coord) for coord in point) for point in json.loads(value)) for value in args.zone]
    output = export_redacted_image(args.input, args.output, boxes=boxes, zones=zones, blur_kernel=args.blur_kernel)
    print(json.dumps({"output": str(output), "redacted": True}, sort_keys=True))
    return 0


def _generate_pilot_report(args: argparse.Namespace) -> int:
    from .operations import generate_pilot_report

    report = generate_pilot_report(
        args.database, camera_id=args.camera_id, output_path=args.output
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _outbox_retry(args: argparse.Namespace) -> int:
    with SQLiteEventRepository(args.database) as repository:
        retried = repository.retry_outbox(event_id=args.event_id)
    print(json.dumps({"requeued": retried, "event_id": args.event_id}, sort_keys=True))
    return 0


def _production_readiness(args: argparse.Namespace) -> int:
    from .operations import generate_readiness_report

    report = generate_readiness_report(
        service_config_path=args.service_config,
        camera_config_path=args.camera_config,
        database_path=args.database,
        dataset_path=args.dataset,
        backup_path=args.backup,
        downstream_approved=args.downstream_approved,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


def _dataset_validate(args: argparse.Namespace) -> int:
    from .training import DatasetValidationOptions, validate_dataset

    classes = tuple(dict.fromkeys(value.strip() for value in args.classes.split(",") if value.strip()))
    if not classes:
        raise ConfigurationError("--classes must contain at least one class")
    report = validate_dataset(
        args.dataset,
        options=DatasetValidationOptions(required_classes=classes),
    )
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
        approved_by=args.approved_by,
        reviewer_signature=args.reviewer_signature,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


def _model_register(args: argparse.Namespace) -> int:
    from .inference import verify_manifest_approval

    manifest = verify_manifest_approval(args.manifest, approved_reviewers=[args.approved_reviewer])
    with SQLiteEventRepository(args.database) as repository:
        inserted = repository.register_model(ModelVersionRecord(
            model_version=manifest.model_version,
            model_name=f"rfdetr-{manifest.architecture}",
            checkpoint_sha256=manifest.checkpoint_sha256,
            created_at=utc_now_iso(),
            approved_at=utc_now_iso(),
            metadata={
                "manifest": str(args.manifest.resolve()),
                "approved_by": manifest.approved_by,
                "reviewer_signature": manifest.reviewer_signature,
                "dataset_version": manifest.dataset_version,
                "classes": list(manifest.classes),
                "input_resolution": manifest.input_resolution,
            },
        ))
    print(json.dumps({"model_version": manifest.model_version, "registered": inserted}, sort_keys=True))
    return 0


def _model_promote(args: argparse.Namespace) -> int:
    from .inference import verify_manifest_approval

    manifest = verify_manifest_approval(args.manifest, approved_reviewers=[args.approved_reviewer])
    with SQLiteEventRepository(args.database) as repository:
        repository.promote_model(
            camera_id=args.camera_id,
            model_version=manifest.model_version,
            manifest_sha256=manifest.checkpoint_sha256,
            approved_by=manifest.approved_by or args.approved_reviewer,
            changed_at=utc_now_iso(),
        )
        active = repository.active_model(args.camera_id)
    print(json.dumps(active, sort_keys=True))
    return 0


def _model_rollback(args: argparse.Namespace) -> int:
    with SQLiteEventRepository(args.database) as repository:
        active = repository.rollback_model(
            camera_id=args.camera_id,
            approved_by=args.approved_reviewer,
            changed_at=utc_now_iso(),
        )
    print(json.dumps(active, sort_keys=True))
    return 0


def _replay(args: argparse.Namespace) -> int:
    from .training import run_replay

    report = run_replay(
        args.video,
        args.service_config,
        args.camera_config,
        args.checkpoint_manifest,
        output_path=args.output,
        seed=args.seed,
        max_frames=args.max_frames,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handover_replay(args: argparse.Namespace) -> int:
    from .training import run_handover_replay

    report = run_handover_replay(
        args.video,
        args.checkpoint_manifest,
        args.zone_profile,
        args.output_dir,
        analysis_fps=args.analysis_fps,
        minimum_confidence=args.minimum_confidence,
        minimum_transfer_seconds=args.minimum_transfer_seconds,
        minimum_customer_dwell_seconds=args.minimum_customer_dwell_seconds,
        minimum_customer_observations=args.minimum_customer_observations,
        minimum_movement_distance=args.minimum_movement_distance,
        sequence_timeout_seconds=args.sequence_timeout_seconds,
        missing_tolerance_seconds=args.missing_tolerance_seconds,
        duplicate_cooldown_seconds=args.duplicate_cooldown_seconds,
        duplicate_distance=args.duplicate_distance,
        maximum_center_distance_pixels=args.maximum_center_distance_pixels,
        lost_track_seconds=args.lost_track_seconds,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        customer_only_minimum_seconds=args.customer_only_minimum_seconds,
        customer_only_minimum_observations=args.customer_only_minimum_observations,
        customer_only_minimum_movement=args.customer_only_minimum_movement,
        customer_only_static_minimum_seconds=args.customer_only_static_minimum_seconds,
        customer_only_static_minimum_observations=(
            args.customer_only_static_minimum_observations
        ),
        max_frames=args.max_frames,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handover_truth_template(args: argparse.Namespace) -> int:
    from .training import build_truth_template

    if args.output.exists() and not args.force:
        raise ValueError(
            f"refusing to overwrite reviewed ground truth: {args.output}; pass --force to replace it"
        )
    template = build_truth_template(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    candidates = sum(
        len(details["transactions"]) for details in template["sessions"].values()
    )
    print(
        f"Wrote {args.output} with {len(template['sessions'])} session(s) and "
        f"{candidates} candidate row(s). Watch each annotated.mp4, delete rows that were "
        f"not real handovers, fix quantity, and add the ones that were missed."
    )
    return 0


def _handover_evaluate(args: argparse.Namespace) -> int:
    from .training import evaluate_handovers, format_summary

    result = evaluate_handovers(args.report, args.truth, tolerance_seconds=args.tolerance)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else format_summary(result))
    overall = result["overall"]
    unmet = [
        name
        for name, minimum in (
            ("precision", args.require_precision),
            ("recall", args.require_recall),
        )
        if minimum is not None and overall[name] < minimum
    ]
    if unmet:
        print(f"Below the requested threshold for: {', '.join(unmet)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        args = _parser().parse_args(argv)
        return int(args.handler(args))
    except (BackupError, ConfigurationError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
