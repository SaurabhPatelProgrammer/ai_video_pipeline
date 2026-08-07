"""Headless single-camera edge inference service."""

from __future__ import annotations

import cv2
import hashlib
import ipaddress
import json
import logging
import mimetypes
import signal
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..capture import LiveFrameSource, RecordedFrameSource
from ..config import load_camera_config, load_service_config
from ..domain import DepositFSMConfig, ReleaseAwareDepositFSM
from ..inference import (
    RFDETRLocalAdapter,
    SupervisionByteTrackAdapter,
    observations_from_detections,
)
from ..operations import HealthRegistry, HealthState, MetricsRegistry
from ..operations.logging import configure_structured_logging, log_context
from ..security import resolve_credential, safe_source_name
from ..storage import (
    EvidenceRecord,
    EvidenceWriter,
    EventRecord,
    ModelVersionRecord,
    SessionRecord,
    SQLiteEventRepository,
)
from ..storage.database import HealthEventRecord, utc_now_iso
from .quality import FrameQualityGate

LOGGER = logging.getLogger("scoop-ai.service")


class _StatusEndpoint:
    def __init__(
        self,
        host: str,
        port: int,
        health: HealthRegistry,
        metrics: MetricsRegistry,
    ) -> None:
        if host != "localhost":
            try:
                if not ipaddress.ip_address(host).is_loopback:
                    raise ValueError("health endpoint must bind to a loopback address")
            except ValueError as exc:
                raise ValueError("health endpoint must bind to localhost/loopback") from exc
        self.health = health
        self.metrics = metrics
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    snapshot = owner.health.snapshot()
                    body = json.dumps(asdict(snapshot), default=str).encode("utf-8")
                    self.send_response(200 if snapshot.live else 503)
                elif self.path == "/metrics":
                    body = json.dumps(owner.metrics.snapshot(), default=str).encode("utf-8")
                    self.send_response(200)
                else:
                    body = b'{"error":"not_found"}'
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="health-endpoint",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _utc_from_packet(packet) -> str:
    if packet.timestamp_domain.value == "utc_epoch":
        value = datetime.fromtimestamp(packet.timestamp_seconds, timezone.utc)
    else:
        value = packet.observed_at_utc.astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds")


def _expire_evidence(repository: SQLiteEventRepository, evidence_root: Path) -> None:
    root = evidence_root.resolve()
    for record in repository.list_expired_evidence(limit=1000):
        path = (root / record.relative_path).resolve()
        if not path.is_relative_to(root):
            LOGGER.error("Refusing retention path outside evidence root", extra={"path": record.relative_path})
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Evidence retention delete failed", extra={"path": record.relative_path})
            continue
        repository.mark_evidence_deleted(record.evidence_id)


def reconcile_startup(
    repository: SQLiteEventRepository,
    evidence_root: Path,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Run at startup to reconcile filesystem evidence vs. database records.

    Returns a summary dict with keys: 'orphans_recovered', 'orphans_deleted',
    'missing_flagged' listing the relative paths affected.
    """
    log = logger or LOGGER
    root = evidence_root.resolve()
    summary: dict[str, list[str]] = {
        "orphans_recovered": [],
        "orphans_deleted": [],
        "missing_flagged": [],
    }

    if not root.is_dir():
        log.info("Evidence root does not exist yet, skipping reconciliation")
        return summary

    # 1. Collect all evidence files on disk
    disk_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            rel = path.relative_to(root).as_posix()
            disk_files.add(rel)

    # 2. Collect all evidence records from DB
    db_evidence = repository.list_all_evidence()
    db_paths: dict[str, EvidenceRecord] = {rec.relative_path: rec for rec in db_evidence}

    # 3. Collect event evidence paths (events.evidence_path column)
    event_paths = repository.list_all_event_evidence_paths()

    # ── Orphan files: on disk but not tracked by an active evidence record ──
    for rel_path in sorted(disk_files - set(db_paths.keys())):
        abs_path = (root / rel_path).resolve()
        if not abs_path.is_relative_to(root):
            continue  # traversal guard

        if rel_path in event_paths:
            # The event table references this file — auto-recover into evidence_artifacts
            try:
                digest = hashlib.sha256(abs_path.read_bytes()).hexdigest()
                size = abs_path.stat().st_size
                media_type = mimetypes.guess_type(abs_path.name)[0] or "application/octet-stream"
                ev_record = EvidenceRecord(
                    evidence_id=str(uuid.uuid4()),
                    relative_path=rel_path,
                    sha256=digest,
                    size_bytes=size,
                    media_type=media_type,
                    created_at=utc_now_iso(),
                    retention_deadline=(
                        datetime.now(timezone.utc) + timedelta(days=14)
                    ).isoformat(timespec="milliseconds"),
                    integrity_status="valid",
                )
                repository.register_evidence(ev_record)
                log.warning(
                    "Reconciliation: recovered orphan file",
                    extra={"path": rel_path},
                )
                summary["orphans_recovered"].append(rel_path)
            except Exception:
                log.exception(
                    "Reconciliation: failed to recover orphan",
                    extra={"path": rel_path},
                )
        else:
            # Not referenced by any event — safe to delete
            try:
                abs_path.unlink()
                log.info(
                    "Reconciliation: deleted unreferenced orphan",
                    extra={"path": rel_path},
                )
                summary["orphans_deleted"].append(rel_path)
            except OSError:
                log.exception(
                    "Reconciliation: could not delete orphan",
                    extra={"path": rel_path},
                )

    # ── Missing files: in DB (active) but not on disk ──────────────────
    for rec in db_evidence:
        abs_path = (root / rec.relative_path).resolve()
        if rec.relative_path not in disk_files and abs_path.is_relative_to(root):
            # Only flag records that haven't been soft-deleted already
            try:
                repository.update_evidence_integrity(rec.evidence_id, "missing")
            except Exception:
                log.exception(
                    "Reconciliation: could not mark evidence missing",
                    extra={"evidence_id": rec.evidence_id},
                )
                continue
            # Flag the linked event as needs_review
            if rec.event_id:
                try:
                    repository.update_event_review_state(
                        rec.event_id,
                        "needs_review",
                        extra_metadata={"evidence_missing": True},
                    )
                except Exception:
                    log.exception(
                        "Reconciliation: could not flag event",
                        extra={"event_id": rec.event_id},
                    )
            # Record a health event
            try:
                repository.record_health_event(
                    HealthEventRecord(
                        health_event_id=str(uuid.uuid4()),
                        component="reconciliation",
                        state="degraded",
                        occurred_at=utc_now_iso(),
                        message=f"Evidence file missing on disk: {rec.relative_path}",
                    )
                )
            except Exception:
                log.exception("Reconciliation: could not write health event")
            log.warning(
                "Reconciliation: evidence file missing",
                extra={"path": rec.relative_path, "evidence_id": rec.evidence_id},
            )
            summary["missing_flagged"].append(rec.relative_path)

    totals = {k: len(v) for k, v in summary.items()}
    log.info("Startup reconciliation complete", extra=totals)
    return summary


def run_service(
    *,
    service_config_path: str | Path,
    camera_config_path: str | Path,
    checkpoint_manifest_path: str | Path,
    stop_event: threading.Event | None = None,
) -> int:
    """Run until a shutdown signal, recorded EOF, or an unrecoverable failure."""

    service_config = load_service_config(service_config_path)
    camera_config = load_camera_config(camera_config_path)
    configure_structured_logging(
        level=service_config.log_level,
        service_name=service_config.name,
    )
    if not camera_config.enabled:
        raise RuntimeError(f"camera {camera_config.camera_id} is disabled")
    if camera_config.tub_zone is None or camera_config.serving_zone is None:
        raise RuntimeError("production camera config requires tub and serving zones")
    source = camera_config.resolve_source(credential_resolver=resolve_credential)
    artifact_root = service_config.artifact_root.resolve()
    database_path = artifact_root / "database" / "events.sqlite3"
    evidence_root = artifact_root / "evidence"
    artifact_root.mkdir(parents=True, exist_ok=True)

    detector = RFDETRLocalAdapter(Path(checkpoint_manifest_path))
    tracker = SupervisionByteTrackAdapter(
        frame_rate=camera_config.analysis_fps,
        confidence_threshold=detector.manifest.confidence_threshold,
    )
    event_engine = ReleaseAwareDepositFSM(
        DepositFSMConfig(
            minimum_confidence=camera_config.event.minimum_confidence,
            approach_distance=camera_config.event.approach_distance,
            exit_distance=camera_config.event.exit_distance,
            minimum_approach_seconds=camera_config.event.minimum_approach_seconds,
            minimum_release_seconds=camera_config.event.minimum_release_seconds,
            sequence_timeout_seconds=camera_config.event.sequence_timeout_seconds,
            missing_tolerance_seconds=camera_config.event.missing_tolerance_seconds,
            tub_zone=camera_config.tub_zone,
            serving_zone=camera_config.serving_zone,
        )
    )
    quality_gate = FrameQualityGate(
        minimum_blur_variance=camera_config.quality.minimum_blur_variance,
        pixel_change_threshold=camera_config.quality.pixel_change_threshold,
        maximum_changed_fraction=camera_config.quality.maximum_changed_fraction,
        analysis_width=camera_config.quality.analysis_width,
    )
    evidence_writer = EvidenceWriter(evidence_root)
    repository = SQLiteEventRepository(database_path)

    # ── Phase 1: startup reconciliation ──────────────────────────────────
    try:
        reconcile_startup(repository, evidence_root)
    except Exception:
        LOGGER.exception("Startup reconciliation failed (non-fatal)")

    repository.register_model(
        ModelVersionRecord(
            model_version=detector.manifest.model_version,
            model_name=f"rfdetr-{detector.manifest.architecture}",
            checkpoint_sha256=detector.manifest.checkpoint_sha256,
            created_at=utc_now_iso(),
            approved_at=utc_now_iso(),
            metadata={
                "dataset_version": detector.manifest.dataset_version,
                "classes": list(detector.manifest.classes),
                "input_resolution": detector.manifest.input_resolution,
            },
        )
    )
    session_id = str(uuid.uuid4())
    repository.start_session(
        SessionRecord(
            session_id=session_id,
            camera_id=camera_config.camera_id,
            started_at=utc_now_iso(),
            model_version=detector.manifest.model_version,
            source_name=safe_source_name(source),
            metadata={"mode": camera_config.mode},
        )
    )

    health = HealthRegistry(required_components={"capture", "inference", "storage"})
    metrics = MetricsRegistry()
    health.report("capture", HealthState.STARTING, "opening source")
    health.report("inference", HealthState.STARTING, "model not warmed")
    health.report("storage", HealthState.HEALTHY, "SQLite WAL ready")
    try:
        status_endpoint = _StatusEndpoint(
            service_config.health_host,
            service_config.health_port,
            health,
            metrics,
        )
        status_endpoint.start()
    except Exception:
        repository.finish_session(session_id, status="failed")
        repository.close()
        raise

    stopping = stop_event or threading.Event()
    if stop_event is None and threading.current_thread() is threading.main_thread():
        def stop_handler(_signum: int, _frame: object) -> None:
            stopping.set()

        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)

    sequence = -1
    next_media_sample = 0.0
    last_live_inference = 0.0
    last_retention = 0.0
    failed = False
    reader = None
    try:
        if camera_config.mode == "recorded":
            if not isinstance(source, str):
                raise RuntimeError("recorded camera source must be a file path")
            reader = RecordedFrameSource(source, source_id=camera_config.camera_id).start()
        else:
            reader = LiveFrameSource(
                source,
                source_id=camera_config.camera_id,
                reconnect_seconds=camera_config.capture.reconnect_seconds,
                reconnect_max_seconds=camera_config.capture.reconnect_max_seconds,
                reconnect_jitter_ratio=camera_config.capture.reconnect_jitter_ratio,
                rtsp_transport=camera_config.capture.rtsp_transport,
                open_timeout_ms=camera_config.capture.open_timeout_ms,
                read_timeout_ms=camera_config.capture.read_timeout_ms,
            ).start()
        with log_context(camera_id=camera_config.camera_id, session_id=session_id):
            while not stopping.is_set():
                if camera_config.mode == "recorded":
                    packet = reader.read()
                    if packet is None:
                        break
                    if packet.timestamp_seconds + 1e-9 < next_media_sample:
                        continue
                    next_media_sample = packet.timestamp_seconds + 1.0 / camera_config.analysis_fps
                else:
                    packet = reader.read(
                        sequence,
                        timeout=camera_config.capture.read_wait_seconds,
                    )
                    if packet is None:
                        frame_age = reader.frame_age_seconds
                        metrics.gauge("last_frame_age_seconds", frame_age or 0.0)
                        health.report("capture", HealthState.DEGRADED, reader.health.detail)
                        continue
                    now = time.monotonic()
                    if now - last_live_inference < 1.0 / camera_config.analysis_fps:
                        sequence = packet.sequence
                        continue
                    last_live_inference = now
                sequence = packet.sequence
                metrics.increment("frames_received_total")
                health.report("capture", HealthState.HEALTHY, "receiving frames")

                assessment = quality_gate.assess(packet.frame)
                metrics.gauge("frame_blur_variance", assessment.blur_variance)
                metrics.gauge("frame_changed_fraction", assessment.changed_fraction)
                if not assessment.acceptable:
                    metrics.increment("frames_quality_rejected_total")
                    health.report(
                        "capture",
                        HealthState.DEGRADED,
                        ",".join(assessment.reasons),
                    )
                    continue

                started = time.perf_counter()
                rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
                detections = detector.predict(rgb, packet.timestamp_seconds)
                tracked = tracker.update(detections, packet.timestamp_seconds)
                observations = observations_from_detections(
                    tracked,
                    timestamp=packet.timestamp_seconds,
                    frame_width=packet.frame.shape[1],
                    frame_height=packet.frame.shape[0],
                )
                events = event_engine.update(packet.timestamp_seconds, observations)
                inference_seconds = time.perf_counter() - started
                metrics.observe("inference_latency_seconds", inference_seconds)
                metrics.increment("frames_processed_total")
                health.report("inference", HealthState.HEALTHY, "inference completed")

                for event in events:
                    event_id = str(uuid.uuid4())
                    ok, encoded = cv2.imencode(".jpg", packet.frame)
                    if not ok:
                        raise RuntimeError("could not encode event evidence")
                    artifact = evidence_writer.write_bytes(
                        session_id,
                        event_id,
                        encoded.tobytes(),
                        extension=".jpg",
                    )
                    repository.insert_event(
                        EventRecord(
                            event_id=event_id,
                            session_id=session_id,
                            camera_id=camera_config.camera_id,
                            event_type="scoop_deposited_candidate",
                            occurred_at=_utc_from_packet(packet),
                            confidence=event.confidence,
                            model_version=detector.manifest.model_version,
                            container_track_id=event.container_track_id,
                            scoop_track_id=event.scoop_track_id,
                            evidence_path=artifact.relative_path,
                            evidence_sha256=artifact.sha256,
                            review_state="unreviewed",
                            metadata={
                                "timestamp_domain": packet.timestamp_domain.value,
                                "media_timestamp_seconds": packet.media_timestamp_seconds,
                                "loaded_at": event.loaded_at,
                                "approached_at": event.approached_at,
                                "released_at": event.released_at,
                                "confirmed_at": event.confirmed_at,
                            },
                        )
                    )
                    deadline = datetime.now(timezone.utc) + timedelta(days=14)
                    repository.register_evidence(
                        EvidenceRecord(
                            evidence_id=event_id,
                            event_id=event_id,
                            relative_path=artifact.relative_path,
                            sha256=artifact.sha256,
                            size_bytes=artifact.size_bytes,
                            media_type=artifact.media_type,
                            created_at=artifact.created_at,
                            retention_deadline=deadline.isoformat(timespec="milliseconds"),
                        )
                    )
                    metrics.increment("scoop_candidates_total")

                now = time.monotonic()
                if now - last_retention >= 3600:
                    _expire_evidence(repository, evidence_root)
                    last_retention = now
    except Exception:
        failed = True
        health.report("inference", HealthState.UNHEALTHY, "service failure")
        LOGGER.exception("Scoop AI service failed")
    finally:
        stopping.set()
        health.mark_stopping()
        if reader is not None:
            reader.stop()
        status_endpoint.stop()
        repository.finish_session(
            session_id,
            status="failed" if failed else "completed",
        )
        repository.close()
    return 1 if failed else 0
