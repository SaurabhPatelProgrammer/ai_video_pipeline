"""SQLite WAL event repository with small, explicit schema migrations."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


def utc_now_iso() -> str:
    """Return a stable, timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_timestamp(value: str, field_name: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _validate_timestamp(value: str, field_name: str) -> None:
    _normalize_timestamp(value, field_name)


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    camera_id: str
    started_at: str
    status: str = "running"
    model_version: str | None = None
    source_name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.session_id.strip() or not self.camera_id.strip():
            raise ValueError("session_id and camera_id are required")
        _validate_timestamp(self.started_at, "started_at")
        if self.status not in {"running", "completed", "failed", "abandoned"}:
            raise ValueError(f"unsupported session status: {self.status}")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    session_id: str
    camera_id: str
    event_type: str
    occurred_at: str
    confidence: float | None = None
    model_version: str | None = None
    container_track_id: int | None = None
    scoop_track_id: int | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    review_state: str = "unreviewed"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        required = (self.event_id, self.session_id, self.camera_id, self.event_type)
        if any(not value.strip() for value in required):
            raise ValueError("event_id, session_id, camera_id and event_type are required")
        _validate_timestamp(self.occurred_at, "occurred_at")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and between 0 and 1")
        if self.evidence_sha256 is not None:
            _validate_digest(self.evidence_sha256, "evidence_sha256")
        if self.review_state not in {"unreviewed", "accepted", "rejected", "needs_review"}:
            raise ValueError(f"unsupported review_state: {self.review_state}")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class ModelVersionRecord:
    model_version: str
    model_name: str
    checkpoint_sha256: str
    created_at: str
    approved_at: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model_version.strip() or not self.model_name.strip():
            raise ValueError("model_version and model_name are required")
        _validate_digest(self.checkpoint_sha256, "checkpoint_sha256")
        _validate_timestamp(self.created_at, "created_at")
        if self.approved_at is not None:
            _validate_timestamp(self.approved_at, "approved_at")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: str
    retention_deadline: str
    event_id: str | None = None
    integrity_status: str = "unverified"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.evidence_id.strip() or not self.relative_path.strip():
            raise ValueError("evidence_id and relative_path are required")
        if Path(self.relative_path).is_absolute() or ".." in Path(self.relative_path).parts:
            raise ValueError("relative_path must stay under the evidence root")
        _validate_digest(self.sha256, "sha256")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        _validate_timestamp(self.created_at, "created_at")
        _validate_timestamp(self.retention_deadline, "retention_deadline")
        if self.integrity_status not in {"unverified", "valid", "corrupt", "missing"}:
            raise ValueError(f"unsupported integrity_status: {self.integrity_status}")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    event_id: str
    decision: str
    reviewer_id: str
    reviewed_at: str
    notes: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.review_id.strip() or not self.event_id.strip() or not self.reviewer_id.strip():
            raise ValueError("review_id, event_id and reviewer_id are required")
        if self.decision not in {"accepted", "rejected", "needs_review"}:
            raise ValueError("unsupported review decision")
        _validate_timestamp(self.reviewed_at, "reviewed_at")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class GroundTruthRecord:
    ground_truth_id: str
    session_id: str
    camera_id: str
    occurred_at: str
    is_completed_scoop: bool
    reviewer_id: str
    container_track_id: int | None = None
    evidence_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        required = (self.ground_truth_id, self.session_id, self.camera_id, self.reviewer_id)
        if any(not value.strip() for value in required):
            raise ValueError("ground_truth_id, session_id, camera_id and reviewer_id are required")
        _validate_timestamp(self.occurred_at, "occurred_at")
        _canonical_json(self.metadata)


@dataclass(frozen=True)
class HealthEventRecord:
    health_event_id: str
    component: str
    state: str
    occurred_at: str
    camera_id: str | None = None
    message: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.health_event_id.strip() or not self.component.strip():
            raise ValueError("health_event_id and component are required")
        if self.state not in {"starting", "healthy", "degraded", "unhealthy", "stopping"}:
            raise ValueError("unsupported health state")
        _validate_timestamp(self.occurred_at, "occurred_at")
        _canonical_json(self.details)


@dataclass(frozen=True)
class AuditLogRecord:
    audit_id: str
    occurred_at: str
    actor: str
    action: str
    target: str
    details: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if any(not value.strip() for value in (self.audit_id, self.actor, self.action, self.target)):
            raise ValueError("audit_id, actor, action and target are required")
        _validate_timestamp(self.occurred_at, "occurred_at")
        _canonical_json(self.details)


@dataclass(frozen=True)
class TelemetryRecord:
    telemetry_id: str
    camera_id: str
    observed_at: str
    fps: float
    blur_variance: float
    changed_fraction: float
    accepted: bool

    def validate(self) -> None:
        if not self.telemetry_id.strip() or not self.camera_id.strip():
            raise ValueError("telemetry_id and camera_id are required")
        _validate_timestamp(self.observed_at, "observed_at")
        if not all(math.isfinite(value) and value >= 0 for value in (
            self.fps, self.blur_variance, self.changed_fraction,
        )):
            raise ValueError("telemetry values must be finite and non-negative")
        if self.changed_fraction > 1:
            raise ValueError("changed_fraction cannot exceed 1")


@dataclass(frozen=True)
class OutboxRecord:
    event_id: str
    payload: Mapping[str, object]
    state: str = "pending"
    attempts: int = 0
    next_attempt_at: str | None = None
    last_error: str | None = None
    signature: str | None = None
    updated_at: str = ""

    def validate(self) -> None:
        if not self.event_id.strip():
            raise ValueError("outbox event_id is required")
        if self.state not in {"pending", "exported", "acknowledged", "failed", "dead_letter"}:
            raise ValueError("unsupported outbox state")
        if self.attempts < 0:
            raise ValueError("outbox attempts cannot be negative")
        _canonical_json(self.payload)


class EventConflictError(RuntimeError):
    """Raised when an event ID is reused with a different payload."""


def _validate_digest(value: str, field_name: str) -> None:
    digest = value.lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field_name} must be a 64-character hex digest")


MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK(status IN ('running','completed','failed','abandoned')),
            model_version TEXT,
            source_name TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE model_versions (
            model_version TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            approved_at TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            camera_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            confidence REAL,
            model_version TEXT,
            container_track_id INTEGER,
            scoop_track_id INTEGER,
            evidence_path TEXT,
            evidence_sha256 TEXT,
            review_state TEXT NOT NULL
                CHECK(review_state IN ('unreviewed','accepted','rejected','needs_review')),
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE evidence_artifacts (
            evidence_id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(event_id),
            relative_path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            media_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retention_deadline TEXT NOT NULL,
            deleted_at TEXT,
            deletion_reason TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE event_reviews (
            review_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES events(event_id),
            decision TEXT NOT NULL CHECK(decision IN ('accepted','rejected','needs_review')),
            reviewer_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            notes TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE ground_truth_events (
            ground_truth_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            camera_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            is_completed_scoop INTEGER NOT NULL CHECK(is_completed_scoop IN (0,1)),
            reviewer_id TEXT NOT NULL,
            container_track_id INTEGER,
            evidence_id TEXT REFERENCES evidence_artifacts(evidence_id),
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE health_events (
            health_event_id TEXT PRIMARY KEY,
            camera_id TEXT,
            component TEXT NOT NULL,
            state TEXT NOT NULL
                CHECK(state IN ('starting','healthy','degraded','unhealthy','stopping')),
            occurred_at TEXT NOT NULL,
            message TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX events_session_time_idx ON events(session_id, occurred_at);
        CREATE INDEX events_camera_time_idx ON events(camera_id, occurred_at);
        CREATE INDEX events_review_state_idx ON events(review_state, occurred_at);
        CREATE INDEX evidence_retention_idx ON evidence_artifacts(deleted_at, retention_deadline);
        CREATE INDEX reviews_event_time_idx ON event_reviews(event_id, reviewed_at);
        CREATE INDEX ground_truth_session_time_idx ON ground_truth_events(session_id, occurred_at);
        CREATE INDEX health_component_time_idx ON health_events(component, occurred_at);

        CREATE TRIGGER event_reviews_no_update
        BEFORE UPDATE ON event_reviews BEGIN
            SELECT RAISE(ABORT, 'event reviews are immutable');
        END;
        CREATE TRIGGER event_reviews_no_delete
        BEFORE DELETE ON event_reviews BEGIN
            SELECT RAISE(ABORT, 'event reviews are immutable');
        END;
        CREATE TRIGGER ground_truth_no_update
        BEFORE UPDATE ON ground_truth_events BEGIN
            SELECT RAISE(ABORT, 'ground truth events are immutable');
        END;
        CREATE TRIGGER ground_truth_no_delete
        BEFORE DELETE ON ground_truth_events BEGIN
            SELECT RAISE(ABORT, 'ground truth events are immutable');
        END;
        CREATE TRIGGER model_versions_no_update
        BEFORE UPDATE ON model_versions BEGIN
            SELECT RAISE(ABORT, 'model versions are immutable');
        END;
        CREATE TRIGGER model_versions_no_delete
        BEFORE DELETE ON model_versions BEGIN
            SELECT RAISE(ABORT, 'model versions are immutable');
        END;
        CREATE TRIGGER health_events_no_update
        BEFORE UPDATE ON health_events BEGIN
            SELECT RAISE(ABORT, 'health events are immutable');
        END;
        CREATE TRIGGER health_events_no_delete
        BEFORE DELETE ON health_events BEGIN
            SELECT RAISE(ABORT, 'health events are immutable');
        END;
        """,
    ),
    (
        2,
        """
        ALTER TABLE evidence_artifacts ADD COLUMN integrity_status TEXT
            CHECK(integrity_status IN ('unverified', 'valid', 'corrupt', 'missing'))
            DEFAULT 'unverified';
        """,
    ),
)


class SQLiteEventRepository:
    """Thread-safe repository for a single edge process.

    Each write is committed in a transaction. WAL allows health/reporting readers
    to inspect the database without blocking event ingestion.
    """

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._last_write_lock_seconds = 0.0
        self._connection = sqlite3.connect(
            self.path,
            timeout=max(0, busy_timeout_ms) / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(f"PRAGMA busy_timeout={int(max(0, busy_timeout_ms))}")
        self._connection.execute("PRAGMA synchronous=FULL")
        mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            self._connection.close()
            raise RuntimeError(f"SQLite WAL mode unavailable for {self.path}")
        self._migrate()

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                int(row[0])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            known = {version for version, _ in MIGRATIONS}
            unknown = applied - known
            if unknown:
                raise RuntimeError(f"database has unsupported migrations: {sorted(unknown)}")
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                statement = ""
                for line in sql.splitlines():
                    statement += line + "\n"
                    if sqlite3.complete_statement(statement):
                        connection.execute(statement)
                        statement = ""
                if statement.strip():
                    raise RuntimeError(f"migration {version} contains incomplete SQL")
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now_iso()),
                )
            # Governance tables are additive and intentionally kept outside the
            # public schema version so existing v2 installations upgrade safely.
            connection.execute("""CREATE TABLE IF NOT EXISTS active_models (
                    camera_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
                    manifest_sha256 TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    activated_at TEXT NOT NULL
                )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS model_activation_history (
                    activation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
                    action TEXT NOT NULL CHECK(action IN ('promote','rollback')),
                    approved_by TEXT NOT NULL,
                    changed_at TEXT NOT NULL
                )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS model_activation_camera_idx
                    ON model_activation_history(camera_id, activation_id);
                """)
            connection.execute("""CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS audit_logs_time_idx
                    ON audit_logs(occurred_at, audit_id)
                """)
            connection.execute("""CREATE TRIGGER IF NOT EXISTS audit_logs_no_update
                    BEFORE UPDATE ON audit_logs BEGIN
                        SELECT RAISE(ABORT, 'audit logs are immutable');
                    END
                """)
            connection.execute("""CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete
                    BEFORE DELETE ON audit_logs BEGIN
                        SELECT RAISE(ABORT, 'audit logs are immutable');
                    END
                """)
            connection.execute("""CREATE TABLE IF NOT EXISTS pilot_telemetry (
                    telemetry_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    fps REAL NOT NULL,
                    blur_variance REAL NOT NULL,
                    changed_fraction REAL NOT NULL,
                    accepted INTEGER NOT NULL CHECK(accepted IN (0,1))
                )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS pilot_telemetry_time_idx
                    ON pilot_telemetry(camera_id, observed_at)
                """)
            connection.execute("""CREATE TABLE IF NOT EXISTS event_outbox (
                    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','exported','acknowledged','failed','dead_letter')),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                    next_attempt_at TEXT,
                    last_error TEXT,
                    signature TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""")
            connection.execute("""CREATE INDEX IF NOT EXISTS event_outbox_poll_idx
                    ON event_outbox(state, next_attempt_at, updated_at)
                """)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        started = time.perf_counter()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            self._last_write_lock_seconds = time.perf_counter() - started
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @property
    def last_write_lock_seconds(self) -> float:
        with self._lock:
            return self._last_write_lock_seconds

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
        return int(row[0])

    def journal_mode(self) -> str:
        with self._lock:
            return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0])

    def start_session(self, session: SessionRecord) -> bool:
        session.validate()
        metadata_json = _canonical_json(session.metadata)
        started_at = _normalize_timestamp(session.started_at, "started_at")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions(
                    session_id, camera_id, started_at, status, model_version,
                    source_name, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (
                    session.session_id,
                    session.camera_id,
                    started_at,
                    session.status,
                    session.model_version,
                    session.source_name,
                    metadata_json,
                    utc_now_iso(),
                ),
            )
            if cursor.rowcount:
                return True
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session.session_id,)
            ).fetchone()
            expected = {
                "session_id": session.session_id,
                "camera_id": session.camera_id,
                "started_at": started_at,
                "status": session.status,
                "model_version": session.model_version,
                "source_name": session.source_name,
                "metadata_json": metadata_json,
            }
            if row is None or any(row[key] != value for key, value in expected.items()):
                raise EventConflictError(f"session_id {session.session_id!r} has another payload")
            return False

    def finish_session(
        self,
        session_id: str,
        *,
        status: str,
        finished_at: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "abandoned"}:
            raise ValueError("finished status must be completed, failed or abandoned")
        finished_at = _normalize_timestamp(finished_at or utc_now_iso(), "finished_at")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET status = ?, finished_at = ? WHERE session_id = ?",
                (status, finished_at, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown session_id: {session_id}")

    def insert_event(self, event: EventRecord) -> bool:
        """Insert once; return False for an identical retry and reject conflicts."""
        event.validate()
        metadata_json = _canonical_json(event.metadata)
        occurred_at = _normalize_timestamp(event.occurred_at, "occurred_at")
        values = (
            event.event_id,
            event.session_id,
            event.camera_id,
            event.event_type,
            occurred_at,
            event.confidence,
            event.model_version,
            event.container_track_id,
            event.scoop_track_id,
            event.evidence_path,
            event.evidence_sha256.lower() if event.evidence_sha256 else None,
            event.review_state,
            metadata_json,
            utc_now_iso(),
        )
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events(
                    event_id, session_id, camera_id, event_type, occurred_at,
                    confidence, model_version, container_track_id, scoop_track_id,
                    evidence_path, evidence_sha256, review_state, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                values,
            )
            if cursor.rowcount:
                return True
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            comparable_keys = (
                "event_id",
                "session_id",
                "camera_id",
                "event_type",
                "occurred_at",
                "confidence",
                "model_version",
                "container_track_id",
                "scoop_track_id",
                "evidence_path",
                "evidence_sha256",
                "review_state",
                "metadata_json",
            )
            expected = dict(zip(comparable_keys, values[:-1]))
            if row is None or any(row[key] != expected[key] for key in comparable_keys):
                raise EventConflictError(f"event_id {event.event_id!r} has another payload")
            return False

    def get_event(self, event_id: str) -> EventRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._event_from_row(row) if row else None

    def list_events(
        self,
        *,
        session_id: str | None = None,
        camera_id: str | None = None,
        review_state: str | None = None,
        event_type: str | None = None,
        occurred_after: str | None = None,
        occurred_before: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if review_state is not None and review_state not in {
            "unreviewed",
            "accepted",
            "rejected",
            "needs_review",
        }:
            raise ValueError("unsupported review_state filter")
        if occurred_after is not None:
            occurred_after = _normalize_timestamp(occurred_after, "occurred_after")
        if occurred_before is not None:
            occurred_before = _normalize_timestamp(occurred_before, "occurred_before")
        query = "SELECT * FROM events"
        clauses: list[str] = []
        parameters_list: list[object] = []
        for column, value in (
            ("session_id", session_id),
            ("camera_id", camera_id),
            ("review_state", review_state),
            ("event_type", event_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters_list.append(value)
        if occurred_after is not None:
            clauses.append("occurred_at >= ?")
            parameters_list.append(occurred_after)
        if occurred_before is not None:
            clauses.append("occurred_at < ?")
            parameters_list.append(occurred_before)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY occurred_at, event_id LIMIT ?"
        parameters_list.append(limit)
        with self._lock:
            rows = self._connection.execute(query, tuple(parameters_list)).fetchall()
        return [self._event_from_row(row) for row in rows]

    def register_model(self, model: ModelVersionRecord) -> bool:
        model.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_versions(
                    model_version, model_name, checkpoint_sha256, created_at,
                    approved_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO NOTHING
                """,
                (
                    model.model_version,
                    model.model_name,
                    model.checkpoint_sha256.lower(),
                    _normalize_timestamp(model.created_at, "created_at"),
                    _normalize_timestamp(model.approved_at, "approved_at")
                    if model.approved_at
                    else None,
                    _canonical_json(model.metadata),
                ),
            )
            return bool(cursor.rowcount)

    def promote_model(
        self,
        *,
        camera_id: str,
        model_version: str,
        manifest_sha256: str,
        approved_by: str,
        changed_at: str,
        action: str = "promote",
    ) -> None:
        if action not in {"promote", "rollback"}:
            raise ValueError("unsupported model activation action")
        if not camera_id.strip() or not model_version.strip() or not approved_by.strip():
            raise ValueError("camera_id, model_version and approved_by are required")
        _validate_digest(manifest_sha256, "manifest_sha256")
        changed_at = _normalize_timestamp(changed_at, "changed_at")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT approved_at FROM model_versions WHERE model_version = ?",
                (model_version,),
            ).fetchone()
            if row is None or row[0] is None:
                raise ValueError("only registered approved models can be activated")
            connection.execute(
                """INSERT INTO active_models(camera_id, model_version, manifest_sha256, approved_by, activated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(camera_id) DO UPDATE SET model_version=excluded.model_version,
                   manifest_sha256=excluded.manifest_sha256, approved_by=excluded.approved_by,
                   activated_at=excluded.activated_at""",
                (camera_id, model_version, manifest_sha256.lower(), approved_by, changed_at),
            )
            connection.execute(
                """INSERT INTO model_activation_history(camera_id, model_version, action, approved_by, changed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (camera_id, model_version, action, approved_by, changed_at),
            )

    def active_model(self, camera_id: str) -> dict[str, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT camera_id, model_version, manifest_sha256, approved_by, activated_at "
                "FROM active_models WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        return dict(row) if row else None

    def rollback_model(self, *, camera_id: str, approved_by: str, changed_at: str) -> dict[str, str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT model_version FROM model_activation_history WHERE camera_id = ? "
                "ORDER BY activation_id DESC", (camera_id,)
            ).fetchall()
        active = self.active_model(camera_id)
        active_version = active["model_version"] if active else None
        target = next((str(row[0]) for row in rows if str(row[0]) != active_version), None)
        if target is None:
            raise ValueError(f"no prior approved model exists for camera {camera_id!r}")
        with self._lock:
            model = self._connection.execute(
                "SELECT checkpoint_sha256 FROM model_versions WHERE model_version = ?", (target,)
            ).fetchone()
        assert model is not None
        self.promote_model(
            camera_id=camera_id, model_version=target, manifest_sha256=str(model[0]),
            approved_by=approved_by, changed_at=changed_at, action="rollback",
        )
        result = self.active_model(camera_id)
        assert result is not None
        return result

    def register_evidence(self, evidence: EvidenceRecord) -> bool:
        evidence.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evidence_artifacts(
                    evidence_id, event_id, relative_path, sha256, size_bytes,
                    media_type, created_at, retention_deadline, integrity_status,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO NOTHING
                """,
                (
                    evidence.evidence_id,
                    evidence.event_id,
                    evidence.relative_path,
                    evidence.sha256.lower(),
                    evidence.size_bytes,
                    evidence.media_type,
                    _normalize_timestamp(evidence.created_at, "created_at"),
                    _normalize_timestamp(evidence.retention_deadline, "retention_deadline"),
                    evidence.integrity_status,
                    _canonical_json(evidence.metadata),
                ),
            )
            return bool(cursor.rowcount)

    def list_expired_evidence(
        self,
        *,
        deadline: str | None = None,
        limit: int = 1000,
    ) -> list[EvidenceRecord]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        deadline = _normalize_timestamp(deadline or utc_now_iso(), "deadline")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM evidence_artifacts
                WHERE deleted_at IS NULL AND retention_deadline <= ?
                ORDER BY retention_deadline, evidence_id LIMIT ?
                """,
                (deadline, limit),
            ).fetchall()
        return [self._evidence_from_row(row) for row in rows]

    def mark_evidence_deleted(
        self,
        evidence_id: str,
        *,
        deleted_at: str | None = None,
        reason: str = "retention_policy",
    ) -> bool:
        deleted_at = _normalize_timestamp(deleted_at or utc_now_iso(), "deleted_at")
        if not reason.strip():
            raise ValueError("deletion reason is required")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_artifacts
                SET deleted_at = ?, deletion_reason = ?
                WHERE evidence_id = ? AND deleted_at IS NULL
                """,
                (deleted_at, reason, evidence_id),
            )
            return bool(cursor.rowcount)

    def add_review(self, review: ReviewRecord) -> bool:
        """Append an immutable decision and update the event's current projection."""
        review.validate()
        reviewed_at = _normalize_timestamp(review.reviewed_at, "reviewed_at")
        metadata_json = _canonical_json(review.metadata)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO event_reviews(
                    review_id, event_id, decision, reviewer_id, reviewed_at,
                    notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO NOTHING
                """,
                (
                    review.review_id,
                    review.event_id,
                    review.decision,
                    review.reviewer_id,
                    reviewed_at,
                    review.notes,
                    metadata_json,
                ),
            )
            if not cursor.rowcount:
                row = connection.execute(
                    "SELECT * FROM event_reviews WHERE review_id = ?", (review.review_id,)
                ).fetchone()
                expected = {
                    "event_id": review.event_id,
                    "decision": review.decision,
                    "reviewer_id": review.reviewer_id,
                    "reviewed_at": reviewed_at,
                    "notes": review.notes,
                    "metadata_json": metadata_json,
                }
                if row is None or any(row[key] != value for key, value in expected.items()):
                    raise EventConflictError(
                        f"review_id {review.review_id!r} has another payload"
                    )
                return False
            latest = connection.execute(
                """
                SELECT decision FROM event_reviews
                WHERE event_id = ? ORDER BY reviewed_at DESC, review_id DESC LIMIT 1
                """,
                (review.event_id,),
            ).fetchone()
            event_cursor = connection.execute(
                "UPDATE events SET review_state = ? WHERE event_id = ?",
                (latest["decision"], review.event_id),
            )
            if event_cursor.rowcount != 1:
                raise KeyError(f"unknown event_id: {review.event_id}")
            if latest["decision"] == "accepted":
                event = connection.execute(
                    "SELECT * FROM events WHERE event_id = ?", (review.event_id,)
                ).fetchone()
                assert event is not None
                payload = {key: event[key] for key in (
                    "event_id", "session_id", "camera_id", "event_type", "occurred_at",
                    "confidence", "model_version", "container_track_id", "scoop_track_id",
                    "evidence_path", "evidence_sha256", "metadata_json",
                )}
                now = utc_now_iso()
                connection.execute(
                    """INSERT INTO event_outbox(event_id, payload_json, state, attempts, created_at, updated_at)
                       VALUES (?, ?, 'pending', 0, ?, ?)
                       ON CONFLICT(event_id) DO UPDATE SET state='pending', updated_at=excluded.updated_at
                       WHERE event_outbox.state IN ('failed','dead_letter')""",
                    (review.event_id, _canonical_json(payload), now, now),
                )
            return True

    def list_outbox(self, *, states: set[str] | None = None, limit: int = 1000) -> list[OutboxRecord]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        allowed = {"pending", "exported", "acknowledged", "failed", "dead_letter"}
        selected = states or allowed
        if not selected or not selected.issubset(allowed):
            raise ValueError("unsupported outbox state filter")
        placeholders = ",".join("?" for _ in selected)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM event_outbox WHERE state IN ({placeholders}) ORDER BY updated_at, event_id LIMIT ?",
                (*sorted(selected), limit),
            ).fetchall()
        return [self._outbox_from_row(row) for row in rows]

    def claim_outbox(self, *, limit: int = 20, now: str | None = None) -> list[OutboxRecord]:
        now = _normalize_timestamp(now or utc_now_iso(), "now")
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT o.* FROM event_outbox o JOIN events e ON e.event_id=o.event_id
                   WHERE o.state IN ('pending','failed') AND e.review_state='accepted'
                   AND (o.next_attempt_at IS NULL OR o.next_attempt_at <= ?)
                   ORDER BY o.updated_at, o.event_id LIMIT ?""", (now, limit)
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE event_outbox SET state='exported', updated_at=? WHERE event_id=?",
                    (now, row["event_id"]),
                )
        return [self._outbox_from_row(row, state_override="exported") for row in rows]

    def mark_outbox_acknowledged(self, event_id: str, *, signature: str, updated_at: str | None = None) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE event_outbox SET state='acknowledged', signature=?, last_error=NULL, updated_at=? WHERE event_id=?",
                (signature, _normalize_timestamp(updated_at or utc_now_iso(), "updated_at"), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown outbox event {event_id!r}")

    def mark_outbox_failure(self, event_id: str, *, error: str, next_attempt_at: str | None, max_attempts: int) -> str:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self.transaction() as connection:
            row = connection.execute("SELECT attempts FROM event_outbox WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown outbox event {event_id!r}")
            attempts = int(row[0]) + 1
            state = "dead_letter" if attempts >= max_attempts else "failed"
            connection.execute(
                "UPDATE event_outbox SET state=?, attempts=?, last_error=?, next_attempt_at=?, updated_at=? WHERE event_id=?",
                (state, attempts, error[:1000], next_attempt_at, utc_now_iso(), event_id),
            )
        return state

    def retry_outbox(self, *, event_id: str | None = None, reset_attempts: bool = True) -> int:
        with self.transaction() as connection:
            clauses = ["state IN ('failed','dead_letter')"]
            params: list[object] = []
            if event_id:
                clauses.append("event_id=?")
                params.append(event_id)
            attempts = "0" if reset_attempts else "attempts"
            cursor = connection.execute(
                f"UPDATE event_outbox SET state='pending', attempts={attempts}, next_attempt_at=NULL, last_error=NULL, updated_at=? WHERE {' AND '.join(clauses)}",
                (utc_now_iso(), *params),
            )
            return int(cursor.rowcount)

    @staticmethod
    def _outbox_from_row(row: sqlite3.Row, *, state_override: str | None = None) -> OutboxRecord:
        return OutboxRecord(
            event_id=row["event_id"], payload=json.loads(row["payload_json"]),
            state=state_override or row["state"], attempts=int(row["attempts"]),
            next_attempt_at=row["next_attempt_at"], last_error=row["last_error"],
            signature=row["signature"], updated_at=row["updated_at"],
        )

    def list_reviews(self, event_id: str) -> list[ReviewRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM event_reviews WHERE event_id = ? ORDER BY reviewed_at, review_id",
                (event_id,),
            ).fetchall()
        return [
            ReviewRecord(
                review_id=row["review_id"],
                event_id=row["event_id"],
                decision=row["decision"],
                reviewer_id=row["reviewer_id"],
                reviewed_at=row["reviewed_at"],
                notes=row["notes"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def add_ground_truth(self, record: GroundTruthRecord) -> bool:
        record.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO ground_truth_events(
                    ground_truth_id, session_id, camera_id, occurred_at,
                    is_completed_scoop, reviewer_id, container_track_id,
                    evidence_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ground_truth_id) DO NOTHING
                """,
                (
                    record.ground_truth_id,
                    record.session_id,
                    record.camera_id,
                    _normalize_timestamp(record.occurred_at, "occurred_at"),
                    int(record.is_completed_scoop),
                    record.reviewer_id,
                    record.container_track_id,
                    record.evidence_id,
                    _canonical_json(record.metadata),
                    utc_now_iso(),
                ),
            )
            return bool(cursor.rowcount)

    def record_health_event(self, record: HealthEventRecord) -> bool:
        record.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO health_events(
                    health_event_id, camera_id, component, state, occurred_at,
                    message, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(health_event_id) DO NOTHING
                """,
                (
                    record.health_event_id,
                    record.camera_id,
                    record.component,
                    record.state,
                    _normalize_timestamp(record.occurred_at, "occurred_at"),
                    record.message,
                    _canonical_json(record.details),
                    utc_now_iso(),
                ),
            )
            return bool(cursor.rowcount)

    def record_audit(self, record: AuditLogRecord) -> bool:
        record.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO audit_logs(audit_id, occurred_at, actor, action, target, details_json)
                   VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(audit_id) DO NOTHING""",
                (
                    record.audit_id,
                    _normalize_timestamp(record.occurred_at, "occurred_at"),
                    record.actor,
                    record.action,
                    record.target,
                    _canonical_json(record.details),
                ),
            )
            return bool(cursor.rowcount)

    def list_audit_logs(self, *, limit: int = 1000) -> list[AuditLogRecord]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM audit_logs ORDER BY occurred_at, audit_id LIMIT ?", (limit,)
            ).fetchall()
        return [AuditLogRecord(
            audit_id=row["audit_id"], occurred_at=row["occurred_at"], actor=row["actor"],
            action=row["action"], target=row["target"], details=json.loads(row["details_json"]),
        ) for row in rows]

    def record_telemetry(self, record: TelemetryRecord) -> bool:
        record.validate()
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO pilot_telemetry(
                    telemetry_id, camera_id, observed_at, fps, blur_variance,
                    changed_fraction, accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(telemetry_id) DO NOTHING""",
                (
                    record.telemetry_id, record.camera_id,
                    _normalize_timestamp(record.observed_at, "observed_at"),
                    record.fps, record.blur_variance, record.changed_fraction,
                    int(record.accepted),
                ),
            )
            return bool(cursor.rowcount)

    def pilot_report(self, *, camera_id: str | None = None) -> dict[str, object]:
        clauses = ["1=1"]
        params: list[object] = []
        if camera_id:
            clauses.append("s.camera_id = ?")
            params.append(camera_id)
        where = " AND ".join(clauses)
        with self._lock:
            sessions = self._connection.execute(
                f"SELECT started_at, finished_at, status FROM sessions s WHERE {where}", params
            ).fetchall()
            telemetry = self._connection.execute(
                "SELECT COUNT(*) AS samples, AVG(fps) AS avg_fps, AVG(blur_variance) AS avg_blur, "
                "AVG(changed_fraction) AS avg_changed, SUM(accepted) AS accepted "
                "FROM pilot_telemetry WHERE camera_id = COALESCE(?, camera_id)",
                (camera_id,),
            ).fetchone()
            event_where = "WHERE camera_id = COALESCE(?, camera_id)"
            candidates = self._connection.execute(
                f"SELECT COUNT(*) FROM events {event_where}", (camera_id,)
            ).fetchone()[0]
            reviews = self._connection.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN decision='accepted' THEN 1 ELSE 0 END) AS accepted "
                "FROM event_reviews r JOIN events e ON e.event_id = r.event_id "
                "WHERE e.camera_id = COALESCE(?, e.camera_id)", (camera_id,)
            ).fetchone()
            health = self._connection.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN state IN ('degraded','unhealthy') THEN 1 ELSE 0 END) AS errors "
                "FROM health_events WHERE camera_id = COALESCE(?, camera_id)", (camera_id,)
            ).fetchone()
        from datetime import datetime, timezone
        uptime = 0.0
        for row in sessions:
            start = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00")) if row["finished_at"] else datetime.now(timezone.utc)
            uptime += max(0.0, (end - start).total_seconds())
        review_total = int(reviews["total"] or 0)
        accepted_reviews = int(reviews["accepted"] or 0)
        return {
            "camera_id": camera_id,
            "sessions": len(sessions),
            "system_uptime_seconds": round(uptime, 3),
            "candidate_events": int(candidates),
            "manual_reviews": review_total,
            "review_agreement_rate": round(accepted_reviews / review_total, 6) if review_total else None,
            "frame_quality": {
                "telemetry_samples": int(telemetry["samples"] or 0),
                "average_fps": round(float(telemetry["avg_fps"] or 0), 6),
                "average_blur_variance": round(float(telemetry["avg_blur"] or 0), 6),
                "average_changed_fraction": round(float(telemetry["avg_changed"] or 0), 6),
                "accepted_frames": int(telemetry["accepted"] or 0),
            },
            "alerts_and_errors": {
                "health_events": int(health["total"] or 0),
                "degraded_or_unhealthy": int(health["errors"] or 0),
            },
        }

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            session_id=row["session_id"],
            camera_id=row["camera_id"],
            event_type=row["event_type"],
            occurred_at=row["occurred_at"],
            confidence=row["confidence"],
            model_version=row["model_version"],
            container_track_id=row["container_track_id"],
            scoop_track_id=row["scoop_track_id"],
            evidence_path=row["evidence_path"],
            evidence_sha256=row["evidence_sha256"],
            review_state=row["review_state"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=row["evidence_id"],
            event_id=row["event_id"],
            relative_path=row["relative_path"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            media_type=row["media_type"],
            created_at=row["created_at"],
            retention_deadline=row["retention_deadline"],
            integrity_status=row["integrity_status"] or "unverified",
            metadata=json.loads(row["metadata_json"]),
        )

    # ── Phase 1: Reconciliation query helpers ────────────────────────────

    def list_all_evidence(self, *, include_deleted: bool = False) -> list[EvidenceRecord]:
        """Return evidence artifact rows, excluding soft-deleted ones by default.

        Reconciliation and consistency checks must not treat a file removed by
        the retention job as a missing/corrupt artifact, so callers get only
        active rows unless they explicitly ask for deleted ones too.
        """
        query = "SELECT * FROM evidence_artifacts"
        if not include_deleted:
            query += " WHERE deleted_at IS NULL"
        query += " ORDER BY created_at"
        with self._lock:
            rows = self._connection.execute(query).fetchall()
        return [self._evidence_from_row(row) for row in rows]

    def list_all_event_evidence_paths(self) -> set[str]:
        """Return distinct evidence_path values referenced by events."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT DISTINCT evidence_path FROM events WHERE evidence_path IS NOT NULL"
            ).fetchall()
        return {row[0] for row in rows}

    def get_event_by_evidence_path(self, path: str) -> EventRecord | None:
        """Find the first event referencing the given evidence path."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM events WHERE evidence_path = ? LIMIT 1",
                (path,),
            ).fetchone()
        return self._event_from_row(row) if row else None

    def update_evidence_integrity(
        self, evidence_id: str, status: str
    ) -> None:
        """Set the integrity_status column for a specific evidence row."""
        if status not in {"unverified", "valid", "corrupt", "missing"}:
            raise ValueError(f"unsupported integrity_status: {status}")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE evidence_artifacts SET integrity_status = ? WHERE evidence_id = ?",
                (status, evidence_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown evidence_id: {evidence_id}")

    def update_event_review_state(
        self, event_id: str, review_state: str, *, extra_metadata: Mapping[str, object] | None = None
    ) -> None:
        """Update the review_state (and optionally merge extra metadata) for an event."""
        if review_state not in {"unreviewed", "accepted", "rejected", "needs_review"}:
            raise ValueError(f"unsupported review_state: {review_state}")
        with self.transaction() as connection:
            if extra_metadata:
                row = connection.execute(
                    "SELECT metadata_json FROM events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown event_id: {event_id}")
                existing = json.loads(row["metadata_json"])
                existing.update(extra_metadata)
                connection.execute(
                    "UPDATE events SET review_state = ?, metadata_json = ? WHERE event_id = ?",
                    (review_state, _canonical_json(existing), event_id),
                )
            else:
                cursor = connection.execute(
                    "UPDATE events SET review_state = ? WHERE event_id = ?",
                    (review_state, event_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown event_id: {event_id}")

    def integrity_check(self) -> bool:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()[0]
        return result == "ok"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteEventRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
