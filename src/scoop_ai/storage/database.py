"""SQLite WAL event repository with small, explicit schema migrations."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
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

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

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
            return True

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
