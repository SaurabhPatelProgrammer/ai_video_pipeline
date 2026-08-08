"""WAL-safe SQLite backup and restore helpers for the edge service."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import MIGRATIONS


class BackupError(RuntimeError):
    """Raised when a backup or restore cannot be trusted or completed."""


CURRENT_SCHEMA_VERSION = max(version for version, _ in MIGRATIONS)
SUPPORTED_SCHEMA_VERSIONS = frozenset(version for version, _ in MIGRATIONS)
BACKUP_SUFFIX = ".sqlite3"
METADATA_SUFFIX = ".metadata.json"
_COUNT_TABLES = (
    "events",
    "event_reviews",
    "ground_truth_events",
    "model_versions",
)


@dataclass(frozen=True)
class BackupResult:
    backup_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    deleted_generations: tuple[str, ...] = ()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows requires a writable handle for FlushFileBuffers/fsync.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _schema_version(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0])


def _record_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in _COUNT_TABLES:
        try:
            counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            counts[table] = 0
    return counts


def _integrity_ok(connection: sqlite3.Connection) -> bool:
    return connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _fsync_file(temporary)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise BackupError(f"could not write backup metadata {path}: {exc}") from exc


def _backup_stem(path: Path) -> str:
    return path.name.removesuffix(BACKUP_SUFFIX)


def _cleanup_generations(output_dir: Path, retain_generations: int) -> tuple[str, ...]:
    if retain_generations < 1:
        raise ValueError("retain_generations must be at least 1")
    metadata_files = sorted(
        output_dir.glob(f"*{METADATA_SUFFIX}"), key=lambda item: item.name, reverse=True
    )
    deleted: list[str] = []
    for metadata_path in metadata_files[retain_generations:]:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            backup_name = Path(str(metadata["backup_file"])).name
            backup_path = output_dir / backup_name
            if backup_path.parent != output_dir or backup_path.suffix != BACKUP_SUFFIX:
                continue
            metadata_path.unlink()
            backup_path.unlink(missing_ok=True)
            deleted.append(_backup_stem(backup_path))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Never delete an unrecognizable generation automatically.
            continue
    return tuple(deleted)


def create_backup(
    database_path: Path | str,
    output_dir: Path | str,
    *,
    retain_generations: int = 5,
) -> BackupResult:
    """Create a consistent backup using SQLite's online backup API."""

    source = Path(database_path).resolve()
    destination_dir = Path(output_dir).resolve()
    if not source.is_file():
        raise BackupError(f"database not found: {source}")
    if retain_generations < 1:
        raise ValueError("retain_generations must be at least 1")
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupError(f"could not create backup directory {destination_dir}: {exc}") from exc

    stem = f"{source.stem}-{_timestamp()}-{uuid.uuid4().hex[:8]}"
    backup_path = destination_dir / f"{stem}{BACKUP_SUFFIX}"
    metadata_path = destination_dir / f"{stem}{METADATA_SUFFIX}"
    temporary_path = destination_dir / f".{stem}.{uuid.uuid4().hex}.tmp"
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(source, uri=False, timeout=5)
        if not _integrity_ok(source_connection):
            raise BackupError("source database failed PRAGMA integrity_check")
        schema_version = _schema_version(source_connection)
        counts = _record_counts(source_connection)
        target_connection = sqlite3.connect(temporary_path)
        source_connection.backup(target_connection)
        target_connection.commit()
        if not _integrity_ok(target_connection):
            raise BackupError("backup failed PRAGMA integrity_check")
    except (OSError, sqlite3.Error) as exc:
        raise BackupError(f"database backup failed: {exc}") from exc
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()

    try:
        _fsync_file(temporary_path)
        os.replace(temporary_path, backup_path)
        size_bytes = backup_path.stat().st_size
        checksum = _sha256(backup_path)
        metadata = {
            "backup_file": backup_path.name,
            "source_database_name": source.name,
            "schema_version": schema_version,
            "backup_timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "size_bytes": size_bytes,
            "sha256": checksum,
            "record_counts": counts,
        }
        _atomic_json(metadata_path, metadata)
        deleted = _cleanup_generations(destination_dir, retain_generations)
        return BackupResult(backup_path, metadata_path, metadata, deleted)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        backup_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise BackupError(f"could not publish backup: {exc}") from exc


def _load_metadata(backup_path: Path) -> dict[str, Any]:
    metadata_path = backup_path.with_name(f"{_backup_stem(backup_path)}{METADATA_SUFFIX}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError(f"backup metadata is missing or invalid: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise BackupError("backup metadata must be a JSON object")
    required = {"backup_file", "source_database_name", "schema_version", "backup_timestamp", "size_bytes", "sha256", "record_counts"}
    if not required.issubset(metadata):
        raise BackupError("backup metadata is incomplete")
    if Path(str(metadata["backup_file"])).name != backup_path.name:
        raise BackupError("backup metadata does not match backup filename")
    if int(metadata["schema_version"]) not in SUPPORTED_SCHEMA_VERSIONS:
        raise BackupError(f"unsupported backup schema version: {metadata['schema_version']}")
    if int(metadata["size_bytes"]) != backup_path.stat().st_size:
        raise BackupError("backup size does not match metadata")
    if _sha256(backup_path) != str(metadata["sha256"]).lower():
        raise BackupError("backup SHA-256 does not match metadata")
    counts = metadata["record_counts"]
    if not isinstance(counts, dict) or any(table not in counts for table in _COUNT_TABLES):
        raise BackupError("backup record counts are incomplete")
    return metadata


def restore_backup(
    backup_path: Path | str,
    output_dir: Path | str,
    *,
    database_name: str = "events.sqlite3",
) -> dict[str, Any]:
    """Restore a verified backup into a new directory without overwriting files."""

    backup = Path(backup_path).resolve()
    destination_dir = Path(output_dir).resolve()
    if not backup.is_file():
        raise BackupError(f"backup not found: {backup}")
    if Path(database_name).name != database_name or Path(database_name).suffix != BACKUP_SUFFIX:
        raise ValueError("database_name must be a filename ending in .sqlite3")
    metadata = _load_metadata(backup)
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupError(f"could not create restore directory {destination_dir}: {exc}") from exc
    restored = destination_dir / database_name
    if restored.exists() or restored.with_name(f"{restored.name}-wal").exists() or restored.with_name(f"{restored.name}-shm").exists():
        raise BackupError(f"refusing to overwrite existing restore target: {restored}")
    temporary = destination_dir / f".{database_name}.{uuid.uuid4().hex}.tmp"
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(backup, uri=False, timeout=5)
        target_connection = sqlite3.connect(temporary)
        source_connection.backup(target_connection)
        target_connection.commit()
        if not _integrity_ok(target_connection):
            raise BackupError("restored database failed PRAGMA integrity_check")
    except (OSError, sqlite3.Error) as exc:
        raise BackupError(f"database restore failed: {exc}") from exc
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()

    try:
        _fsync_file(temporary)
        os.replace(temporary, restored)
        verification = sqlite3.connect(restored)
        try:
            if not _integrity_ok(verification):
                raise BackupError("restored database failed final integrity check")
            counts = _record_counts(verification)
        finally:
            verification.close()
        expected = {table: int(metadata["record_counts"][table]) for table in _COUNT_TABLES}
        if counts != expected:
            raise BackupError(f"restored record counts do not match metadata: {counts} != {expected}")
        return {
            "restored_database": str(restored),
            "source_backup": str(backup),
            "schema_version": int(metadata["schema_version"]),
            "integrity": "ok",
            "record_counts": counts,
        }
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        restored.unlink(missing_ok=True)
        raise BackupError(f"could not publish restored database: {exc}") from exc
    except BackupError:
        temporary.unlink(missing_ok=True)
        restored.unlink(missing_ok=True)
        raise
