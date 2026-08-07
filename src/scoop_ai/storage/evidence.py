"""Crash-safe evidence file writes with checksums and traversal protection."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".mp4", ".webm"})


@dataclass(frozen=True)
class EvidenceArtifact:
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class EvidenceWriter:
    def __init__(
        self,
        root: Path | str,
        *,
        allowed_extensions: frozenset[str] = DEFAULT_EXTENSIONS,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_extensions = frozenset(ext.lower() for ext in allowed_extensions)

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        if not SAFE_COMPONENT.fullmatch(value):
            raise ValueError(f"{field_name} contains unsafe path characters")
        return value

    def _target(self, session_id: str, evidence_id: str, extension: str) -> Path:
        session_id = self._safe_component(session_id, "session_id")
        evidence_id = self._safe_component(evidence_id, "evidence_id")
        extension = extension.lower()
        if not extension.startswith("."):
            extension = "." + extension
        if extension not in self.allowed_extensions:
            raise ValueError(f"unsupported evidence extension: {extension}")
        target = (self.root / session_id / f"{evidence_id}{extension}").resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("evidence target escapes configured root")
        return target

    def write_bytes(
        self,
        session_id: str,
        evidence_id: str,
        payload: bytes,
        *,
        extension: str = ".jpg",
        replace: bool = False,
    ) -> EvidenceArtifact:
        return self.write_stream(
            session_id,
            evidence_id,
            _BytesReader(payload),
            extension=extension,
            replace=replace,
        )

    def write_stream(
        self,
        session_id: str,
        evidence_id: str,
        stream: BinaryIO,
        *,
        extension: str = ".jpg",
        replace: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> EvidenceArtifact:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        target = self._target(session_id, evidence_id, extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("evidence stream must return bytes")
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())

            if replace:
                os.replace(temporary_path, target)
            else:
                # A hard link publishes the fully fsynced temporary file atomically
                # and fails safely if another writer already owns this event ID.
                os.link(temporary_path, target)
                temporary_path.unlink()
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        relative_path = target.relative_to(self.root).as_posix()
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return EvidenceArtifact(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size_bytes=size,
            media_type=media_type,
            created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )

    def verify(self, artifact: EvidenceArtifact) -> bool:
        path = (self.root / artifact.relative_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.is_symlink():
            return False
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return size == artifact.size_bytes and digest.hexdigest() == artifact.sha256


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        self._payload = payload
        self._consumed = False

    def read(self, _: int = -1) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        return self._payload
