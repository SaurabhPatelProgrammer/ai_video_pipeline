"""Security boundaries for camera sources, logs and executable checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|credential|authorization)",
    re.IGNORECASE,
)
URL_IN_TEXT = re.compile(r"\b(?:rtsp|rtsps|http|https)://[^\s\"'<>]+", re.IGNORECASE)
KEY_VALUE_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
CHECKPOINT_EXTENSIONS = frozenset({".pth", ".pt", ".safetensors"})
# Credential references may be hierarchical. Keep this grammar aligned with
# ``camera.credential_key`` in config.py so example configurations can be
# resolved by the same key that credential-set provisions.
CREDENTIAL_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
DEFAULT_CREDENTIAL_SERVICE = "scoop-ai"


class CredentialBackendError(RuntimeError):
    """Raised when the OS credential backend is unavailable or fails."""


class CredentialNotFoundError(LookupError):
    """Raised when an expected credential has not been provisioned."""


def _credential_backend() -> Any:
    try:
        backend = importlib.import_module("keyring")
    except ImportError as exc:
        raise CredentialBackendError(
            "OS credential backend unavailable; install and configure 'keyring' "
            "for the production service account"
        ) from exc
    return backend


def _validate_credential_key(key: str) -> str:
    if not CREDENTIAL_KEY.fullmatch(key):
        raise ValueError("credential key contains unsafe characters")
    return key


def resolve_credential(
    key: str,
    *,
    service_name: str = DEFAULT_CREDENTIAL_SERVICE,
) -> str:
    """Read a secret from the OS backend; deliberately never fall back to env."""
    key = _validate_credential_key(key)
    if not service_name.strip():
        raise ValueError("credential service_name is required")
    backend = _credential_backend()
    try:
        value = backend.get_password(service_name, key)
    except Exception as exc:
        raise CredentialBackendError(
            f"OS credential backend could not read key {key!r}"
        ) from exc
    if value is None:
        raise CredentialNotFoundError(f"credential {key!r} is not provisioned")
    return str(value)


def store_credential(
    key: str,
    value: str,
    *,
    service_name: str = DEFAULT_CREDENTIAL_SERVICE,
) -> None:
    """Provision a non-empty secret in the OS backend."""
    key = _validate_credential_key(key)
    if not service_name.strip():
        raise ValueError("credential service_name is required")
    if not isinstance(value, str) or not value:
        raise ValueError("credential value must be a non-empty string")
    backend = _credential_backend()
    try:
        backend.set_password(service_name, key, value)
    except Exception as exc:
        raise CredentialBackendError(
            f"OS credential backend could not store key {key!r}"
        ) from exc


def safe_source_name(source: int | str) -> str:
    """Return a useful source label without userinfo, query values or fragments."""
    if isinstance(source, int):
        return f"webcam:{source}"
    text = str(source).strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return Path(text).name or "video-source"
    if not parts.scheme or not parts.hostname:
        return Path(text).name or "video-source"
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parts.port}" if parts.port is not None else ""
    except ValueError:
        port = ""
    return urlunsplit((parts.scheme.lower(), f"{host}{port}", parts.path, "", ""))


def redact_text(value: str) -> str:
    """Best-effort redaction for structured-log messages."""
    value = URL_IN_TEXT.sub(lambda match: safe_source_name(match.group(0)), value)
    return KEY_VALUE_SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value)


def redact_secrets(value: Any, *, key: str | None = None) -> Any:
    """Recursively sanitize values before serialization to a log sink."""
    if key is not None and SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, str):
        if key is not None and key.lower() in {"source", "camera_url", "url"}:
            return safe_source_name(value)
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_secrets(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_secrets(item) for item in value]
    return value


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedCheckpoint:
    path: Path
    sha256: str
    size_bytes: int


def verify_checkpoint(
    path: Path | str,
    allowed_sha256: Sequence[str] | set[str] | frozenset[str],
    *,
    allowed_root: Path | str | None = None,
    maximum_bytes: int = 8 * 1024 * 1024 * 1024,
    allowed_extensions: frozenset[str] = CHECKPOINT_EXTENSIONS,
) -> VerifiedCheckpoint:
    """Verify an approved model before handing it to a pickle-capable loader."""
    checkpoint = Path(path)
    if checkpoint.is_symlink():
        raise ValueError("checkpoint symlinks are not allowed")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be a regular file")
    if checkpoint.suffix.lower() not in allowed_extensions:
        raise ValueError(f"unsupported checkpoint extension: {checkpoint.suffix}")
    if allowed_root is not None:
        root = Path(allowed_root).resolve(strict=True)
        if not checkpoint.is_relative_to(root):
            raise ValueError("checkpoint is outside the approved model directory")
    stat = checkpoint.stat()
    if maximum_bytes <= 0 or stat.st_size > maximum_bytes:
        raise ValueError("checkpoint exceeds the configured size limit")

    normalized: list[str] = []
    for value in allowed_sha256:
        digest = value.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("allow-list contains an invalid SHA-256 digest")
        normalized.append(digest)
    if not normalized:
        raise ValueError("checkpoint SHA-256 allow-list cannot be empty")

    actual = sha256_file(checkpoint)
    if not any(hmac.compare_digest(actual, expected) for expected in normalized):
        raise PermissionError("checkpoint SHA-256 is not approved")
    return VerifiedCheckpoint(path=checkpoint, sha256=actual, size_bytes=stat.st_size)
