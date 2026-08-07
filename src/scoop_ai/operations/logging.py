"""Credential-safe JSON logging for the edge service."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator, TextIO

from scoop_ai.security import redact_secrets


_CONTEXT: ContextVar[dict[str, object]] = ContextVar("log_context", default={})
_STANDARD_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


@contextmanager
def log_context(**fields: object) -> Iterator[None]:
    """Attach camera/session/event identifiers to logs in the current context."""
    merged = {**_CONTEXT.get(), **fields}
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class JsonFormatter(logging.Formatter):
    def __init__(self, *, service_name: str = "scoop-ai") -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            **_CONTEXT.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            redact_secrets(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


def configure_structured_logging(
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    service_name: str = "scoop-ai",
) -> logging.Logger:
    """Configure one replaceable root handler and return the service logger."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(service_name=service_name))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger(service_name)
