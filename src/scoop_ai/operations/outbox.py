"""Signed, retrying HTTP worker for reviewed-event exports."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..security import redact_text
from ..storage import OutboxRecord, SQLiteEventRepository


def sign_export_batch(events: list[OutboxRecord], signing_key: bytes) -> str:
    if not signing_key:
        raise ValueError("signing_key cannot be empty")
    payload = json.dumps(
        [{"event_id": event.event_id, "payload": event.payload} for event in events],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "hmac-sha256=" + hmac.new(signing_key, payload, hashlib.sha256).hexdigest()


class OutboxWorker(threading.Thread):
    def __init__(
        self,
        repository: SQLiteEventRepository,
        *,
        endpoint: str,
        signing_key: bytes,
        stop_event: threading.Event,
        poll_seconds: float = 2.0,
        max_attempts: int = 5,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 10.0,
        opener=urlopen,
    ) -> None:
        super().__init__(name="outbox-worker", daemon=True)
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("export endpoint must use HTTP or HTTPS")
        if poll_seconds <= 0 or max_attempts < 1 or backoff_seconds <= 0:
            raise ValueError("invalid outbox worker timing configuration")
        self.repository = repository
        self.endpoint = endpoint
        self.signing_key = signing_key
        self.stop_event = stop_event
        self.poll_seconds = poll_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def run(self) -> None:
        while not self.stop_event.is_set():
            processed = self.process_once()
            if not processed:
                self.stop_event.wait(self.poll_seconds)

    def process_once(self) -> int:
        events = self.repository.claim_outbox(limit=20)
        if not events:
            return 0
        for event in events:
            self._send(event)
        return len(events)

    def _send(self, event: OutboxRecord) -> None:
        signature = sign_export_batch([event], self.signing_key)
        body = json.dumps(
            {"events": [{"event_id": event.event_id, "payload": event.payload}], "signature": signature},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self.endpoint, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": event.event_id,
                "X-Scoop-Export-Signature": signature,
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                if not 200 <= int(response.status) < 300:
                    raise RuntimeError(f"downstream returned HTTP {response.status}")
            self.repository.mark_outbox_acknowledged(event.event_id, signature=signature)
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            delay = self.backoff_seconds * (2 ** max(0, event.attempts))
            next_attempt = datetime.now(timezone.utc) + timedelta(seconds=delay)
            self.repository.mark_outbox_failure(
                event.event_id,
                error=redact_text(str(exc)),
                next_attempt_at=next_attempt.isoformat(timespec="milliseconds"),
                max_attempts=self.max_attempts,
            )
