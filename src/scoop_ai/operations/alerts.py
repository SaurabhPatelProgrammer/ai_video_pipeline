"""Bounded operational alert monitor backed by health events and logs."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.database import HealthEventRecord, SQLiteEventRepository, utc_now_iso
from .health import HealthRegistry, HealthState, MetricsRegistry


LOGGER = logging.getLogger("scoop-ai.alerts")


@dataclass(frozen=True, slots=True)
class AlertThresholds:
    poll_seconds: float = 5.0
    stale_after_seconds: float = 10.0
    reconnect_warning: int = 3
    reconnect_critical: int = 10
    low_disk_warning_gb: float = 10.0
    low_disk_critical_gb: float = 2.0


class AlertMonitor:
    """Poll bounded health inputs and emit only alert state transitions."""

    def __init__(
        self,
        health: HealthRegistry,
        metrics: MetricsRegistry,
        repository: SQLiteEventRepository,
        *,
        camera_id: str,
        artifact_root: Path | str,
        reader_supplier: Any,
        thresholds: AlertThresholds | None = None,
        stop_event: threading.Event | None = None,
        monotonic: Any = None,
    ) -> None:
        self.health = health
        self.metrics = metrics
        self.repository = repository
        self.camera_id = camera_id
        self.artifact_root = Path(artifact_root).resolve()
        self.reader_supplier = reader_supplier
        self.thresholds = thresholds or AlertThresholds()
        if self.thresholds.poll_seconds <= 0 or self.thresholds.stale_after_seconds <= 0:
            raise ValueError("alert polling and stale thresholds must be positive")
        if self.thresholds.reconnect_warning < 1 or self.thresholds.reconnect_critical < self.thresholds.reconnect_warning:
            raise ValueError("reconnect alert thresholds are invalid")
        if self.thresholds.low_disk_critical_gb > self.thresholds.low_disk_warning_gb:
            raise ValueError("critical disk threshold cannot exceed warning threshold")
        self.stop_event = stop_event or threading.Event()
        self._monotonic = monotonic or time.monotonic
        self._states: dict[str, str] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> "AlertMonitor":
        if self._thread is not None:
            raise RuntimeError("alert monitor cannot be started twice")
        self._thread = threading.Thread(target=self._run, name="scoop-ai-alerts", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, timeout))

    def evaluate_once(self) -> None:
        snapshot = self.health.snapshot(
            stale_after_seconds=self.thresholds.stale_after_seconds,
            stale_state=HealthState.DEGRADED,
        )
        for component in snapshot.components:
            if component.state in {HealthState.DEGRADED, HealthState.UNHEALTHY}:
                severity = "critical" if component.state is HealthState.UNHEALTHY else "warning"
                self._transition(
                    f"health.{component.name}",
                    component.name,
                    severity,
                    component.state,
                    component.message,
                    {"age_seconds": component.age_seconds, **dict(component.details)},
                )
            else:
                self._clear(f"health.{component.name}", component.name)

        reader = self.reader_supplier()
        if reader is not None:
            capture = reader.health
            reconnects = capture.consecutive_failures
            self.metrics.gauge("camera_fps", float(getattr(capture, "frames_per_second", 0.0)))
            self.metrics.gauge("camera_reconnect_failures", float(reconnects))
            frame_age = capture.frame_age_seconds(self._monotonic())
            self.metrics.gauge("last_frame_age_seconds", float(frame_age or 0.0))
            if reconnects >= self.thresholds.reconnect_critical:
                self._transition(
                    "camera.reconnects",
                    "capture",
                    "critical",
                    HealthState.UNHEALTHY,
                    "camera reconnect failure threshold exceeded",
                    {"consecutive_failures": reconnects},
                )
            elif reconnects >= self.thresholds.reconnect_warning:
                self._transition(
                    "camera.reconnects",
                    "capture",
                    "warning",
                    HealthState.DEGRADED,
                    "camera reconnect warning threshold exceeded",
                    {"consecutive_failures": reconnects},
                )
            else:
                self._clear("camera.reconnects", "capture")

        free_bytes = shutil.disk_usage(self.artifact_root).free
        self.metrics.gauge("free_storage_bytes", float(free_bytes))
        warning_bytes = self.thresholds.low_disk_warning_gb * 1024**3
        critical_bytes = self.thresholds.low_disk_critical_gb * 1024**3
        if free_bytes < critical_bytes:
            self._transition(
                "storage.disk",
                "storage",
                "critical",
                HealthState.UNHEALTHY,
                "free storage is below critical threshold",
                {"free_bytes": free_bytes},
            )
        elif free_bytes < warning_bytes:
            self._transition(
                "storage.disk",
                "storage",
                "warning",
                HealthState.DEGRADED,
                "free storage is below warning threshold",
                {"free_bytes": free_bytes},
            )
        else:
            self._clear("storage.disk", "storage")

        self.metrics.gauge("database_write_lock_seconds", self.repository.last_write_lock_seconds)

    def _run(self) -> None:
        while not self.stop_event.wait(self.thresholds.poll_seconds):
            try:
                self.evaluate_once()
            except Exception:
                LOGGER.exception("alert evaluation failed")

    def _clear(self, code: str, component: str) -> None:
        if code not in self._states:
            return
        self._transition(
            code,
            component,
            "info",
            HealthState.HEALTHY,
            "alert condition recovered",
            {},
        )

    def _transition(
        self,
        code: str,
        component: str,
        severity: str,
        state: HealthState,
        message: str,
        details: dict[str, object],
    ) -> None:
        state_key = f"{severity}:{state.value}"
        if self._states.get(code) == state_key:
            return
        alert_id = str(uuid.uuid4())
        payload = {"alert_id": alert_id, "alert_code": code, "severity": severity, **details}
        self.repository.record_health_event(
            HealthEventRecord(
                health_event_id=alert_id,
                camera_id=self.camera_id,
                component=component,
                state=state.value,
                occurred_at=utc_now_iso(),
                message=message,
                details=payload,
            )
        )
        self._states[code] = state_key
        if severity in {"warning", "critical"}:
            LOGGER.warning("operational alert: %s", message, extra=payload)
        else:
            LOGGER.info("operational alert recovered: %s", message, extra=payload)
