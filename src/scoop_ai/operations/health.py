"""Dependency-free health state and bounded in-process metrics."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping


METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


class HealthState(str, Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    state: HealthState
    message: str
    observed_at: str
    age_seconds: float
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthSnapshot:
    state: HealthState
    ready: bool
    live: bool
    generated_at: str
    components: tuple[ComponentHealth, ...]


@dataclass
class _Heartbeat:
    state: HealthState
    message: str
    observed_at: str
    monotonic_at: float
    details: Mapping[str, object]


class HealthRegistry:
    def __init__(self, *, required_components: set[str] | None = None) -> None:
        self.required_components = frozenset(required_components or set())
        self._components: dict[str, _Heartbeat] = {}
        self._stopping = False
        self._lock = threading.RLock()

    def report(
        self,
        component: str,
        state: HealthState,
        message: str = "",
        *,
        details: Mapping[str, object] | None = None,
        observed_at: datetime | None = None,
        monotonic_at: float | None = None,
    ) -> None:
        if not component.strip():
            raise ValueError("component is required")
        observed_at = observed_at or datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            raise ValueError("observed_at must include a timezone")
        heartbeat = _Heartbeat(
            state=HealthState(state),
            message=message,
            observed_at=observed_at.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            monotonic_at=time.monotonic() if monotonic_at is None else monotonic_at,
            details=dict(details or {}),
        )
        with self._lock:
            self._components[component] = heartbeat

    def mark_stopping(self) -> None:
        with self._lock:
            self._stopping = True

    def snapshot(
        self,
        *,
        stale_after_seconds: float = 10.0,
        stale_state: HealthState = HealthState.UNHEALTHY,
        monotonic_now: float | None = None,
    ) -> HealthSnapshot:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        stale_state = HealthState(stale_state)
        now = time.monotonic() if monotonic_now is None else monotonic_now
        with self._lock:
            components = dict(self._components)
            stopping = self._stopping

        output: list[ComponentHealth] = []
        effective_states: dict[str, HealthState] = {}
        for name, heartbeat in sorted(components.items()):
            age = max(0.0, now - heartbeat.monotonic_at)
            state = heartbeat.state
            message = heartbeat.message
            if age > stale_after_seconds and state not in {
                HealthState.UNHEALTHY,
                HealthState.STOPPING,
            }:
                state = stale_state
                message = f"stale heartbeat ({age:.1f}s)"
            effective_states[name] = state
            output.append(
                ComponentHealth(
                    name=name,
                    state=state,
                    message=message,
                    observed_at=heartbeat.observed_at,
                    age_seconds=round(age, 3),
                    details=heartbeat.details,
                )
            )

        if stopping:
            overall = HealthState.STOPPING
        elif not self.required_components.issubset(effective_states):
            overall = HealthState.STARTING
        else:
            required_states = [effective_states[name] for name in self.required_components]
            if any(state == HealthState.UNHEALTHY for state in required_states):
                overall = HealthState.UNHEALTHY
            elif any(
                state in {HealthState.DEGRADED, HealthState.STARTING}
                for state in required_states
            ):
                overall = HealthState.DEGRADED
            else:
                overall = HealthState.HEALTHY
        return HealthSnapshot(
            state=overall,
            ready=overall == HealthState.HEALTHY,
            live=overall not in {HealthState.UNHEALTHY, HealthState.STOPPING},
            generated_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            components=tuple(output),
        )


class MetricsRegistry:
    """Small thread-safe metric collector suitable for JSON/status export."""

    def __init__(self, *, sample_limit: int = 2048, max_metric_names: int = 128) -> None:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        if max_metric_names <= 0:
            raise ValueError("max_metric_names must be positive")
        self._sample_limit = sample_limit
        self._max_metric_names = max_metric_names
        self._known_names: set[str] = set()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate(name: str, value: float) -> None:
        if not METRIC_NAME.fullmatch(name):
            raise ValueError(f"invalid metric name: {name!r}")
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")

    def _claim_name(self, name: str) -> None:
        if name in self._known_names:
            return
        if len(self._known_names) >= self._max_metric_names:
            raise ValueError("metric name cardinality limit exceeded")
        self._known_names.add(name)

    def increment(self, name: str, amount: float = 1.0) -> None:
        self._validate(name, amount)
        if amount < 0:
            raise ValueError("counter increments cannot be negative")
        with self._lock:
            self._claim_name(name)
            self._counters[name] += amount

    def gauge(self, name: str, value: float) -> None:
        self._validate(name, value)
        with self._lock:
            self._claim_name(name)
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        self._validate(name, value)
        with self._lock:
            self._claim_name(name)
            samples = self._samples.setdefault(name, deque(maxlen=self._sample_limit))
            samples.append(value)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * percentile))
        return ordered[index]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            samples = {name: list(values) for name, values in self._samples.items()}
        summaries = {
            name: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 0.50),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
            }
            for name, values in samples.items()
            if values
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "counters": counters,
            "gauges": gauges,
            "summaries": summaries,
        }
