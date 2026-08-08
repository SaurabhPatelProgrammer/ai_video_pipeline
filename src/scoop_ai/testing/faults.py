"""Deterministic failure injection without production-only dependencies."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Callable, TypeVar


class InjectedFault(RuntimeError):
    """Base class for an intentionally injected operational failure."""


class InjectedGPUOutOfMemory(InjectedFault): pass
class InjectedDatabaseLock(InjectedFault): pass
class InjectedDiskFull(InjectedFault): pass
class InjectedFrameDropout(InjectedFault): pass
class InjectedNetworkLatency(InjectedFault): pass


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    after: int = 1
    delay_seconds: float = 0.0
    repeat: bool = False

    def __post_init__(self) -> None:
        if self.after < 1 or self.delay_seconds < 0:
            raise ValueError("fault after must be >= 1 and delay_seconds cannot be negative")


_FAULTS = {
    "gpu_oom": InjectedGPUOutOfMemory,
    "database_lock": InjectedDatabaseLock,
    "disk_full": InjectedDiskFull,
    "frame_dropout": InjectedFrameDropout,
    "network_latency": InjectedNetworkLatency,
}


class FaultInjector:
    def __init__(self, spec: FaultSpec) -> None:
        if spec.kind not in _FAULTS:
            raise ValueError(f"unsupported fault kind: {spec.kind}")
        self.spec = spec
        self.count = 0
        self._triggered = False
        self._lock = Lock()

    def checkpoint(self) -> None:
        with self._lock:
            self.count += 1
            should_trigger = self.count >= self.spec.after and (self.spec.repeat or not self._triggered)
            if should_trigger:
                self._triggered = True
        if not should_trigger:
            return
        if self.spec.kind == "network_latency":
            time.sleep(self.spec.delay_seconds)
            return
        exception = _FAULTS[self.spec.kind]
        raise exception(f"injected {self.spec.kind} at call {self.count}")


F = TypeVar("F", bound=Callable[..., object])


def fault_injected(kind: str, *, after: int = 1, delay_seconds: float = 0.0, repeat: bool = False):
    """Decorate a call site with a deterministic injected fault."""
    injector = FaultInjector(FaultSpec(kind, after, delay_seconds, repeat))
    def decorator(function: F) -> F:
        @wraps(function)
        def wrapped(*args: object, **kwargs: object):
            injector.checkpoint()
            return function(*args, **kwargs)
        return wrapped  # type: ignore[return-value]
    return decorator


class FaultyCaptureSource:
    """Proxy that drops selected reads and/or adds deterministic latency."""

    def __init__(self, source: object, *, drop_every: int | None = None, latency_seconds: float = 0.0) -> None:
        if drop_every is not None and drop_every < 1:
            raise ValueError("drop_every must be positive")
        if latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")
        self.source = source
        self.drop_every = drop_every
        self.latency_seconds = latency_seconds
        self.read_count = 0

    def read(self, *args: object, **kwargs: object):
        self.read_count += 1
        if self.latency_seconds:
            time.sleep(self.latency_seconds)
        if self.drop_every is not None and self.read_count % self.drop_every == 0:
            return None
        return self.source.read(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.source, name)
