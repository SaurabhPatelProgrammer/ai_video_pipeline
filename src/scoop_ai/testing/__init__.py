"""Small fault-injection primitives used by recovery tests and drills."""

from .faults import (
    FaultInjector,
    FaultSpec,
    FaultyCaptureSource,
    InjectedDatabaseLock,
    InjectedDiskFull,
    InjectedFrameDropout,
    InjectedGPUOutOfMemory,
    InjectedNetworkLatency,
    fault_injected,
)

__all__ = [
    "FaultInjector", "FaultSpec", "FaultyCaptureSource", "InjectedDatabaseLock",
    "InjectedDiskFull", "InjectedFrameDropout", "InjectedGPUOutOfMemory",
    "InjectedNetworkLatency", "fault_injected",
]
