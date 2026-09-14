"""Compute-target resolution so the same bundle runs on GPU and CPU machines.

Every helper imports Torch lazily. Configuration validation, the dashboard, the
desktop shell, and the test suite all need to describe the available hardware on
machines where Torch is absent or where importing it would be wasteful.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEVICE_PREFERENCES = ("auto", "cuda", "cpu")

# A CPU forward pass through RF-DETR nano costs roughly 0.4-1.5s per frame on a
# current desktop part. Analysing above this rate cannot keep up with the camera,
# so the queue grows without bound and evidence timestamps drift.
CPU_ANALYSIS_FPS_CEILING = 2.0
CUDA_ANALYSIS_FPS_CEILING = 10.0


class ComputeError(RuntimeError):
    """Raised when a requested compute target is unavailable."""


@dataclass(frozen=True, slots=True)
class ComputeCapability:
    """A snapshot of what this machine can run, safe to serialise to the UI."""

    device: str
    cuda_available: bool
    device_name: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    total_memory_bytes: int | None = None
    cpu_threads: int | None = None
    torch_error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accelerated(self) -> bool:
        return self.device == "cuda"

    @property
    def recommended_analysis_fps(self) -> float:
        return CUDA_ANALYSIS_FPS_CEILING if self.accelerated else CPU_ANALYSIS_FPS_CEILING

    @property
    def headline(self) -> str:
        if self.torch_error is not None:
            return "AI runtime unavailable"
        if self.accelerated:
            return f"GPU acceleration - {self.device_name or 'NVIDIA GPU'}"
        return f"CPU only - {self.cpu_threads or '?'} threads"

    def as_payload(self) -> dict[str, object]:
        return {
            "device": self.device,
            "accelerated": self.accelerated,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "total_memory_bytes": self.total_memory_bytes,
            "cpu_threads": self.cpu_threads,
            "recommended_analysis_fps": self.recommended_analysis_fps,
            "headline": self.headline,
            "torch_error": self.torch_error,
            "warnings": list(self.warnings),
        }


def normalise_preference(value: object, name: str = "device") -> str:
    """Validate a configured preference without touching Torch."""

    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ComputeError(f"{name} must be one of {', '.join(DEVICE_PREFERENCES)}")
    preference = value.strip().lower()
    if preference in {"", "default"}:
        return "auto"
    if preference in {"gpu", "nvidia"}:
        preference = "cuda"
    if preference not in DEVICE_PREFERENCES:
        raise ComputeError(f"{name} must be one of {', '.join(DEVICE_PREFERENCES)}")
    return preference


def _logical_cpus() -> int:
    # os.process_cpu_count honours affinity masks, which matters on machines that
    # pin the edge service to a subset of cores.
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, int(count))


def describe_compute(preference: str = "auto") -> ComputeCapability:
    """Report the effective device, degrading to CPU rather than failing."""

    preference = normalise_preference(preference)
    threads = _logical_cpus()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without Torch
        return ComputeCapability(
            device="cpu",
            cuda_available=False,
            cpu_threads=threads,
            torch_error=str(exc),
            warnings=("PyTorch is not installed; run setup.ps1 before monitoring.",),
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken driver installs
        cuda_available = False

    warnings: list[str] = []
    if preference == "cuda" and not cuda_available:
        warnings.append(
            "GPU was requested but no working NVIDIA runtime was found; using the CPU."
        )
        device = "cpu"
    elif preference == "cpu":
        device = "cpu"
    else:
        device = "cuda" if cuda_available else "cpu"

    device_name: str | None = None
    cuda_version: str | None = None
    total_memory: int | None = None
    if device == "cuda":
        try:
            device_name = str(torch.cuda.get_device_name(0))
            total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:  # pragma: no cover - driver reported a device it cannot open
            warnings.append("The NVIDIA device could not be queried; using the CPU.")
            device = "cpu"
        cuda_version = getattr(torch.version, "cuda", None)
    if device == "cpu":
        device_name = None
        total_memory = None
        warnings.append(
            "CPU inference is far slower than GPU inference. "
            f"Keep analysis_fps at or below {CPU_ANALYSIS_FPS_CEILING:g}."
        )

    return ComputeCapability(
        device=device,
        cuda_available=cuda_available,
        device_name=device_name,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        cuda_version=cuda_version,
        total_memory_bytes=total_memory,
        cpu_threads=threads,
        warnings=tuple(warnings),
    )


def resolve_device(preference: str = "auto") -> str:
    """Return the Torch device string this machine should actually use."""

    return describe_compute(preference).device


def configure_torch_threads(device: str, *, reserve_cores: int = 1) -> int:
    """Cap CPU inference threads so capture and the UI keep a core to run on.

    Torch defaults to every logical core, which starves the capture thread and
    makes frames arrive late. Returns the thread count that was applied.
    """

    if device != "cpu":
        return 0
    try:
        import torch
    except Exception:  # pragma: no cover - exercised only without Torch
        return 0
    threads = max(1, _logical_cpus() - max(0, reserve_cores))
    try:
        torch.set_num_threads(threads)
    except Exception:  # pragma: no cover - some builds forbid late changes
        return 0
    return threads


def clamp_analysis_fps(analysis_fps: float, device: str) -> float:
    """Hold the configured rate to something the device can actually sustain."""

    ceiling = CUDA_ANALYSIS_FPS_CEILING if device == "cuda" else CPU_ANALYSIS_FPS_CEILING
    return min(float(analysis_fps), ceiling)
