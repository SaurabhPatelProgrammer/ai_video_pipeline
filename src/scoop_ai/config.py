"""Strict TOML configuration for the production capture service."""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when configuration is missing, ambiguous, or unsafe."""


def _unknown_keys(data: Mapping[str, object], allowed: set[str], section: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown key(s) in {section}: {sorted(unknown)}")


def _required_table(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{key}] table is required")
    return value


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be finite")
    invalid = number < 0 if allow_zero else number <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"{name} must be {qualifier}")
    return number


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    reconnect_seconds: float = 3.0
    reconnect_max_seconds: float = 30.0
    reconnect_jitter_ratio: float = 0.20
    open_timeout_ms: int = 5000
    read_timeout_ms: int = 5000
    read_wait_seconds: float = 2.0
    rtsp_transport: str = "tcp"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object], section: str = "capture") -> "CaptureConfig":
        _unknown_keys(
            data,
            {
                "reconnect_seconds",
                "reconnect_max_seconds",
                "reconnect_jitter_ratio",
                "open_timeout_ms",
                "read_timeout_ms",
                "read_wait_seconds",
                "rtsp_transport",
            },
            section,
        )
        transport = data.get("rtsp_transport", "tcp")
        if transport not in {"tcp", "udp"}:
            raise ConfigurationError(f"{section}.rtsp_transport must be 'tcp' or 'udp'")
        reconnect_seconds = _positive_number(
            data.get("reconnect_seconds", 3.0),
            f"{section}.reconnect_seconds",
            allow_zero=True,
        )
        reconnect_max_seconds = _positive_number(
            data.get("reconnect_max_seconds", 30.0),
            f"{section}.reconnect_max_seconds",
            allow_zero=True,
        )
        if reconnect_max_seconds < reconnect_seconds:
            raise ConfigurationError(
                f"{section}.reconnect_max_seconds cannot be below reconnect_seconds"
            )
        reconnect_jitter_ratio = _positive_number(
            data.get("reconnect_jitter_ratio", 0.20),
            f"{section}.reconnect_jitter_ratio",
            allow_zero=True,
        )
        if reconnect_jitter_ratio > 1:
            raise ConfigurationError(
                f"{section}.reconnect_jitter_ratio must be between 0 and 1"
            )
        return cls(
            reconnect_seconds=reconnect_seconds,
            reconnect_max_seconds=reconnect_max_seconds,
            reconnect_jitter_ratio=reconnect_jitter_ratio,
            open_timeout_ms=_integer(
                data.get("open_timeout_ms", 5000),
                f"{section}.open_timeout_ms",
                minimum=1,
                maximum=120_000,
            ),
            read_timeout_ms=_integer(
                data.get("read_timeout_ms", 5000),
                f"{section}.read_timeout_ms",
                minimum=1,
                maximum=120_000,
            ),
            read_wait_seconds=_positive_number(
                data.get("read_wait_seconds", 2.0),
                f"{section}.read_wait_seconds",
            ),
            rtsp_transport=str(transport),
        )


@dataclass(frozen=True, slots=True)
class QualityConfig:
    minimum_blur_variance: float = 20.0
    pixel_change_threshold: int = 35
    maximum_changed_fraction: float = 0.65
    analysis_width: int = 320

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "QualityConfig":
        _unknown_keys(
            data,
            {
                "minimum_blur_variance",
                "pixel_change_threshold",
                "maximum_changed_fraction",
                "analysis_width",
            },
            "quality",
        )
        changed = _positive_number(
            data.get("maximum_changed_fraction", 0.65),
            "quality.maximum_changed_fraction",
        )
        if changed > 1:
            raise ConfigurationError("quality.maximum_changed_fraction cannot exceed 1")
        return cls(
            minimum_blur_variance=_positive_number(
                data.get("minimum_blur_variance", 20.0),
                "quality.minimum_blur_variance",
                allow_zero=True,
            ),
            pixel_change_threshold=_integer(
                data.get("pixel_change_threshold", 35),
                "quality.pixel_change_threshold",
                minimum=0,
                maximum=255,
            ),
            maximum_changed_fraction=changed,
            analysis_width=_integer(
                data.get("analysis_width", 320),
                "quality.analysis_width",
                minimum=32,
                maximum=4096,
            ),
        )


@dataclass(frozen=True, slots=True)
class EventConfig:
    minimum_confidence: float = 0.35
    approach_distance: float = 0.18
    exit_distance: float = 0.24
    minimum_approach_seconds: float = 0.10
    minimum_release_seconds: float = 0.08
    sequence_timeout_seconds: float = 8.0
    missing_tolerance_seconds: float = 0.35

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "EventConfig":
        allowed = set(cls.__dataclass_fields__)
        _unknown_keys(data, allowed, "event")
        values = {
            name: _positive_number(
                data.get(name, field.default),
                f"event.{name}",
                allow_zero=name in {
                    "minimum_confidence",
                    "minimum_approach_seconds",
                    "minimum_release_seconds",
                    "missing_tolerance_seconds",
                },
            )
            for name, field in cls.__dataclass_fields__.items()
        }
        if values["minimum_confidence"] > 1:
            raise ConfigurationError("event.minimum_confidence cannot exceed 1")
        if values["exit_distance"] <= values["approach_distance"]:
            raise ConfigurationError("event.exit_distance must exceed approach_distance")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    name: str
    environment: str
    artifact_root: Path
    log_level: str
    health_host: str
    health_port: int
    shutdown_timeout_seconds: float
    capture: CaptureConfig = field(default_factory=CaptureConfig)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    enabled: bool
    mode: str
    source: str | int | None
    source_env: str | None
    credential_key: str | None
    analysis_fps: float
    tub_zone: tuple[tuple[float, float], ...] | None = None
    serving_zone: tuple[tuple[float, float], ...] | None = None
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    event: EventConfig = field(default_factory=EventConfig)

    def resolve_source(
        self,
        environ: Mapping[str, str] | None = None,
        credential_resolver: Callable[[str], str] | None = None,
    ) -> str | int:
        if self.source is not None:
            return self.source
        if self.credential_key is not None:
            if credential_resolver is None:
                raise ConfigurationError(
                    f"A credential resolver is required for key {self.credential_key!r}"
                )
            resolved = credential_resolver(self.credential_key).strip()
            if not resolved:
                raise ConfigurationError(
                    f"Credential {self.credential_key!r} is empty for camera {self.camera_id!r}"
                )
            return int(resolved) if resolved.isdigit() else resolved
        values = os.environ if environ is None else environ
        assert self.source_env is not None
        resolved = values.get(self.source_env, "").strip()
        if not resolved:
            raise ConfigurationError(
                f"Environment variable {self.source_env!r} is required for camera {self.camera_id!r}"
            )
        return int(resolved) if resolved.isdigit() else resolved


def _read_toml(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file not found: {resolved}")
    try:
        with resolved.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"Invalid TOML in {resolved}: {exc}") from exc


def _normalized_polygon(value: object, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigurationError(f"{name} must contain at least three [x,y] points")
    output: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ConfigurationError(f"{name} points must be [x,y] pairs")
        x = _positive_number(point[0], f"{name}.x", allow_zero=True)
        y = _positive_number(point[1], f"{name}.y", allow_zero=True)
        if x > 1 or y > 1:
            raise ConfigurationError(f"{name} points must be normalized between 0 and 1")
        output.append((x, y))
    return tuple(output)


def load_service_config(path: str | Path) -> ServiceConfig:
    data = _read_toml(path)
    _unknown_keys(data, {"service", "capture"}, "document")
    service = _required_table(data, "service")
    _unknown_keys(
        service,
        {
            "name",
            "environment",
            "artifact_root",
            "log_level",
            "health_host",
            "health_port",
            "shutdown_timeout_seconds",
        },
        "service",
    )
    name = str(service.get("name", "")).strip()
    if not name:
        raise ConfigurationError("service.name cannot be empty")
    environment = str(service.get("environment", "production")).strip().lower()
    if environment not in {"development", "test", "production"}:
        raise ConfigurationError("service.environment must be development, test, or production")
    log_level = str(service.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("service.log_level is invalid")
    artifact_text = str(service.get("artifact_root", "")).strip()
    if not artifact_text:
        raise ConfigurationError("service.artifact_root cannot be empty")
    capture_data = data.get("capture", {})
    if not isinstance(capture_data, dict):
        raise ConfigurationError("[capture] must be a table")
    return ServiceConfig(
        name=name,
        environment=environment,
        artifact_root=Path(artifact_text),
        log_level=log_level,
        health_host=str(service.get("health_host", "127.0.0.1")),
        health_port=_integer(
            service.get("health_port", 8080),
            "service.health_port",
            minimum=1,
            maximum=65_535,
        ),
        shutdown_timeout_seconds=_positive_number(
            service.get("shutdown_timeout_seconds", 10.0),
            "service.shutdown_timeout_seconds",
        ),
        capture=CaptureConfig.from_mapping(capture_data),
    )


def load_camera_config(path: str | Path) -> CameraConfig:
    data = _read_toml(path)
    _unknown_keys(data, {"camera", "capture", "zones", "quality", "event"}, "document")
    camera = _required_table(data, "camera")
    _unknown_keys(
        camera,
        {
            "camera_id",
            "enabled",
            "mode",
            "source",
            "source_env",
            "credential_key",
            "analysis_fps",
        },
        "camera",
    )
    camera_id = str(camera.get("camera_id", "")).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", camera_id):
        raise ConfigurationError(
            "camera.camera_id must be 3-64 lowercase letters, digits, '_' or '-'"
        )
    enabled = camera.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigurationError("camera.enabled must be true or false")
    mode = str(camera.get("mode", "live")).lower()
    if mode not in {"live", "recorded"}:
        raise ConfigurationError("camera.mode must be 'live' or 'recorded'")
    source = camera.get("source")
    source_env = camera.get("source_env")
    credential_key = camera.get("credential_key")
    source_options = sum(
        item is not None for item in (source, source_env, credential_key)
    )
    if source_options != 1:
        raise ConfigurationError(
            "exactly one of camera.source, camera.credential_key, or camera.source_env is required"
        )
    if source is not None:
        if isinstance(source, bool) or not isinstance(source, (str, int)):
            raise ConfigurationError("camera.source must be a string or webcam integer")
        if isinstance(source, str) and not source.strip():
            raise ConfigurationError("camera.source cannot be empty")
    if source_env is not None:
        if not isinstance(source_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", source_env):
            raise ConfigurationError("camera.source_env must be an uppercase environment variable name")
    if credential_key is not None:
        if not isinstance(credential_key, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}", credential_key
        ):
            raise ConfigurationError(
                "camera.credential_key must be a 3-128 character credential reference"
            )
    capture_data = data.get("capture", {})
    if not isinstance(capture_data, dict):
        raise ConfigurationError("[capture] must be a table")
    zones_data = data.get("zones", {})
    if not isinstance(zones_data, dict):
        raise ConfigurationError("[zones] must be a table")
    _unknown_keys(zones_data, {"tub", "serving"}, "zones")
    tub_zone = (
        _normalized_polygon(zones_data["tub"], "zones.tub")
        if "tub" in zones_data
        else None
    )
    serving_zone = (
        _normalized_polygon(zones_data["serving"], "zones.serving")
        if "serving" in zones_data
        else None
    )
    quality_data = data.get("quality", {})
    event_data = data.get("event", {})
    if not isinstance(quality_data, dict) or not isinstance(event_data, dict):
        raise ConfigurationError("[quality] and [event] must be tables")
    return CameraConfig(
        camera_id=camera_id,
        enabled=enabled,
        mode=mode,
        source=source,
        source_env=source_env,
        credential_key=credential_key,
        analysis_fps=_positive_number(
            camera.get("analysis_fps", 10.0),
            "camera.analysis_fps",
        ),
        tub_zone=tub_zone,
        serving_zone=serving_zone,
        capture=CaptureConfig.from_mapping(capture_data),
        quality=QualityConfig.from_mapping(quality_data),
        event=EventConfig.from_mapping(event_data),
    )
