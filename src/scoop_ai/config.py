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
    reference_drift_threshold: float = 0.20
    zone_change_threshold: float = 0.45
    obstruction_seconds: float = 1.0
    minimum_fps_ratio: float = 0.50

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "QualityConfig":
        _unknown_keys(
            data,
            {
                "minimum_blur_variance",
                "pixel_change_threshold",
                "maximum_changed_fraction",
                "analysis_width",
                "reference_drift_threshold",
                "zone_change_threshold",
                "obstruction_seconds",
                "minimum_fps_ratio",
            },
            "quality",
        )
        changed = _positive_number(
            data.get("maximum_changed_fraction", 0.65),
            "quality.maximum_changed_fraction",
        )
        if changed > 1:
            raise ConfigurationError("quality.maximum_changed_fraction cannot exceed 1")
        drift = _positive_number(data.get("reference_drift_threshold", 0.20), "quality.reference_drift_threshold")
        zone_change = _positive_number(data.get("zone_change_threshold", 0.45), "quality.zone_change_threshold")
        fps_ratio = _positive_number(data.get("minimum_fps_ratio", 0.50), "quality.minimum_fps_ratio")
        if drift > 1 or zone_change > 1 or fps_ratio > 1:
            raise ConfigurationError("quality drift, obstruction and FPS ratios cannot exceed 1")
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
            reference_drift_threshold=drift,
            zone_change_threshold=zone_change,
            obstruction_seconds=_positive_number(
                data.get("obstruction_seconds", 1.0),
                "quality.obstruction_seconds",
                allow_zero=True,
            ),
            minimum_fps_ratio=fps_ratio,
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
class HandoverConfig:
    """Tracking and temporal thresholds for one-class item handovers."""

    minimum_confidence: float | None = None
    maximum_center_distance_pixels: float = 120.0
    lost_track_seconds: float = 1.0
    minimum_consecutive_frames: int = 1
    duplicate_iou_threshold: float = 0.50
    minimum_transfer_seconds: float = 0.20
    minimum_customer_dwell_seconds: float = 0.12
    minimum_customer_observations: int = 2
    minimum_movement_distance: float = 0.03
    sequence_timeout_seconds: float = 5.0
    missing_tolerance_seconds: float = 1.0
    duplicate_cooldown_seconds: float = 3.5
    duplicate_distance: float = 0.12

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HandoverConfig":
        _unknown_keys(data, set(cls.__dataclass_fields__), "handover")
        confidence_value = data.get("minimum_confidence")
        confidence = None
        if confidence_value is not None:
            confidence = _positive_number(
                confidence_value, "handover.minimum_confidence", allow_zero=True
            )
            if confidence > 1:
                raise ConfigurationError("handover.minimum_confidence cannot exceed 1")
        duplicate_iou = _positive_number(
            data.get("duplicate_iou_threshold", 0.50),
            "handover.duplicate_iou_threshold",
        )
        if duplicate_iou > 1:
            raise ConfigurationError("handover.duplicate_iou_threshold cannot exceed 1")
        return cls(
            minimum_confidence=confidence,
            maximum_center_distance_pixels=_positive_number(
                data.get("maximum_center_distance_pixels", 120.0),
                "handover.maximum_center_distance_pixels",
            ),
            lost_track_seconds=_positive_number(
                data.get("lost_track_seconds", 1.0),
                "handover.lost_track_seconds",
            ),
            minimum_consecutive_frames=_integer(
                data.get("minimum_consecutive_frames", 1),
                "handover.minimum_consecutive_frames",
                minimum=1,
                maximum=100,
            ),
            duplicate_iou_threshold=duplicate_iou,
            minimum_transfer_seconds=_positive_number(
                data.get("minimum_transfer_seconds", 0.20),
                "handover.minimum_transfer_seconds",
                allow_zero=True,
            ),
            minimum_customer_dwell_seconds=_positive_number(
                data.get("minimum_customer_dwell_seconds", 0.12),
                "handover.minimum_customer_dwell_seconds",
                allow_zero=True,
            ),
            minimum_customer_observations=_integer(
                data.get("minimum_customer_observations", 2),
                "handover.minimum_customer_observations",
                minimum=1,
                maximum=100,
            ),
            minimum_movement_distance=_positive_number(
                data.get("minimum_movement_distance", 0.03),
                "handover.minimum_movement_distance",
                allow_zero=True,
            ),
            sequence_timeout_seconds=_positive_number(
                data.get("sequence_timeout_seconds", 5.0),
                "handover.sequence_timeout_seconds",
            ),
            missing_tolerance_seconds=_positive_number(
                data.get("missing_tolerance_seconds", 1.0),
                "handover.missing_tolerance_seconds",
                allow_zero=True,
            ),
            duplicate_cooldown_seconds=_positive_number(
                data.get("duplicate_cooldown_seconds", 3.5),
                "handover.duplicate_cooldown_seconds",
                allow_zero=True,
            ),
            duplicate_distance=_positive_number(
                data.get("duplicate_distance", 0.12),
                "handover.duplicate_distance",
                allow_zero=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    name: str
    environment: str
    artifact_root: Path
    log_level: str
    health_host: str
    health_port: int
    shutdown_timeout_seconds: float
    minimum_free_space_gb: float = 10.0
    approved_reviewers: tuple[str, ...] = ()
    billing_mode: str = "disabled"
    automatic_exports: bool = False
    export: "ExportConfig" = field(default_factory=lambda: ExportConfig())
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    alerts: "AlertConfig" = field(default_factory=lambda: AlertConfig())


@dataclass(frozen=True, slots=True)
class AlertConfig:
    poll_seconds: float = 5.0
    stale_after_seconds: float = 10.0
    reconnect_warning: int = 3
    reconnect_critical: int = 10
    low_disk_warning_gb: float = 10.0
    low_disk_critical_gb: float = 2.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AlertConfig":
        _unknown_keys(
            data,
            {
                "poll_seconds",
                "stale_after_seconds",
                "reconnect_warning",
                "reconnect_critical",
                "low_disk_warning_gb",
                "low_disk_critical_gb",
            },
            "alerts",
        )
        warning = _integer(data.get("reconnect_warning", 3), "alerts.reconnect_warning", minimum=1, maximum=1_000_000)
        critical = _integer(data.get("reconnect_critical", 10), "alerts.reconnect_critical", minimum=warning, maximum=1_000_000)
        warning_disk = _positive_number(data.get("low_disk_warning_gb", 10.0), "alerts.low_disk_warning_gb", allow_zero=True)
        critical_disk = _positive_number(data.get("low_disk_critical_gb", 2.0), "alerts.low_disk_critical_gb", allow_zero=True)
        if critical_disk > warning_disk:
            raise ConfigurationError("alerts.low_disk_critical_gb cannot exceed warning threshold")
        return cls(
            poll_seconds=_positive_number(data.get("poll_seconds", 5.0), "alerts.poll_seconds"),
            stale_after_seconds=_positive_number(data.get("stale_after_seconds", 10.0), "alerts.stale_after_seconds"),
            reconnect_warning=warning,
            reconnect_critical=critical,
            low_disk_warning_gb=warning_disk,
            low_disk_critical_gb=critical_disk,
        )


@dataclass(frozen=True, slots=True)
class ExportConfig:
    enabled: bool = False
    endpoint: str | None = None
    poll_seconds: float = 2.0
    max_attempts: int = 5
    backoff_seconds: float = 2.0
    signing_key_credential: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ExportConfig":
        _unknown_keys(data, {
            "enabled", "endpoint", "poll_seconds", "max_attempts",
            "backoff_seconds", "signing_key_credential",
        }, "export")
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigurationError("export.enabled must be true or false")
        endpoint = data.get("endpoint")
        if enabled and (not isinstance(endpoint, str) or not endpoint.strip()):
            raise ConfigurationError("enabled export requires export.endpoint")
        credential = data.get("signing_key_credential")
        if enabled and (not isinstance(credential, str) or not credential.strip()):
            raise ConfigurationError("enabled export requires export.signing_key_credential")
        attempts = _integer(data.get("max_attempts", 5), "export.max_attempts", minimum=1, maximum=100)
        return cls(
            enabled=enabled,
            endpoint=str(endpoint).strip() if endpoint else None,
            poll_seconds=_positive_number(data.get("poll_seconds", 2.0), "export.poll_seconds"),
            max_attempts=attempts,
            backoff_seconds=_positive_number(data.get("backoff_seconds", 2.0), "export.backoff_seconds"),
            signing_key_credential=str(credential).strip() if credential else None,
        )


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    enabled: bool
    mode: str
    source: str | int | None
    source_env: str | None
    credential_key: str | None
    analysis_fps: float
    pipeline: str = "auto"
    calibration_profile: Path | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    tub_zone: tuple[tuple[float, float], ...] | None = None
    serving_zone: tuple[tuple[float, float], ...] | None = None
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    event: EventConfig = field(default_factory=EventConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)

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
    _unknown_keys(data, {"service", "capture", "alerts", "export"}, "document")
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
            "minimum_free_space_gb",
            "approved_reviewers",
            "billing_mode",
            "automatic_exports",
        },
        "service",
    )
    name = str(service.get("name", "")).strip()
    if not name:
        raise ConfigurationError("service.name cannot be empty")
    environment = str(service.get("environment", "production")).strip().lower()
    if environment not in {"development", "test", "production"}:
        raise ConfigurationError("service.environment must be development, test, or production")
    billing_mode = str(service.get("billing_mode", "disabled")).strip().lower()
    if billing_mode != "disabled":
        raise ConfigurationError("service.billing_mode must be 'disabled' during the silent pilot")
    automatic_exports = service.get("automatic_exports", False)
    if not isinstance(automatic_exports, bool) or automatic_exports:
        raise ConfigurationError("service.automatic_exports must be false during the silent pilot")
    log_level = str(service.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("service.log_level is invalid")
    artifact_text = str(service.get("artifact_root", "")).strip()
    if not artifact_text:
        raise ConfigurationError("service.artifact_root cannot be empty")
    capture_data = data.get("capture", {})
    if not isinstance(capture_data, dict):
        raise ConfigurationError("[capture] must be a table")
    alerts_data = data.get("alerts", {})
    if not isinstance(alerts_data, dict):
        raise ConfigurationError("[alerts] must be a table")
    export_data = data.get("export", {})
    if not isinstance(export_data, dict):
        raise ConfigurationError("[export] must be a table")
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
        minimum_free_space_gb=_positive_number(
            service.get("minimum_free_space_gb", 10.0),
            "service.minimum_free_space_gb",
            allow_zero=True,
        ),
        approved_reviewers=tuple(
            str(item).strip() for item in service.get("approved_reviewers", [])
            if str(item).strip()
        ),
        billing_mode=billing_mode,
        automatic_exports=automatic_exports,
        export=ExportConfig.from_mapping(export_data),
        capture=CaptureConfig.from_mapping(capture_data),
        alerts=AlertConfig.from_mapping(alerts_data),
    )


def load_camera_config(path: str | Path) -> CameraConfig:
    data = _read_toml(path)
    _unknown_keys(
        data,
        {"camera", "capture", "zones", "quality", "event", "handover"},
        "document",
    )
    camera = _required_table(data, "camera")
    _unknown_keys(
        camera,
        {
            "camera_id",
            "enabled",
            "mode",
            "pipeline",
            "source",
            "source_env",
            "credential_key",
            "analysis_fps",
            "calibration_profile",
            "expected_width",
            "expected_height",
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
    pipeline = str(camera.get("pipeline", "auto")).strip().lower()
    if pipeline not in {"auto", "deposit", "handover"}:
        raise ConfigurationError("camera.pipeline must be 'auto', 'deposit', or 'handover'")
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
    calibration_profile = camera.get("calibration_profile")
    if calibration_profile is not None and (not isinstance(calibration_profile, str) or not calibration_profile.strip()):
        raise ConfigurationError("camera.calibration_profile must be a non-empty path")
    expected_width = camera.get("expected_width")
    expected_height = camera.get("expected_height")
    if expected_width is not None:
        expected_width = _integer(expected_width, "camera.expected_width", minimum=1, maximum=16_384)
    if expected_height is not None:
        expected_height = _integer(expected_height, "camera.expected_height", minimum=1, maximum=16_384)
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
    handover_data = data.get("handover", {})
    if not all(isinstance(value, dict) for value in (quality_data, event_data, handover_data)):
        raise ConfigurationError("[quality], [event], and [handover] must be tables")
    return CameraConfig(
        camera_id=camera_id,
        enabled=enabled,
        mode=mode,
        pipeline=pipeline,
        source=source,
        source_env=source_env,
        credential_key=credential_key,
        analysis_fps=_positive_number(
            camera.get("analysis_fps", 10.0),
            "camera.analysis_fps",
        ),
        calibration_profile=Path(calibration_profile) if calibration_profile is not None else None,
        expected_width=expected_width,
        expected_height=expected_height,
        tub_zone=tub_zone,
        serving_zone=serving_zone,
        capture=CaptureConfig.from_mapping(capture_data),
        quality=QualityConfig.from_mapping(quality_data),
        event=EventConfig.from_mapping(event_data),
        handover=HandoverConfig.from_mapping(handover_data),
    )
