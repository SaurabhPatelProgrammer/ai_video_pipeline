"""Production package for deterministic scoop-event processing."""

from .config import (
    CameraConfig,
    CaptureConfig,
    ConfigurationError,
    ServiceConfig,
    load_camera_config,
    load_service_config,
)

__all__ = [
    "CameraConfig",
    "CaptureConfig",
    "ConfigurationError",
    "ServiceConfig",
    "load_camera_config",
    "load_service_config",
]

__version__ = "0.1.0"
