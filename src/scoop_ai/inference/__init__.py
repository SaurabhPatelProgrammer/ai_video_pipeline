"""Model-independent inference interfaces and checkpoint contracts."""

from .checkpoint_manifest import (
    CANONICAL_CLASSES,
    CheckpointManifest,
    ManifestValidationError,
    create_checkpoint_manifest,
    load_checkpoint_manifest,
)
from ..compute import (
    CPU_ANALYSIS_FPS_CEILING,
    DEVICE_PREFERENCES,
    ComputeCapability,
    ComputeError,
    clamp_analysis_fps,
    configure_torch_threads,
    describe_compute,
    normalise_preference,
    resolve_device,
)
from .interfaces import (
    Detection,
    DetectorAdapter,
    TrackerAdapter,
    observations_from_detections,
)
from .rfdetr_adapter import RFDETRLocalAdapter
from .proximity_tracker import ProximityTrackerAdapter
from .supervision_tracker import SupervisionByteTrackAdapter
from .governance import (
    approval_payload,
    expected_reviewer_signature,
    validate_model_camera_compatibility,
    verify_manifest_approval,
)

__all__ = [
    "CANONICAL_CLASSES",
    "CPU_ANALYSIS_FPS_CEILING",
    "DEVICE_PREFERENCES",
    "CheckpointManifest",
    "ComputeCapability",
    "ComputeError",
    "Detection",
    "clamp_analysis_fps",
    "configure_torch_threads",
    "describe_compute",
    "normalise_preference",
    "resolve_device",
    "DetectorAdapter",
    "ManifestValidationError",
    "create_checkpoint_manifest",
    "RFDETRLocalAdapter",
    "ProximityTrackerAdapter",
    "SupervisionByteTrackAdapter",
    "TrackerAdapter",
    "load_checkpoint_manifest",
    "observations_from_detections",
    "approval_payload",
    "expected_reviewer_signature",
    "validate_model_camera_compatibility",
    "verify_manifest_approval",
]
