"""Model-independent inference interfaces and checkpoint contracts."""

from .checkpoint_manifest import (
    CANONICAL_CLASSES,
    CheckpointManifest,
    ManifestValidationError,
    create_checkpoint_manifest,
    load_checkpoint_manifest,
)
from .interfaces import (
    Detection,
    DetectorAdapter,
    TrackerAdapter,
    observations_from_detections,
)
from .rfdetr_adapter import RFDETRLocalAdapter
from .supervision_tracker import SupervisionByteTrackAdapter

__all__ = [
    "CANONICAL_CLASSES",
    "CheckpointManifest",
    "Detection",
    "DetectorAdapter",
    "ManifestValidationError",
    "create_checkpoint_manifest",
    "RFDETRLocalAdapter",
    "SupervisionByteTrackAdapter",
    "TrackerAdapter",
    "load_checkpoint_manifest",
    "observations_from_detections",
]
