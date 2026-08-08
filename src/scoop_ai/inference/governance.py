"""Approval and camera compatibility checks for model deployment."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Iterable

from ..config import CameraConfig
from .checkpoint_manifest import CheckpointManifest, ManifestValidationError, load_checkpoint_manifest


def approval_payload(manifest: CheckpointManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "model_family": manifest.model_family,
        "architecture": manifest.architecture,
        "classes": list(manifest.classes),
        "checkpoint_file": manifest.checkpoint_file,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "dataset_version": manifest.dataset_version,
        "input_resolution": manifest.input_resolution,
        "confidence_threshold": manifest.confidence_threshold,
        "model_version": manifest.model_version,
        "approved_by": manifest.approved_by,
    }


def expected_reviewer_signature(manifest: CheckpointManifest) -> str:
    canonical = json.dumps(approval_payload(manifest), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_manifest_approval(
    manifest_path: str | Path,
    *,
    approved_reviewers: Iterable[str] = (),
    require_approval: bool = True,
) -> CheckpointManifest:
    manifest = load_checkpoint_manifest(Path(manifest_path), verify_checkpoint=True)
    if not require_approval:
        return manifest
    if not manifest.approved_by or not manifest.reviewer_signature:
        raise ManifestValidationError("checkpoint manifest is missing approval fields")
    reviewers = {item.strip() for item in approved_reviewers if item.strip()}
    if manifest.approved_by not in reviewers:
        raise ManifestValidationError(
            f"manifest reviewer {manifest.approved_by!r} is not in the approved reviewers list"
        )
    if not hmac.compare_digest(manifest.reviewer_signature, expected_reviewer_signature(manifest)):
        raise ManifestValidationError("checkpoint reviewer signature is invalid")
    return manifest


def validate_model_camera_compatibility(
    manifest: CheckpointManifest,
    camera: CameraConfig,
) -> None:
    expected = {"scoop", "loaded_scoop", "serving_container"}
    if set(manifest.classes) != expected:
        raise ManifestValidationError("model classes are incompatible with the deposit state machine")
    if camera.tub_zone is None or camera.serving_zone is None:
        raise ManifestValidationError("camera zones are required for model compatibility validation")
    if camera.expected_width is not None and camera.expected_width < manifest.input_resolution:
        raise ManifestValidationError("camera width is smaller than the model input resolution")
    if camera.expected_height is not None and camera.expected_height < manifest.input_resolution:
        raise ManifestValidationError("camera height is smaller than the model input resolution")
