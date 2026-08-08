"""Validation for local, versioned model checkpoint bundles."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


CANONICAL_CLASSES = ("scoop", "loaded_scoop", "serving_container")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_SCHEMA_VERSIONS = {1}
SUPPORTED_ARCHITECTURES = {"nano", "small", "medium", "large"}


class ManifestValidationError(ValueError):
    """Raised when a checkpoint bundle is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    model_family: str
    architecture: str
    classes: tuple[str, ...]
    checkpoint_file: str
    checkpoint_sha256: str
    dataset_version: str
    input_resolution: int
    confidence_threshold: float
    model_version: str
    approved_by: str | None = None
    reviewer_signature: str | None = None

    @property
    def canonical_classes(self) -> tuple[str, ...]:
        return tuple(sorted(self.classes))


def create_checkpoint_manifest(
    checkpoint_path: Path,
    output_path: Path,
    *,
    architecture: str,
    dataset_version: str,
    model_version: str,
    input_resolution: int = 576,
    confidence_threshold: float = 0.35,
    approved_by: str | None = None,
    reviewer_signature: str | None = None,
    classes: tuple[str, ...] = CANONICAL_CLASSES,
) -> CheckpointManifest:
    """Create and immediately verify a manifest beside a trusted checkpoint."""

    checkpoint = checkpoint_path.resolve(strict=True)
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_checkpoint = checkpoint.relative_to(output.parent)
    except ValueError as exc:
        raise ManifestValidationError(
            "checkpoint must be inside the output manifest directory"
        ) from exc
    payload = {
        "schema_version": 1,
        "model_family": "rfdetr",
        "architecture": architecture,
        "classes": list(classes),
        "checkpoint_file": relative_checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_version": dataset_version,
        "input_resolution": input_resolution,
        "confidence_threshold": confidence_threshold,
        "model_version": model_version,
        "approved_by": approved_by,
        "reviewer_signature": reviewer_signature,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return load_checkpoint_manifest(
        output,
        verify_checkpoint=True,
        expected_classes=classes,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_checkpoint_path(bundle_directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("checkpoint_file must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestValidationError("checkpoint_file must stay inside the manifest directory")
    root = bundle_directory.resolve()
    checkpoint = (root / relative).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError("checkpoint_file escapes the manifest directory") from exc
    return checkpoint


def load_checkpoint_manifest(
    manifest_path: Path,
    *,
    expected_architecture: str | None = None,
    expected_classes: tuple[str, ...] | None = CANONICAL_CLASSES,
    verify_checkpoint: bool = True,
) -> CheckpointManifest:
    """Load a manifest and optionally verify its local checkpoint checksum."""

    if not manifest_path.is_file():
        raise ManifestValidationError(f"manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"could not read manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestValidationError("manifest root must be a JSON object")

    if isinstance(data.get("schema_version"), bool) or isinstance(
        data.get("input_resolution"), bool
    ):
        raise ManifestValidationError("schema_version and input_resolution must be integers")

    try:
        schema_version = int(data["schema_version"])
        model_family = str(data["model_family"]).strip().lower()
        architecture = str(data["architecture"]).strip().lower()
        classes = tuple(str(value).strip() for value in data["classes"])
        checkpoint_file = str(data["checkpoint_file"])
        checkpoint_sha256 = str(data["checkpoint_sha256"]).strip().lower()
        dataset_version = str(data["dataset_version"]).strip()
        input_resolution = int(data["input_resolution"])
        confidence_threshold = float(data["confidence_threshold"])
        model_version = str(data["model_version"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestValidationError(f"missing or invalid manifest field: {exc}") from exc

    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestValidationError(f"unsupported schema_version: {schema_version}")
    if model_family != "rfdetr":
        raise ManifestValidationError("model_family must be 'rfdetr'")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ManifestValidationError(f"unsupported architecture: {architecture}")
    if expected_architecture is not None and architecture != expected_architecture.lower():
        raise ManifestValidationError(
            f"checkpoint architecture {architecture!r} does not match {expected_architecture!r}"
        )
    if not classes or len(classes) != len(set(classes)):
        raise ManifestValidationError("classes must contain unique class names")
    invalid_classes = [name for name in classes if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)]
    if invalid_classes:
        raise ManifestValidationError(
            f"classes must use lowercase letters, digits, and underscores: {invalid_classes!r}"
        )
    if expected_classes is not None and set(classes) != set(expected_classes):
        raise ManifestValidationError(
            f"classes must be exactly {list(expected_classes)!r} without duplicates"
        )
    if not SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise ManifestValidationError("checkpoint_sha256 must be 64 lowercase hex characters")
    if not dataset_version or not model_version:
        raise ManifestValidationError("dataset_version and model_version cannot be empty")
    if input_resolution < 128 or input_resolution > 4096:
        raise ManifestValidationError("input_resolution must be between 128 and 4096")
    if not isfinite(confidence_threshold) or not 0.0 < confidence_threshold < 1.0:
        raise ManifestValidationError("confidence_threshold must be between 0 and 1")

    checkpoint_path = _safe_checkpoint_path(manifest_path.parent, checkpoint_file)
    if verify_checkpoint:
        if not checkpoint_path.is_file():
            raise ManifestValidationError(f"checkpoint not found: {checkpoint_path}")
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != checkpoint_sha256:
            raise ManifestValidationError(
                f"checkpoint checksum mismatch: expected {checkpoint_sha256}, received {actual_hash}"
            )

    return CheckpointManifest(
        schema_version=schema_version,
        model_family=model_family,
        architecture=architecture,
        classes=classes,
        checkpoint_file=checkpoint_file,
        checkpoint_sha256=checkpoint_sha256,
        dataset_version=dataset_version,
        input_resolution=input_resolution,
        confidence_threshold=confidence_threshold,
        model_version=model_version,
        approved_by=(str(data["approved_by"]).strip() if data.get("approved_by") else None),
        reviewer_signature=(str(data["reviewer_signature"]).strip() if data.get("reviewer_signature") else None),
    )
