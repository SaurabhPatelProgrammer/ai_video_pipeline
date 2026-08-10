"""Lazy RF-DETR adapter for a checksum-verified local checkpoint bundle."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Sequence

from .checkpoint_manifest import CheckpointManifest, load_checkpoint_manifest
from .checkpoint_manifest import CANONICAL_CLASSES
from .interfaces import Detection


class RFDETRLocalAdapter:
    """Run RF-DETR without downloading weights or coupling it to tracking.

    Construction validates the manifest and local checkpoint SHA-256, but imports
    PyTorch/RF-DETR and allocates the model only on the first ``predict`` call.
    The input frame must be an RGB NumPy-like image accepted by RF-DETR. Returned
    detections use pixel ``xyxy`` boxes and have ``track_id=None``; pass them
    through a ``TrackerAdapter`` before ``observations_from_detections``.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        device: str | None = None,
        confidence_threshold: float | None = None,
        expected_architecture: str | None = None,
        expected_classes: tuple[str, ...] | None = CANONICAL_CLASSES,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest: CheckpointManifest = load_checkpoint_manifest(
            self.manifest_path,
            expected_architecture=expected_architecture,
            expected_classes=expected_classes,
            verify_checkpoint=True,
        )
        if confidence_threshold is not None and not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.device = device
        self.confidence_threshold = (
            self.manifest.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        # Heavy libraries are intentionally lazy so validation, APIs, and test
        # tooling remain usable on machines without a GPU runtime.
        import torch
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        checkpoint_path = self.manifest_path.parent / self.manifest.checkpoint_file
        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model_classes[self.manifest.architecture](
            device=selected_device,
            pretrain_weights=str(checkpoint_path),
            resolution=self.manifest.input_resolution,
        )
        return self._model

    def predict(self, frame: object, timestamp: float) -> Sequence[Detection]:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        model = self._load_model()
        result = model.predict(
            frame,
            threshold=self.confidence_threshold,
            include_source_image=False,
        )
        xyxy = getattr(result, "xyxy", ())
        confidences = getattr(result, "confidence", ())
        class_ids = getattr(result, "class_id", ())
        attached_names = getattr(result, "data", {}).get("class_name")
        output: list[Detection] = []
        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy, confidences, class_ids)
        ):
            class_name = self._class_name(model, attached_names, index, int(class_id))
            output.append(
                Detection(
                    class_name=class_name,
                    confidence=float(confidence),
                    xyxy=tuple(float(value) for value in box),
                )
            )
        return output

    def _class_name(
        self,
        model: Any,
        attached_names: Any,
        index: int,
        class_id: int,
    ) -> str:
        if attached_names is not None and index < len(attached_names):
            attached = str(attached_names[index]).strip()
            if attached:
                return attached
        class_names = getattr(model, "class_names", self.manifest.classes)
        if isinstance(class_names, dict):
            value = class_names.get(class_id)
            if value is not None:
                return str(value)
        elif 0 <= class_id < len(class_names):
            return str(class_names[class_id])
        raise ValueError(f"detector returned unknown class_id {class_id}")
