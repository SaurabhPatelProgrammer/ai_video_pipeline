"""Create immutable review-only profiles for the served-order state machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .motion_baseline import _atomic_json, _load_json, _sha256


@dataclass(frozen=True)
class ServedOrderProfileResult:
    profile_path: Path
    manifest_path: Path
    profile_sha256: str


def create_served_order_profile(
    base_profile_path: str | Path,
    output_path: str | Path,
    *,
    tub_threshold: float,
    customer_threshold: float,
    minimum_preparation_seconds: float,
    order_timeout_seconds: float,
    minimum_container_frames: int,
    minimum_customer_frames: int,
    cooldown_seconds: float,
    labeled_events: int,
) -> ServedOrderProfileResult:
    """Write a served-order profile without mutating its calibrated parent."""

    if tub_threshold <= 0 or customer_threshold <= 0:
        raise ValueError("motion thresholds must be positive")
    if minimum_preparation_seconds < 0 or order_timeout_seconds <= minimum_preparation_seconds:
        raise ValueError("order timeout must exceed minimum preparation time")
    if minimum_container_frames < 1 or minimum_customer_frames < 1:
        raise ValueError("minimum frame counts must be positive")
    if cooldown_seconds < 0 or labeled_events < 1:
        raise ValueError("cooldown must be non-negative and labeled_events positive")
    base = Path(base_profile_path).resolve()
    output = Path(output_path).resolve()
    manifest = output.with_name(f"{output.stem}.manifest.json")
    for artifact in (output, manifest):
        if artifact.exists():
            raise ValueError(f"refusing to overwrite immutable model artifact: {artifact}")
    payload = _load_json(base)
    if not isinstance(payload, dict):
        raise ValueError("base profile must be a JSON object")
    if "tub_zone" not in payload or "serving_zone" not in payload:
        raise ValueError("base profile must contain calibrated tub and serving zones")
    payload.pop("minimum_transfer_seconds", None)
    payload.pop("transfer_timeout_seconds", None)
    payload.update(
        {
            "profile_name": "served-order-two-video-v1",
            "event_mode": "served_order",
            "tub_motion_threshold": tub_threshold,
            "serving_motion_threshold": customer_threshold,
            "minimum_loading_frames": minimum_container_frames,
            "minimum_serving_frames": minimum_customer_frames,
            "minimum_preparation_seconds": minimum_preparation_seconds,
            "order_timeout_seconds": order_timeout_seconds,
            "cooldown_seconds": cooldown_seconds,
            "served_order_tuning": {
                "parent_profile_sha256": _sha256(base),
                "labeled_events": labeled_events,
                "method": "deterministic_grid_search_on_labeled_videos",
                "event_definition": (
                    "one count after ice-cream preparation starts in the container area "
                    "and the finished cup/cone is served in the customer area"
                ),
                "evaluation_scope": "in_sample_only",
                "production_approved": False,
                "limitations": [
                    "Only two positive served-order labels were available.",
                    "Motion alone cannot distinguish every customer movement from a handoff.",
                    "Use as a review candidate generator, never for billing.",
                ],
            },
        }
    )
    _atomic_json(output, payload)
    digest = _sha256(output)
    _atomic_json(
        manifest,
        {
            "artifact": output.name,
            "artifact_sha256": digest,
            "artifact_type": "level1_served_order_profile",
            "event_mode": "served_order",
            "labeled_events": labeled_events,
            "production_approved": False,
        },
    )
    return ServedOrderProfileResult(output, manifest, digest)
