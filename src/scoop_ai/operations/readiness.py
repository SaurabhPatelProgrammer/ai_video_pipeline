"""Evidence-backed final production-gate readiness reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import load_camera_config, load_service_config
from ..storage import SQLiteEventRepository


def _check(name: str, passed: bool, detail: str, *, blocking: bool = True) -> dict[str, object]:
    return {"name": name, "status": "pass" if passed else "blocked", "detail": detail, "blocking": blocking}


def generate_readiness_report(
    *,
    service_config_path: str | Path,
    camera_config_path: str | Path,
    database_path: str | Path,
    dataset_path: str | Path | None = None,
    backup_path: str | Path | None = None,
    downstream_approved: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    service = load_service_config(service_config_path)
    camera = load_camera_config(camera_config_path)
    checks: list[dict[str, object]] = [
        _check("billing_disabled", service.billing_mode == "disabled", "billing_mode must be disabled"),
        _check("automatic_exports_disabled", not service.automatic_exports, "automatic_exports must be false"),
        _check("camera_zones_approved", camera.tub_zone is not None and camera.serving_zone is not None, "tub and serving zones are configured"),
        _check("calibration_profile_present", camera.calibration_profile is not None and camera.calibration_profile.is_file(), "versioned calibration profile is present"),
        _check("representative_dataset_present", dataset_path is not None and Path(dataset_path).exists(), "representative production-camera dataset is available"),
        _check("backup_drill_evidence", backup_path is not None and Path(backup_path).is_file(), "verified backup artifact is supplied"),
    ]
    with SQLiteEventRepository(database_path) as repository:
        integrity = repository.integrity_check()
        active = repository.active_model(camera.camera_id)
        dead_letters = len(repository.list_outbox(states={"dead_letter"}))
        checks.extend([
            _check("database_integrity", integrity, "SQLite integrity check"),
            _check("active_approved_model", active is not None, "active model is registered for the camera"),
            _check("outbox_dead_letters", dead_letters == 0, f"dead-letter exports: {dead_letters}"),
        ])
    checks.append(_check(
        "downstream_separate_approval",
        downstream_approved or not service.export.enabled,
        "downstream export is disabled or separately approved",
    ))
    blocking_failures = [item["name"] for item in checks if item["blocking"] and item["status"] != "pass"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "product_label": "production-oriented review and silent-pilot edge analytics",
        "ready": not blocking_failures,
        "blocking_failures": blocking_failures,
        "checks": checks,
        "active_model": active,
        "camera_id": camera.camera_id,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return report
