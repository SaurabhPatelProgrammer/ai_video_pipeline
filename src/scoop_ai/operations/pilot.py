"""Controlled silent-pilot reporting and safety assertions."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ServiceConfig
from ..storage import SQLiteEventRepository


def assert_silent_pilot(config: ServiceConfig) -> None:
    if config.billing_mode != "disabled":
        raise ValueError("silent pilot requires billing_mode='disabled'")
    if config.automatic_exports:
        raise ValueError("silent pilot requires automatic_exports=false")


def generate_pilot_report(
    database_path: str | Path,
    *,
    camera_id: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    with SQLiteEventRepository(database_path) as repository:
        report = repository.pilot_report(camera_id=camera_id)
    report["schema_version"] = 1
    report["silent_pilot"] = True
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return report
