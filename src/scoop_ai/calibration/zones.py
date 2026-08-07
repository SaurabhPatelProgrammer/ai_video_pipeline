"""Mouse-driven container and customer-area polygon calibration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..security import safe_source_name

NormalizedPoint = tuple[float, float]


@dataclass(frozen=True)
class CalibrationResult:
    profile_path: Path
    profile_sha256: str
    container_zone: tuple[NormalizedPoint, ...]
    customer_zone: tuple[NormalizedPoint, ...]
    frame_width: int
    frame_height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed_area(points: Sequence[NormalizedPoint]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )


def _orientation(a: NormalizedPoint, b: NormalizedPoint, c: NormalizedPoint) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    a: NormalizedPoint,
    b: NormalizedPoint,
    c: NormalizedPoint,
    d: NormalizedPoint,
) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    return first * second < 0 and third * fourth < 0


def validate_zone(
    points: Iterable[Sequence[float]],
    name: str,
) -> tuple[NormalizedPoint, ...]:
    """Return a validated simple normalized polygon."""

    normalized: list[NormalizedPoint] = []
    for raw in points:
        if len(raw) != 2:
            raise ValueError(f"{name} points must contain x and y")
        x, y = float(raw[0]), float(raw[1])
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"{name} points must be finite and normalized to [0,1]")
        point = (round(x, 6), round(y, 6))
        if not normalized or point != normalized[-1]:
            normalized.append(point)
    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()
    if len(set(normalized)) < 3:
        raise ValueError(f"{name} needs at least three unique points")
    size = len(normalized)
    for first in range(size):
        a, b = normalized[first], normalized[(first + 1) % size]
        for second in range(first + 1, size):
            if second in {first, (first + 1) % size}:
                continue
            if first == 0 and second == size - 1:
                continue
            c, d = normalized[second], normalized[(second + 1) % size]
            if _segments_cross(a, b, c, d):
                raise ValueError(f"{name} edges cross; redraw a simple polygon")
    if abs(_signed_area(normalized)) < 0.001:
        raise ValueError(f"{name} area is too small")
    return tuple(normalized)


def _overlap_fraction(
    first: Sequence[NormalizedPoint], second: Sequence[NormalizedPoint]
) -> float:
    size = 512
    masks = []
    for zone in (first, second):
        mask = np.zeros((size, size), dtype=np.uint8)
        polygon = np.array(
            [[min(size - 1, round(x * size)), min(size - 1, round(y * size))] for x, y in zone],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        masks.append(mask)
    intersection = cv2.countNonZero(cv2.bitwise_and(masks[0], masks[1]))
    smaller = min(cv2.countNonZero(masks[0]), cv2.countNonZero(masks[1]))
    return intersection / smaller if smaller else 1.0


def save_calibrated_profile(
    base_profile_path: str | Path,
    output_path: str | Path,
    *,
    container_zone: Iterable[Sequence[float]],
    customer_zone: Iterable[Sequence[float]],
    frame_width: int,
    frame_height: int,
    source_name: str,
    overwrite: bool = False,
) -> CalibrationResult:
    """Write a calibrated derivative while preserving learned thresholds."""

    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    container = validate_zone(container_zone, "container_zone")
    customer = validate_zone(customer_zone, "customer_zone")
    if _overlap_fraction(container, customer) > 0.10:
        raise ValueError("container and customer zones overlap too much")
    base = Path(base_profile_path).resolve()
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise ValueError(f"calibrated profile already exists: {output}; use --force to replace it")
    try:
        payload = json.loads(base.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read base profile {base}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("base profile must be a JSON object")
    for threshold in ("tub_motion_threshold", "serving_motion_threshold"):
        if threshold not in payload:
            raise ValueError(f"base profile is missing {threshold}")
    parent_digest = _sha256(base)
    payload.update(
        {
            "profile_name": f"{payload.get('profile_name', 'level1')}-user-calibrated",
            "working_zone": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "tub_zone": [[x, y] for x, y in container],
            "serving_zone": [[x, y] for x, y in customer],
            "calibration": {
                "calibrated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "parent_profile_sha256": parent_digest,
                "source_name": source_name,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "container_semantics": "ice_cream_tub_or_freezer_area",
                "customer_semantics": "serving_or_customer_handoff_area",
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return CalibrationResult(
        profile_path=output,
        profile_sha256=_sha256(output),
        container_zone=container,
        customer_zone=customer,
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _fit(frame: np.ndarray, maximum_width: int = 1280, maximum_height: int = 800) -> np.ndarray:
    scale = min(maximum_width / frame.shape[1], maximum_height / frame.shape[0])
    if abs(scale - 1.0) < 1e-6:
        return frame.copy()
    return cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _draw_polygon(
    canvas: np.ndarray,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    label: str,
    *,
    closed: bool,
) -> None:
    if points:
        array = np.asarray(points, dtype=np.int32)
        cv2.polylines(canvas, [array], closed, color, 3, cv2.LINE_AA)
        for index, point in enumerate(points, start=1):
            cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index),
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        anchor = points[0]
        cv2.putText(
            canvas,
            label,
            (anchor[0], max(24, anchor[1] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def run_zone_calibrator(
    *,
    source: str,
    base_profile_path: str | Path,
    output_path: str | Path,
    seek_seconds: float = 0.0,
    overwrite: bool = False,
) -> int:
    """Open the interactive calibrator and save on explicit confirmation."""

    if seek_seconds < 0:
        raise ValueError("seek_seconds cannot be negative")
    parsed_source: int | str = int(source) if source.strip().isdigit() else source
    capture = cv2.VideoCapture(parsed_source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open calibration source: {safe_source_name(parsed_source)}")
    try:
        if seek_seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError("could not read calibration frame")
    display = _fit(frame)
    height, width = display.shape[:2]
    window = "Scoop AI zone calibration"
    state: dict[str, object] = {"stage": 0, "zones": [[], []], "message": ""}

    def mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        stage = int(state["stage"])
        zones = state["zones"]
        assert isinstance(zones, list)
        if stage > 1:
            return
        points = zones[stage]
        assert isinstance(points, list)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((min(width - 1, max(0, x)), min(height - 1, max(0, y))))
            state["message"] = ""
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            state["message"] = "Last point removed"

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            canvas = display.copy()
            zones = state["zones"]
            assert isinstance(zones, list)
            stage = int(state["stage"])
            _draw_polygon(canvas, zones[0], (0, 190, 255), "CONTAINER / TUB", closed=stage > 0)
            _draw_polygon(canvas, zones[1], (60, 255, 80), "CUSTOMER / SERVING", closed=stage > 1)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 92), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            if stage == 0:
                title = "STEP 1/2: Click around ICE-CREAM CONTAINER / FREEZER area"
                controls = "Click boundary in order | C:clear | Backspace:undo | Enter:next | R:reset | Q:cancel"
            elif stage == 1:
                title = "STEP 2/2: Click around CUSTOMER / SERVING handoff area"
                controls = "Click clockwise | C:clear | Backspace:undo | Enter:preview | S:save | Q:cancel"
            else:
                title = "PREVIEW: Orange=container, Green=customer area"
                controls = "S:save | C:redraw customer | R:redraw everything | Q:cancel"
            cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, controls, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            message = str(state.get("message", ""))
            if message:
                cv2.putText(canvas, message, (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 180, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), ord("Q"), 27):
                return 1
            if key in (8, 127) and stage <= 1:
                points = zones[stage]
                if points:
                    points.pop()
                    state["message"] = "Last point removed"
            elif key in (ord("r"), ord("R")):
                state.update({"stage": 0, "zones": [[], []], "message": "Redraw started"})
            elif key in (ord("c"), ord("C")):
                if stage == 2:
                    zones[1].clear()
                    state["stage"] = 1
                    state["message"] = "Customer zone cleared"
                else:
                    zones[stage].clear()
                    state["message"] = "Current zone cleared"
            elif key in (10, 13, ord("n"), ord("N")) and stage <= 1:
                points = zones[stage]
                normalized = [(x / width, y / height) for x, y in points]
                try:
                    validate_zone(normalized, "container_zone" if stage == 0 else "customer_zone")
                except ValueError as exc:
                    state["message"] = str(exc)
                else:
                    state["stage"] = stage + 1
                    state["message"] = "Zone accepted"
            elif key in (ord("s"), ord("S")):
                if stage == 0:
                    state["message"] = "Finish the container zone and press Enter first"
                    continue
                if stage == 1:
                    normalized = [(x / width, y / height) for x, y in zones[1]]
                    try:
                        validate_zone(normalized, "customer_zone")
                    except ValueError as exc:
                        state["message"] = str(exc)
                        continue
                    state["stage"] = 2
                container = [(x / width, y / height) for x, y in zones[0]]
                customer = [(x / width, y / height) for x, y in zones[1]]
                try:
                    result = save_calibrated_profile(
                        base_profile_path,
                        output_path,
                        container_zone=container,
                        customer_zone=customer,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        source_name=safe_source_name(parsed_source),
                        overwrite=overwrite,
                    )
                except ValueError as exc:
                    state["message"] = str(exc)
                else:
                    print(
                        json.dumps(
                            {
                                "profile": str(result.profile_path),
                                "sha256": result.profile_sha256,
                                "container_zone": result.container_zone,
                                "customer_zone": result.customer_zone,
                            },
                            indent=2,
                        )
                    )
                    success = display.copy()
                    shade = success.copy()
                    cv2.rectangle(shade, (0, 0), (success.shape[1], 110), (0, 0, 0), -1)
                    cv2.addWeighted(shade, 0.78, success, 0.22, 0, success)
                    cv2.putText(
                        success,
                        "PROFILE SAVED SUCCESSFULLY",
                        (18, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (80, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        success,
                        str(result.profile_path),
                        (18, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window, success)
                    cv2.waitKey(1500)
                    return 0
    finally:
        cv2.destroyWindow(window)
