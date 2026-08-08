"""Optional privacy-preserving image export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


NormalizedBox = tuple[float, float, float, float]
NormalizedZone = tuple[tuple[float, float], ...]


def redact_frame(
    frame: np.ndarray,
    *,
    boxes: Iterable[NormalizedBox] = (),
    zones: Iterable[NormalizedZone] = (),
    blur_kernel: int = 51,
) -> np.ndarray:
    """Return a copy with tracked-person boxes and/or polygon zones blurred."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty BGR image")
    if blur_kernel < 3 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd integer >= 3")
    height, width = frame.shape[:2]
    output = frame.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        if not all(0 <= value <= 1 for value in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
            raise ValueError("redaction boxes must be normalized and ordered")
        cv2.rectangle(
            mask,
            (round(x1 * width), round(y1 * height)),
            (round(x2 * width), round(y2 * height)),
            255,
            thickness=-1,
        )
    for zone in zones:
        if len(zone) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in zone):
            raise ValueError("redaction zones must contain normalized polygons")
        polygon = np.array([(round(x * width), round(y * height)) for x, y in zone], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    blurred = cv2.GaussianBlur(output, (blur_kernel, blur_kernel), 0)
    output[mask > 0] = blurred[mask > 0]
    return output


def export_redacted_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    boxes: Iterable[NormalizedBox] = (),
    zones: Iterable[NormalizedZone] = (),
    blur_kernel: int = 51,
) -> Path:
    source = Path(input_path)
    destination = Path(output_path)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {source}")
    redacted = redact_frame(image, boxes=boxes, zones=zones, blur_kernel=blur_kernel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    if not cv2.imwrite(str(temporary), redacted):
        raise RuntimeError(f"could not write redacted evidence: {destination}")
    temporary.replace(destination)
    return destination
