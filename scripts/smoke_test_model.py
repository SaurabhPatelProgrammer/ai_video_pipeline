"""Download RF-DETR Nano weights and run one GPU inference."""

from __future__ import annotations

import time

import numpy as np
import torch
from rfdetr import RFDETRNano


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RF-DETR Nano on {device}...")
    model = RFDETRNano(device=device)

    # RGB image. A blank frame is enough to validate model loading and execution.
    image = np.zeros((384, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    detections = model.predict(
        image,
        threshold=0.40,
        include_source_image=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    count = len(getattr(detections, "xyxy", []))
    print(f"PASS: inference completed in {elapsed_ms:.0f} ms; detections={count}")


if __name__ == "__main__":
    main()

