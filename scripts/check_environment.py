"""Print actionable installation diagnostics without downloading model weights."""

from __future__ import annotations

import platform

import cv2
import rfdetr
import torch


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"RF-DETR: {getattr(rfdetr, '__version__', 'installed')}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is unavailable; inference will be much slower on CPU.")


if __name__ == "__main__":
    main()

