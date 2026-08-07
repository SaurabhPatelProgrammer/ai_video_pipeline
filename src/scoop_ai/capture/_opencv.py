"""Small OpenCV adapter kept injectable for deterministic tests."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, property_id: int) -> float: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int | str], VideoCaptureLike]

_FFMPEG_ENV_LOCK = threading.Lock()


def default_capture_factory(source: int | str) -> VideoCaptureLike:
    return cv2.VideoCapture(source)


def open_live_capture(
    source: int | str,
    *,
    rtsp_transport: str,
    open_timeout_ms: int,
    read_timeout_ms: int,
    capture_factory: CaptureFactory | None,
) -> VideoCaptureLike:
    if capture_factory is not None:
        return capture_factory(source)

    source_text = str(source).lower()
    if not source_text.startswith(("rtsp://", "rtsps://")):
        capture = cv2.VideoCapture(source)
    else:
        # OpenCV reads this process-global option while opening the FFmpeg
        # capture. Serialize and restore it so one camera cannot leak transport
        # settings into another camera's open operation.
        key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        with _FFMPEG_ENV_LOCK:
            previous = os.environ.get(key)
            os.environ[key] = f"rtsp_transport;{rtsp_transport}"
            try:
                capture = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        open_timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        read_timeout_ms,
                    ],
                )
            finally:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture
