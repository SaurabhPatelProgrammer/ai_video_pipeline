"""Low-latency camera reader that always exposes the newest available frame."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from scoop_ai.capture import RecordedFrameSource

LOGGER = logging.getLogger(__name__)


def parse_source(value: str) -> int | str:
    value = value.strip()
    return int(value) if value.isdigit() else value


def safe_source_name(source: int | str) -> str:
    """Return a log-safe source name without credentials or query tokens."""
    if isinstance(source, int):
        return f"webcam:{source}"
    parts = urlsplit(source)
    if parts.scheme and parts.hostname:
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{parts.hostname}{port}{parts.path}"
    return Path(source).name or "video-source"


class LatestFrameSource:
    """Read in a background thread so slow inference does not build an RTSP queue."""

    def __init__(
        self,
        source: int | str,
        reconnect_seconds: float = 3.0,
        rtsp_transport: str = "tcp",
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        self.source = source
        self.reconnect_seconds = reconnect_seconds
        self.rtsp_transport = rtsp_transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = -1
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._recorded_source: RecordedFrameSource | None = None
        self.media_timestamp_seconds: float | None = None
        self.recorded_eof = False
        self.last_error: str | None = None

    def start(self) -> "LatestFrameSource":
        if isinstance(self.source, str) and Path(self.source).is_file():
            if self._recorded_source is None:
                self._recorded_source = RecordedFrameSource(
                    self.source,
                    source_id=safe_source_name(self.source),
                ).start()
            return self
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()
        return self

    def read(
        self,
        after_sequence: int = -1,
        timeout: float = 2.0,
    ) -> tuple[int, np.ndarray | None]:
        if self._recorded_source is not None:
            packet = self._recorded_source.read()
            if packet is None:
                self.recorded_eof = True
                return after_sequence, None
            self.media_timestamp_seconds = packet.media_timestamp_seconds
            return packet.sequence, packet.frame.copy()
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != after_sequence or self._stopping.is_set(),
                timeout=timeout,
            )
            if self._sequence == after_sequence or self._frame is None:
                return after_sequence, None
            return self._sequence, self._frame.copy()

    def stop(self) -> None:
        if self._recorded_source is not None:
            self._recorded_source.stop()
            self._recorded_source = None
            return
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _open(self) -> cv2.VideoCapture:
        source_text = str(self.source).lower()
        if source_text.startswith(("rtsp://", "rtsps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}"
            )
            capture = cv2.VideoCapture(
                self.source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    self.open_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    self.read_timeout_ms,
                ],
            )
        else:
            capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _reader_loop(self) -> None:
        display_name = safe_source_name(self.source)
        while not self._stopping.is_set():
            capture = self._open()
            if not capture.isOpened():
                self.last_error = f"Could not open {display_name}"
                LOGGER.warning("%s; retrying in %.1fs", self.last_error, self.reconnect_seconds)
                capture.release()
                self._stopping.wait(self.reconnect_seconds)
                continue

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            LOGGER.info(
                "Connected to %s (%dx%d, reported %.1f FPS)",
                display_name,
                width,
                height,
                fps,
            )
            self.last_error = None

            while not self._stopping.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.last_error = f"Frame read failed for {display_name}"
                    LOGGER.warning(
                        "%s; reconnecting in %.1fs",
                        self.last_error,
                        self.reconnect_seconds,
                    )
                    break
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()

            capture.release()
            if not self._stopping.is_set():
                self._stopping.wait(self.reconnect_seconds)
