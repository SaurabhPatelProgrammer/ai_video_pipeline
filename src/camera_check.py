"""Verify a webcam, RTSP URL, HTTP stream, or video file before loading AI."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from video_source import LatestFrameSource, parse_source, safe_source_name


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check that a camera stream can be read.")
    parser.add_argument("--source", default=os.getenv("CAMERA_URL", "0"))
    parser.add_argument("--seconds", type=float, default=0, help="0 means run until Q.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    source = parse_source(args.source)
    recorded_file = isinstance(source, str) and Path(source).is_file()
    camera = LatestFrameSource(
        source,
        reconnect_seconds=float(os.getenv("RECONNECT_SECONDS", "3")),
        rtsp_transport=os.getenv("RTSP_TRANSPORT", "tcp"),
        open_timeout_ms=int(os.getenv("CAMERA_OPEN_TIMEOUT_MS", "5000")),
        read_timeout_ms=int(os.getenv("CAMERA_READ_TIMEOUT_MS", "5000")),
    ).start()

    started = time.monotonic()
    last_sequence = -1
    frames = 0
    received_frame = False
    last_report = started
    logging.info("Checking %s. Press Q to stop.", safe_source_name(source))

    try:
        while True:
            sequence, frame = camera.read(last_sequence, timeout=2.0)
            if frame is None:
                if recorded_file and camera.recorded_eof:
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                continue
            last_sequence = sequence
            received_frame = True
            frames += 1
            now = time.monotonic()

            if now - last_report >= 2.0:
                logging.info(
                    "Camera OK: %dx%d, received %.1f FPS",
                    frame.shape[1],
                    frame.shape[0],
                    frames / (now - last_report),
                )
                frames = 0
                last_report = now

            if not args.headless:
                cv2.imshow("Camera check - Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.seconds and now - started >= args.seconds:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if not received_frame:
        logging.error("No readable frame was received from %s.", safe_source_name(source))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
