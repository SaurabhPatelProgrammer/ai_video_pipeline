"""Local MediaMTX relay, raw recording and safe event-clip helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .storage import EvidenceRecord, EvidenceWriter, RecordingSegmentRecord, SQLiteEventRepository, utc_now_iso
from .capture import RecordedFrameSource
from .capture.models import CaptureHealth, FramePacket, TimestampDomain


class RecordingError(RuntimeError):
    pass


class BufferedRecordingSource:
    """Read only completed local recording segments, never the slow camera stream.

    New segments are picked up after MediaMTX has closed them.  Existing files
    are deliberately ignored at startup: this is a live buffered mode, not a
    surprise replay of old shop footage after a restart.
    """

    def __init__(self, recordings_root: Path, camera_id: str) -> None:
        self.root = recordings_root / camera_id
        self.camera_id = camera_id
        self._seen: set[Path] = set()
        self._reader: RecordedFrameSource | None = None
        self._segment_started: datetime | None = None
        self._active_path: Path | None = None
        self._sequence = -1

    @property
    def health(self) -> CaptureHealth:
        return self._reader.health if self._reader is not None else CaptureHealth.initial(self.camera_id)

    def start(self) -> "BufferedRecordingSource":
        self.root.mkdir(parents=True, exist_ok=True)
        self._seen = set(self.root.rglob("*.mp4"))
        return self

    def stop(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

    def frame_age_seconds(self) -> float | None:
        return self.health.frame_age_seconds(time.monotonic())

    def read(self) -> FramePacket | None:
        while True:
            if self._reader is None:
                candidate = self._next_completed_segment()
                if candidate is None:
                    return None
                started = iso_from_filename(candidate)
                if started is None:
                    self._seen.add(candidate)
                    continue
                self._segment_started = started
                try:
                    self._reader = RecordedFrameSource(candidate, source_id=self.camera_id).start()
                except RuntimeError:
                    # A just-closed segment can take a moment to become readable
                    # on some network filesystems. Leave it eligible for retry.
                    self._reader = None
                    return None
                self._active_path = candidate
            packet = self._reader.read()
            if packet is None:
                self._reader.stop()
                self._reader = None
                self._segment_started = None
                if self._active_path is not None:
                    self._seen.add(self._active_path)
                    self._active_path = None
                continue
            self._sequence += 1
            assert self._segment_started is not None
            return replace(
                packet,
                sequence=self._sequence,
                timestamp_seconds=self._segment_started.timestamp() + packet.timestamp_seconds,
                timestamp_domain=TimestampDomain.UTC_EPOCH,
            )

    def _next_completed_segment(self) -> Path | None:
        cutoff = time.time() - 2.0
        for path in sorted(self.root.rglob("*.mp4")):
            if path not in self._seen and path.stat().st_mtime <= cutoff:
                return path
        return None


def local_relay_url(camera_id: str, port: int = 8554) -> str:
    return f"rtsp://127.0.0.1:{port}/{camera_id}"


def find_mediamtx(product_root: Path) -> Path:
    """Find only locally installed/bundled binaries; never download at runtime."""
    candidates = [
        product_root / "bin" / "mediamtx.exe",
        Path(__file__).resolve().parent / "bin" / "mediamtx.exe",
    ]
    configured = os.environ.get("SCOOP_AI_MEDIAMTX")
    if configured:
        candidates.insert(0, Path(configured))
    found = shutil.which("mediamtx") or shutil.which("mediamtx.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RecordingError(
        "MediaMTX was not found. Install the bundled Scoop AI recorder component or set SCOOP_AI_MEDIAMTX."
    )


def find_ffmpeg(product_root: Path) -> Path:
    """Find the local FFmpeg companion before falling back to PATH."""

    candidates = [product_root / "bin" / "ffmpeg.exe", product_root / "bin" / "ffmpeg"]
    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RecordingError("FFmpeg was not found. Install ffmpeg.exe under Scoop AI\\bin or put it on PATH.")


def _quote(value: str) -> str:
    # YAML double strings with JSON-compatible escaping; keep credentials out of logs.
    import json
    return json.dumps(value)


def relay_config(*, source: str, camera_id: str, recordings_root: Path, port: int,
                 segment_seconds: int, retention_days: int) -> str:
    """Build an intentionally loopback-only MediaMTX configuration."""
    if urlsplit(source).scheme not in {"rtsp", "rtsps"}:
        raise RecordingError("local relay requires an RTSP camera source")
    if not camera_id.replace("-", "").replace("_", "").isalnum():
        raise RecordingError("camera ID is unsafe for a MediaMTX path")
    # MediaMTX requires the path placeholder in recordPath.  Without %path it
    # rejects the configuration before opening the camera, leaving monitoring
    # alive but producing no recording segments.
    record_path = (recordings_root / "%path" / "%Y-%m-%d_%H-%M-%S-%f").as_posix()
    return "\n".join((
        "logLevel: info",
        f"rtspAddress: 127.0.0.1:{port}",
        "api: yes",
        "apiAddress: 127.0.0.1:9997",
        "hls: no",
        "webrtc: no",
        "pathDefaults:",
        "  record: yes",
        f"  recordPath: {_quote(record_path)}",
        "  recordFormat: fmp4",
        "  recordPartDuration: 1s",
        f"  recordSegmentDuration: {segment_seconds}s",
        f"  recordDeleteAfter: {retention_days * 24}h",
        "paths:",
        f"  {camera_id}:",
        f"    source: {_quote(source)}",
        "    rtspTransport: tcp",
        "",
    ))


class RelaySupervisor:
    """Own one local recorder process and remove its secret-bearing config on stop."""

    def __init__(self, *, product_root: Path, source: str, camera_id: str, recordings_root: Path,
                 port: int = 8554, segment_seconds: int = 300, retention_days: int = 7) -> None:
        self.product_root, self.source, self.camera_id = product_root, source, camera_id
        self.recordings_root, self.port = recordings_root, port
        self.segment_seconds, self.retention_days = segment_seconds, retention_days
        self.process: subprocess.Popen[bytes] | None = None
        self._config: Path | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> str:
        if self.running:
            return local_relay_url(self.camera_id, self.port)
        executable = find_mediamtx(self.product_root)
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        runtime = self.product_root / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix="mediamtx-", suffix=".yml", dir=runtime)
        self._config = Path(temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(relay_config(source=self.source, camera_id=self.camera_id,
                recordings_root=self.recordings_root, port=self.port,
                segment_seconds=self.segment_seconds, retention_days=self.retention_days))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen([str(executable), str(self._config)], cwd=str(self.product_root), creationflags=flags)
        # MediaMTX validates its configuration synchronously and exits quickly
        # on errors.  Do not hand the service a dead loopback URL in that case.
        time.sleep(0.25)
        if self.process.poll() is not None:
            exit_code = self.process.returncode
            self.process = None
            if self._config is not None:
                self._config.unlink(missing_ok=True)
                self._config = None
            raise RecordingError(f"MediaMTX stopped during startup (exit code {exit_code}).")
        return local_relay_url(self.camera_id, self.port)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._config is not None:
            self._config.unlink(missing_ok=True)
            self._config = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_from_filename(path: Path) -> datetime | None:
    """Parse MediaMTX's configured YYYY-mm-dd_HH-MM-SS-microseconds filename."""
    try:
        return datetime.strptime(path.stem.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def index_completed_segments(repository: SQLiteEventRepository, recordings_root: Path, camera_id: str,
                             segment_seconds: int) -> int:
    """Idempotently index MediaMTX files; suitable after restart or clip work."""
    root = recordings_root / camera_id
    if not root.is_dir():
        return 0
    indexed = 0
    for path in root.rglob("*.mp4"):
        started = iso_from_filename(path)
        if started is None or not path.is_file():
            continue
        relative = path.relative_to(recordings_root).as_posix()
        if repository.register_recording_segment(RecordingSegmentRecord(
            segment_id=hashlib.sha256(relative.encode("utf-8")).hexdigest(), camera_id=camera_id,
            relative_path=relative, started_at=started.isoformat(),
            finished_at=(started + timedelta(seconds=segment_seconds)).isoformat(),
            size_bytes=path.stat().st_size, sha256=sha256_file(path),
        )):
            indexed += 1
    return indexed


class EventClipWorker:
    """Extract bounded event clips after MediaMTX has completed the segment.

    A segment may close up to ``segment_seconds`` after an event, therefore
    queued rows stay pending and are retried; an event itself is never lost when
    FFmpeg or a camera recording is unavailable.
    """

    def __init__(self, repository: SQLiteEventRepository, evidence: EvidenceWriter, recordings_root: Path,
                 camera_id: str, segment_seconds: int, before_seconds: int, after_seconds: int) -> None:
        self.repository, self.evidence, self.root, self.camera_id = repository, evidence, recordings_root, camera_id
        self.segment_seconds, self.before, self.after = segment_seconds, before_seconds, after_seconds

    def queue(self, event_id: str, occurred_at: str, session_id: str) -> None:
        event_time = datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).astimezone(timezone.utc)
        begin = event_time.timestamp() - self.before
        end = event_time.timestamp() + self.after
        self.repository.queue_event_video(
            event_id,
            source_started_at=datetime.fromtimestamp(begin, timezone.utc).isoformat(),
            source_finished_at=datetime.fromtimestamp(end, timezone.utc).isoformat(),
        )
        thread = threading.Thread(target=self._extract, args=(event_id, event_time, session_id), daemon=True,
            name=f"scoop-event-clip-{event_id[:8]}")
        thread.start()

    def _extract(self, event_id: str, event_time: datetime, session_id: str) -> None:
        # A completed segment is necessary for a stable fMP4 input.
        import time
        time.sleep(self.after + min(self.segment_seconds, 30))
        try:
            index_completed_segments(self.repository, self.root, self.camera_id, self.segment_seconds)
            ffmpeg = find_ffmpeg(self.root.parent)
            candidates: list[tuple[datetime, Path]] = []
            camera_root = self.root / self.camera_id
            for path in camera_root.rglob("*.mp4") if camera_root.is_dir() else ():
                started = iso_from_filename(path)
                if started is not None and started <= event_time + timedelta(seconds=self.after) and \
                   started + timedelta(seconds=self.segment_seconds) >= event_time - timedelta(seconds=self.before):
                    candidates.append((started, path))
            if not candidates:
                raise RecordingError("recording segment is not available yet")
            candidates.sort()
            # FFmpeg concat keeps the operation deterministic and avoids reading a live camera again.
            with tempfile.TemporaryDirectory(prefix="scoop-clip-") as temporary:
                work = Path(temporary)
                listing = work / "segments.txt"
                listing.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for _, path in candidates), encoding="utf-8")
                output = work / "event.mp4"
                first = candidates[0][0]
                offset = max(0.0, (event_time.timestamp() - self.before) - first.timestamp())
                duration = self.before + self.after
                result = subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-ss", f"{offset:.3f}", "-t", str(duration), "-c", "copy", "-movflags", "+faststart", str(output)], capture_output=True, timeout=120)
                if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                    raise RecordingError("FFmpeg could not create the event clip")
                artifact = self.evidence.write_bytes(session_id, f"{event_id}-video", output.read_bytes(), extension=".mp4")
            self.repository.complete_event_video(event_id, relative_path=artifact.relative_path, sha256=artifact.sha256, size_bytes=artifact.size_bytes)
            self.repository.register_evidence(EvidenceRecord(
                evidence_id=str(uuid.uuid4()), event_id=event_id, relative_path=artifact.relative_path,
                sha256=artifact.sha256, size_bytes=artifact.size_bytes, media_type="video/mp4",
                created_at=artifact.created_at, retention_deadline=(datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
                metadata={"kind": "event_clip", "event_id": event_id},
            ))
        except Exception as exc:
            self.repository.fail_event_video(event_id, str(exc))
            self.repository.update_event_review_state(
                event_id, "needs_review", extra_metadata={"event_video_error": str(exc)[:500]}
            )
