"""First-run onboarding and local monitoring lifecycle for the client product."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..calibration import create_reference_fingerprint, validate_zone
from ..config import load_camera_config, load_service_config
from ..inference.checkpoint_manifest import load_checkpoint_manifest
from ..security import safe_source_name, store_credential
from ..storage import AuditLogRecord, SQLiteEventRepository, utc_now_iso
from .discovery import DiscoveredCamera, discover_onvif_cameras


CAMERA_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")


@dataclass(frozen=True, slots=True)
class ProductPaths:
    root: Path
    database: Path
    evidence: Path
    service_config: Path
    camera_config: Path
    calibration: Path
    settings: Path
    identity: Path

    @classmethod
    def under(cls, root: str | Path) -> "ProductPaths":
        base = Path(root).resolve()
        return cls(
            root=base,
            database=base / "database" / "events.sqlite3",
            evidence=base / "evidence",
            service_config=base / "service" / "service.toml",
            camera_config=base / "service" / "camera.toml",
            calibration=base / "calibration" / "camera.json",
            settings=base / "service" / "product.json",
            identity=base / "service" / "identity.json",
        )


@dataclass(slots=True)
class PreviewSession:
    source: str
    frame: np.ndarray
    created_at: float
    camera_device_identity: str | None = None


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _toml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _zone_text(points: tuple[tuple[float, float], ...]) -> str:
    return "[" + ", ".join(f"[{x:.6f}, {y:.6f}]" for x, y in points) + "]"


class ProductManager:
    """Own the client configuration without ever persisting the camera secret."""

    def __init__(
        self,
        root: str | Path,
        checkpoint_manifest: str | Path,
        *,
        credential_writer: Callable[[str, str], None] = store_credential,
        discovery_provider: Callable[[], list[DiscoveredCamera]] = discover_onvif_cameras,
        process_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.paths = ProductPaths.under(root)
        self.checkpoint_manifest = Path(checkpoint_manifest).resolve()
        self.credential_writer = credential_writer
        self.discovery_provider = discovery_provider
        self.process_factory = process_factory
        self._previews: dict[str, PreviewSession] = {}
        self._discoveries: dict[str, DiscoveredCamera] = {}
        self._lock = threading.RLock()
        self._monitor: subprocess.Popen[bytes] | None = None
        self._ensure_identity()

    def _ensure_identity(self) -> dict[str, object]:
        if self.paths.identity.is_file():
            try:
                loaded = json.loads(self.paths.identity.read_text(encoding="utf-8"))
                if (
                    isinstance(loaded, dict)
                    and isinstance(loaded.get("device_id"), str)
                    and isinstance(loaded.get("site_id"), str)
                ):
                    return loaded
            except (OSError, json.JSONDecodeError):
                pass
        identity: dict[str, object] = {
            "schema_version": 1,
            "device_id": f"edge-{uuid.uuid4()}",
            "site_id": f"local-{uuid.uuid4()}",
            "binding_state": "local_only",
            "created_at": utc_now_iso(),
        }
        _atomic_text(self.paths.identity, json.dumps(identity, indent=2, sort_keys=True) + "\n")
        return identity

    @property
    def configured(self) -> bool:
        return all(path.is_file() for path in (
            self.paths.settings, self.paths.service_config, self.paths.camera_config, self.paths.calibration,
        ))

    def status(self) -> dict[str, object]:
        settings: dict[str, object] = {}
        if self.paths.settings.is_file():
            try:
                loaded = json.loads(self.paths.settings.read_text(encoding="utf-8"))
                settings = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                settings = {}
        identity = self._ensure_identity()
        return {
            "configured": self.configured,
            "shop_name": settings.get("shop_name"),
            "camera_name": settings.get("camera_name"),
            "camera_id": settings.get("camera_id"),
            "monitoring": self.monitoring,
            "pilot_mode": True,
            "device_id": identity["device_id"],
            "site_id": identity["site_id"],
            "binding_state": identity.get("binding_state", "local_only"),
        }

    @property
    def monitoring(self) -> bool:
        with self._lock:
            if self._monitor is not None and self._monitor.poll() is not None:
                self._monitor = None
            return self._monitor is not None

    def discover_cameras(self) -> list[dict[str, object]]:
        cameras = self.discovery_provider()
        with self._lock:
            self._discoveries = {camera.device_id: camera for camera in cameras}
        return [camera.as_payload() for camera in cameras]

    def test_camera(
        self,
        source: str,
        *,
        camera_device_identity: str | None = None,
        timeout_seconds: float = 12.0,
    ) -> dict[str, object]:
        source = source.strip()
        if not source or len(source) > 2048:
            raise ValueError("camera URL or webcam number is required")
        if camera_device_identity is not None:
            with self._lock:
                if camera_device_identity not in self._discoveries:
                    raise ValueError("selected camera discovery expired; scan the network again")
        parsed_source: int | str = int(source) if source.isdigit() else source
        capture = cv2.VideoCapture()
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        capture.open(parsed_source)
        deadline = time.monotonic() + timeout_seconds
        frame: np.ndarray | None = None
        try:
            while time.monotonic() < deadline:
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
                    break
        finally:
            capture.release()
        if frame is None:
            raise ValueError("camera did not return a frame; check URL, password, network and codec")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("camera preview could not be encoded")
        token = uuid.uuid4().hex
        with self._lock:
            cutoff = time.monotonic() - 900
            self._previews = {key: item for key, item in self._previews.items() if item.created_at >= cutoff}
            self._previews[token] = PreviewSession(
                source=source,
                frame=frame.copy(),
                created_at=time.monotonic(),
                camera_device_identity=camera_device_identity,
            )
        return {
            "preview_token": token,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "jpeg": encoded.tobytes(),
            "source_name": safe_source_name(parsed_source),
        }

    def preview_jpeg(self, token: str) -> bytes | None:
        with self._lock:
            preview = self._previews.get(token)
            if preview is None or time.monotonic() - preview.created_at > 900:
                return None
            ok, encoded = cv2.imencode(".jpg", preview.frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return encoded.tobytes() if ok else None

    def save_setup(
        self,
        *,
        preview_token: str,
        shop_name: str,
        camera_name: str,
        camera_id: str,
        pickup_zone: object,
        customer_zone: object,
    ) -> dict[str, object]:
        shop_name = shop_name.strip()
        camera_name = camera_name.strip()
        camera_id = camera_id.strip().lower()
        if not 2 <= len(shop_name) <= 100 or not 2 <= len(camera_name) <= 100:
            raise ValueError("shop and camera names must be between 2 and 100 characters")
        if not CAMERA_ID.fullmatch(camera_id):
            raise ValueError("camera ID must use 3-64 lowercase letters, numbers or hyphens")
        if not isinstance(pickup_zone, list) or not isinstance(customer_zone, list):
            raise ValueError("both camera zones are required")
        pickup = validate_zone(pickup_zone, "pickup_zone")
        customer = validate_zone(customer_zone, "customer_zone")
        with self._lock:
            preview = self._previews.get(preview_token)
            if preview is None or time.monotonic() - preview.created_at > 900:
                raise ValueError("camera preview expired; test the camera again")
        height, width = preview.frame.shape[:2]
        masks: list[np.ndarray] = []
        for zone in (pickup, customer):
            points = np.asarray([[round(x * (width - 1)), round(y * (height - 1))] for x, y in zone], dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
        intersection = int(np.count_nonzero(masks[0] & masks[1]))
        smaller = min(int(np.count_nonzero(masks[0])), int(np.count_nonzero(masks[1])))
        if smaller and intersection / smaller > 0.10:
            raise ValueError("pickup and customer zones overlap too much")
        if self.monitoring:
            self.stop_monitoring()
        manifest = load_checkpoint_manifest(
            self.checkpoint_manifest, expected_classes=("ice_cream_item",), verify_checkpoint=True
        )
        credential_key = f"scoop-ai/{camera_id}/rtsp-url"
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.evidence.mkdir(parents=True, exist_ok=True)
        self.credential_writer(credential_key, preview.source)

        calibration_payload = {
            "schema_version": 1,
            "tub_zone": [[x, y] for x, y in pickup],
            "serving_zone": [[x, y] for x, y in customer],
            "calibration": {
                "calibrated_at_utc": utc_now_iso(),
                "source_name": safe_source_name(preview.source),
                "frame_width": int(preview.frame.shape[1]),
                "frame_height": int(preview.frame.shape[0]),
                "reference_fingerprint": create_reference_fingerprint(preview.frame),
            },
        }
        _atomic_text(self.paths.calibration, json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n")
        service_toml = f'''[service]
name = "scoop-ai-client"
environment = "development"
artifact_root = {_toml_string(self.paths.root)}
log_level = "INFO"
health_host = "127.0.0.1"
health_port = 8080
shutdown_timeout_seconds = 10.0
minimum_free_space_gb = 2.0
approved_reviewers = []
billing_mode = "disabled"
automatic_exports = false

[export]
enabled = false

[alerts]
poll_seconds = 5.0
stale_after_seconds = 10.0
reconnect_warning = 3
reconnect_critical = 10
low_disk_warning_gb = 5.0
low_disk_critical_gb = 1.0

[capture]
reconnect_seconds = 3.0
reconnect_max_seconds = 30.0
reconnect_jitter_ratio = 0.20
open_timeout_ms = 5000
read_timeout_ms = 5000
read_wait_seconds = 2.0
rtsp_transport = "tcp"
'''
        camera_toml = f'''[camera]
camera_id = {_toml_string(camera_id)}
enabled = true
mode = "live"
pipeline = "handover"
credential_key = {_toml_string(credential_key)}
analysis_fps = 6.0
calibration_profile = {_toml_string(self.paths.calibration)}
expected_width = {int(preview.frame.shape[1])}
expected_height = {int(preview.frame.shape[0])}

[zones]
tub = {_zone_text(pickup)}
serving = {_zone_text(customer)}

[quality]
minimum_blur_variance = 20.0
pixel_change_threshold = 35
maximum_changed_fraction = 0.65
analysis_width = 320
reference_drift_threshold = 0.20
zone_change_threshold = 0.45
obstruction_seconds = 1.0
minimum_fps_ratio = 0.50

[handover]
maximum_center_distance_pixels = 120.0
lost_track_seconds = 1.0
minimum_consecutive_frames = 1
duplicate_iou_threshold = 0.50
minimum_transfer_seconds = 0.20
minimum_customer_dwell_seconds = 0.12
minimum_customer_observations = 2
minimum_movement_distance = 0.03
sequence_timeout_seconds = 5.0
missing_tolerance_seconds = 1.0
duplicate_cooldown_seconds = 3.5
duplicate_distance = 0.12
'''
        _atomic_text(self.paths.service_config, service_toml)
        _atomic_text(self.paths.camera_config, camera_toml)
        load_service_config(self.paths.service_config)
        load_camera_config(self.paths.camera_config)
        settings = {
            "schema_version": 1,
            "shop_name": shop_name,
            "camera_name": camera_name,
            "camera_id": camera_id,
            "credential_key": credential_key,
            "camera_device_identity": preview.camera_device_identity,
            "model_version": manifest.model_version,
            "checkpoint_manifest": str(self.checkpoint_manifest),
            "pilot_mode": True,
            "auto_start_monitoring": True,
            "configured_at": utc_now_iso(),
        }
        _atomic_text(self.paths.settings, json.dumps(settings, indent=2, sort_keys=True) + "\n")
        identity = self._ensure_identity()
        identity["site_name"] = shop_name
        _atomic_text(self.paths.identity, json.dumps(identity, indent=2, sort_keys=True) + "\n")
        with SQLiteEventRepository(self.paths.database) as repository:
            repository.record_audit(AuditLogRecord(
                audit_id=str(uuid.uuid4()), occurred_at=utc_now_iso(), actor="local-setup-wizard",
                action="camera_onboarding", target=camera_id,
                details={"shop_name": shop_name, "camera_name": camera_name, "source": safe_source_name(preview.source)},
            ))
        with self._lock:
            self._previews.pop(preview_token, None)
        return self.status()

    def start_monitoring(self) -> dict[str, object]:
        if not self.configured:
            raise ValueError("finish camera setup before starting monitoring")
        with self._lock:
            if self.monitoring:
                return self.status()
            command = [sys.executable]
            if getattr(sys, "frozen", False):
                command.append("--run-service")
            else:
                command.extend(["-m", "scoop_ai.cli", "service"])
            command.extend([
                "--service-config", str(self.paths.service_config),
                "--camera-config", str(self.paths.camera_config),
                "--checkpoint-manifest", str(self.checkpoint_manifest),
            ])
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._monitor = self.process_factory(command, cwd=str(self.paths.root), creationflags=flags)
        return self.status()

    def stop_monitoring(self) -> dict[str, object]:
        with self._lock:
            process = self._monitor
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            self._monitor = None
        return self.status()

    def auto_start(self) -> None:
        if not self.configured:
            return
        try:
            settings = json.loads(self.paths.settings.read_text(encoding="utf-8"))
            if settings.get("auto_start_monitoring"):
                self.start_monitoring()
        except (OSError, ValueError, json.JSONDecodeError):
            return

    def close(self) -> None:
        # The monitoring subprocess intentionally survives dashboard restarts only
        # when hosted by the installed Windows service. In shortcut mode, stop it.
        self.stop_monitoring()
