"""PyWin32 wrapper for running the edge pipeline under Windows SCM."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:  # pragma: no cover - exercised only on non-Windows installs
    servicemanager = None
    win32event = None
    win32service = None
    win32serviceutil = None


DEFAULT_SETTINGS = Path(r"D:\ip-camera-ai-data\service\windows-service.json")


def _settings_path() -> Path:
    return Path(os.getenv("SCOOP_AI_WINDOWS_SERVICE_SETTINGS", str(DEFAULT_SETTINGS)))


def _load_settings() -> dict[str, str]:
    path = _settings_path()
    try:
        # Windows PowerShell 5.1 writes a BOM for ``-Encoding UTF8``.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load Windows service settings {path}: {exc}") from exc
    required = {"service_config", "camera_config", "checkpoint_manifest"}
    if not isinstance(data, dict) or set(data) != required:
        raise RuntimeError(f"Windows service settings must contain exactly {sorted(required)}")
    output = {key: str(value) for key, value in data.items()}
    for key, value in output.items():
        if not Path(value).is_file():
            raise RuntimeError(f"{key} file not found: {value}")
    return output


if win32serviceutil is not None:
    class ScoopAIWindowsService(win32serviceutil.ServiceFramework):
        """Importable service class required by the Windows SCM."""

        _svc_name_ = "ScoopAIEdge"
        _svc_display_name_ = "Scoop AI Edge Counter"
        _svc_description_ = "Single-camera local scoop-event inference service"

        def __init__(self, args) -> None:  # noqa: ANN001
            super().__init__(args)
            self.stop_event = threading.Event()
            self.stop_handle = win32event.CreateEvent(None, 0, 0, None)

        def SvcStop(self) -> None:  # noqa: N802
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            self.stop_event.set()
            win32event.SetEvent(self.stop_handle)

        def SvcDoRun(self) -> None:  # noqa: N802
            from .application.service import run_service

            settings = _load_settings()
            servicemanager.LogInfoMsg("Scoop AI Edge service starting")
            result = run_service(
                service_config_path=settings["service_config"],
                camera_config_path=settings["camera_config"],
                checkpoint_manifest_path=settings["checkpoint_manifest"],
                stop_event=self.stop_event,
            )
            if result:
                raise RuntimeError(f"Scoop AI service exited with code {result}")
            servicemanager.LogInfoMsg("Scoop AI Edge service stopped")


def main() -> None:
    if win32serviceutil is None:
        raise SystemExit("pywin32 is required to manage the Windows service")

    win32serviceutil.HandleCommandLine(ScoopAIWindowsService)


if __name__ == "__main__":
    main()
