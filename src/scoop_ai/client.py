"""Frozen-friendly entry point for the installed Windows client.

Double-clicking the shortcut opens a desktop window, not a browser. ``--browser``
keeps the loopback web dashboard available for support and troubleshooting.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def _defaults() -> tuple[Path, Path]:
    data_root = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "ScoopAI"
    manifest = _bundle_root() / "models" / "ice-cream-item-rfdetr-nano-v2" / "model-manifest.json"
    return data_root, manifest


def _parser() -> argparse.ArgumentParser:
    data_root, manifest = _defaults()
    parser = argparse.ArgumentParser(prog="Scoop AI")
    parser.add_argument("--product-root", type=Path, default=data_root)
    parser.add_argument("--checkpoint-manifest", type=Path, default=manifest)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the loopback dashboard in a web browser instead of the desktop window",
    )
    parser.add_argument(
        "--software-rendering",
        action="store_true",
        help="Render the desktop window without the display GPU (use over Remote Desktop)",
    )
    parser.add_argument(
        "--start-hidden",
        action="store_true",
        help="Start minimised to the notification area",
    )
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-service", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--service-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--camera-config", type=Path, help=argparse.SUPPRESS)
    return parser


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/product/status", timeout=0.8):
            return True
    except OSError:
        return False


def _run_browser_mode(args: argparse.Namespace) -> int:
    """Serve the dashboard, then hand the operator a browser tab."""

    url = f"http://127.0.0.1:{args.port}"
    if args.no_browser:
        from scoop_ai.dashboard import run_product

        return run_product(
            args.product_root, args.checkpoint_manifest, port=args.port, open_browser=False
        )
    if not _reachable(url):
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        subprocess.Popen(
            [
                sys.executable, "--product-root", str(args.product_root),
                "--checkpoint-manifest", str(args.checkpoint_manifest),
                "--port", str(args.port), "--browser", "--no-browser",
            ],
            creationflags=flags,
            close_fds=True,
        )
        for _ in range(40):
            if _reachable(url):
                break
            time.sleep(0.25)
    webbrowser.open(url)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_service:
        if args.service_config is None or args.camera_config is None:
            raise SystemExit("service configuration is incomplete")
        from scoop_ai.application.service import run_service

        return run_service(
            service_config_path=args.service_config,
            camera_config_path=args.camera_config,
            checkpoint_manifest_path=args.checkpoint_manifest,
        )
    if args.browser or args.no_browser:
        return _run_browser_mode(args)

    from scoop_ai.desktop import DesktopLaunchError, run_desktop

    try:
        return run_desktop(
            args.product_root,
            args.checkpoint_manifest,
            port=args.port,
            software_rendering=args.software_rendering,
            start_hidden=args.start_hidden,
        )
    except DesktopLaunchError as exc:
        # A machine that cannot host the embedded window can still be supported
        # through the loopback dashboard, so fall back instead of failing.
        print(f"{exc} Falling back to the browser dashboard.", file=sys.stderr)
        return _run_browser_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
