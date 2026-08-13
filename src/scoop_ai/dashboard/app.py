"""Loopback-only dashboard backed by the local event database."""

from __future__ import annotations

import getpass
import ipaddress
import json
import mimetypes
import threading
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, time, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ..storage import ReviewRecord, SQLiteEventRepository, utc_now_iso
from .product import ProductManager


STATIC_ROOT = Path(__file__).with_name("static")
MAX_REQUEST_BYTES = 16 * 1024
REVIEW_STATES = {"unreviewed", "accepted", "rejected", "needs_review"}


def _local_day_bounds(value: str | None) -> tuple[str, str, str]:
    try:
        selected = date.fromisoformat(value) if value else datetime.now().astimezone().date()
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    local_zone = datetime.now().astimezone().tzinfo
    start = datetime.combine(selected, time.min, tzinfo=local_zone).astimezone(timezone.utc)
    end = datetime.combine(selected + timedelta(days=1), time.min, tzinfo=local_zone).astimezone(timezone.utc)
    return start.isoformat(), end.isoformat(), selected.isoformat()


def _quantity(event: Any, reviews: list[ReviewRecord]) -> int:
    if reviews:
        corrected = reviews[-1].metadata.get("corrected_quantity")
        if isinstance(corrected, int) and not isinstance(corrected, bool) and corrected > 0:
            return corrected
    raw = event.metadata.get("quantity", 1)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else 1


def _event_payload(repository: SQLiteEventRepository, event: Any) -> dict[str, object]:
    reviews = repository.list_reviews(event.event_id)
    route = event.metadata.get("route")
    item_track_id = event.metadata.get("item_track_id")
    return {
        "event_id": event.event_id,
        "camera_id": event.camera_id,
        "event_type": event.event_type,
        "occurred_at": event.occurred_at,
        "confidence": event.confidence,
        "review_state": event.review_state,
        "quantity": _quantity(event, reviews),
        "route": route if isinstance(route, str) else None,
        "item_track_id": item_track_id if isinstance(item_track_id, int) else None,
        "has_evidence": bool(event.evidence_path),
        "review_count": len(reviews),
        "latest_notes": reviews[-1].notes if reviews else None,
    }


class DashboardServer:
    """Serve a small same-origin UI; never exposes the database on the network."""

    def __init__(
        self,
        database_path: str | Path,
        evidence_root: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        health_url: str = "http://127.0.0.1:8080/health",
        product_manager: ProductManager | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("dashboard must bind to localhost/loopback")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.database_path = Path(database_path).resolve()
        self.evidence_root = Path(evidence_root).resolve()
        self.health_url = health_url
        self.product_manager = product_manager
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                owner._handle_get(self)

            def do_POST(self) -> None:  # noqa: N802
                owner._handle_post(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="scoop-dashboard", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> "DashboardServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'")
        handler.send_header("Referrer-Policy", "no-referrer")
        handler.end_headers()
        handler.wfile.write(body)

    def _json(self, handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
        self._send(handler, status, json.dumps(payload, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._local_request(handler, require_same_origin=False):
            self._json(handler, 403, {"error": "local_access_only"})
            return
        parsed = urlsplit(handler.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                name = (
                    "setup.html"
                    if self.product_manager is not None and not self.product_manager.configured
                    else "index.html"
                )
                self._serve_static(handler, name, "text/html; charset=utf-8")
            elif parsed.path == "/setup":
                if self.product_manager is None:
                    self._json(handler, 404, {"error": "setup_not_available"})
                else:
                    self._serve_static(handler, "setup.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/product/status":
                self._product_status(handler)
            elif parsed.path.startswith("/api/setup/preview/"):
                self._setup_preview(handler, unquote(parsed.path.removeprefix("/api/setup/preview/")))
            elif parsed.path == "/api/dashboard":
                self._dashboard(handler, parse_qs(parsed.query))
            elif parsed.path.startswith("/evidence/"):
                self._evidence(handler, unquote(parsed.path.removeprefix("/evidence/")))
            else:
                self._json(handler, 404, {"error": "not_found"})
        except ValueError as exc:
            self._json(handler, 400, {"error": str(exc)})
        except Exception:
            self._json(handler, 500, {"error": "dashboard_request_failed"})

    def _serve_static(self, handler: BaseHTTPRequestHandler, name: str, content_type: str) -> None:
        path = STATIC_ROOT / name
        self._send(handler, 200, path.read_bytes(), content_type)

    def _dashboard(self, handler: BaseHTTPRequestHandler, query: dict[str, list[str]]) -> None:
        start, end, selected_date = _local_day_bounds(query.get("date", [None])[0])
        state = query.get("state", [None])[0]
        if state in {"all", ""}:
            state = None
        if state is not None and state not in REVIEW_STATES:
            raise ValueError("unsupported review state")
        with SQLiteEventRepository(self.database_path) as repository:
            all_events = repository.list_events(occurred_after=start, occurred_before=end, limit=1000)
            visible = [event for event in all_events if state is None or event.review_state == state]
            payloads = [_event_payload(repository, event) for event in reversed(visible)]
            accepted_items = sum(
                _quantity(event, repository.list_reviews(event.event_id))
                for event in all_events
                if event.review_state == "accepted"
            )
        states = {name: 0 for name in REVIEW_STATES}
        for event in all_events:
            states[event.review_state] += 1
        self._json(handler, 200, {
            "date": selected_date,
            "summary": {
                "candidate_events": len(all_events),
                "accepted_events": states["accepted"],
                "confirmed_items": accepted_items,
                "pending_review": states["unreviewed"] + states["needs_review"],
                "rejected_events": states["rejected"],
            },
            "system": self._health(),
            "events": payloads,
        })

    def _health(self) -> dict[str, object]:
        try:
            with urllib.request.urlopen(self.health_url, timeout=0.6) as response:
                payload = json.loads(response.read(256 * 1024))
            return {"online": True, "state": "healthy" if payload.get("live") else "attention", "detail": "AI service connected"}
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read(256 * 1024))
                if isinstance(payload, dict) and "live" in payload:
                    return {"online": True, "state": "attention", "detail": "AI service needs attention"}
            except (OSError, ValueError):
                pass
            return {"online": False, "state": "offline", "detail": "AI service is not reachable"}
        except (OSError, ValueError, urllib.error.URLError):
            return {"online": False, "state": "offline", "detail": "AI service is not reachable"}

    def _evidence(self, handler: BaseHTTPRequestHandler, event_id: str) -> None:
        if not event_id or "/" in event_id or "\\" in event_id:
            raise ValueError("invalid event id")
        with SQLiteEventRepository(self.database_path) as repository:
            event = repository.get_event(event_id)
        if event is None or not event.evidence_path:
            self._json(handler, 404, {"error": "evidence_not_found"})
            return
        path = (self.evidence_root / event.evidence_path).resolve()
        if not path.is_relative_to(self.evidence_root) or not path.is_file():
            self._json(handler, 404, {"error": "evidence_not_found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(handler, 200, path.read_bytes(), content_type)

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._local_request(handler, require_same_origin=True):
            self._json(handler, 403, {"error": "local_access_only"})
            return
        parsed = urlsplit(handler.path)
        if parsed.path == "/api/setup/camera-test":
            self._setup_camera_test(handler)
            return
        if parsed.path == "/api/setup/save":
            self._setup_save(handler)
            return
        if parsed.path in {"/api/monitoring/start", "/api/monitoring/stop"}:
            self._monitoring(handler, parsed.path.rsplit("/", 1)[-1])
            return
        if not parsed.path.startswith("/api/events/") or not parsed.path.endswith("/review"):
            self._json(handler, 404, {"error": "not_found"})
            return
        event_id = unquote(parsed.path.removeprefix("/api/events/").removesuffix("/review")).strip("/")
        if not event_id or "/" in event_id or "\\" in event_id:
            self._json(handler, 400, {"error": "invalid event id"})
            return
        try:
            data = self._read_json(handler)
        except ValueError as exc:
            self._json(handler, 400, {"error": str(exc)})
            return
        try:
            decision = data.get("decision")
            if decision not in {"accepted", "rejected", "needs_review"}:
                raise ValueError("unsupported review decision")
            quantity = data.get("quantity")
            if decision == "accepted" and (isinstance(quantity, bool) or not isinstance(quantity, int) or not 1 <= quantity <= 50):
                raise ValueError("accepted quantity must be between 1 and 50")
            notes = data.get("notes")
            if notes is not None and (not isinstance(notes, str) or len(notes) > 500):
                raise ValueError("notes must be at most 500 characters")
            with SQLiteEventRepository(self.database_path) as repository:
                if repository.get_event(event_id) is None:
                    self._json(handler, 404, {"error": "event_not_found"})
                    return
                repository.add_review(ReviewRecord(
                    review_id=str(uuid.uuid4()),
                    event_id=event_id,
                    decision=decision,
                    reviewer_id=getpass.getuser() or "local-operator",
                    reviewed_at=utc_now_iso(),
                    notes=notes,
                    metadata={"application": "local-web-dashboard", **({"corrected_quantity": quantity} if decision == "accepted" else {})},
                ))
                event = repository.get_event(event_id)
                assert event is not None
                payload = _event_payload(repository, event)
            self._json(handler, 200, {"event": payload})
        except ValueError as exc:
            self._json(handler, 400, {"error": str(exc)})

    def _read_json(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        content_type = handler.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if content_type != "application/json" or not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("a small JSON body is required")
        try:
            data = json.loads(handler.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError("request body is not valid JSON") from exc

    @staticmethod
    def _local_request(handler: BaseHTTPRequestHandler, *, require_same_origin: bool) -> bool:
        try:
            if not ipaddress.ip_address(handler.client_address[0]).is_loopback:
                return False
        except ValueError:
            return False
        host = urlsplit("//" + handler.headers.get("Host", "")).hostname
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return False
        origin = handler.headers.get("Origin")
        if require_same_origin and origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme != "http" or parsed_origin.hostname not in {"localhost", "127.0.0.1", "::1"}:
                return False
        return True

    def _product_status(self, handler: BaseHTTPRequestHandler) -> None:
        if self.product_manager is None:
            self._json(handler, 404, {"error": "product_mode_not_enabled"})
            return
        self._json(handler, 200, self.product_manager.status())

    def _setup_camera_test(self, handler: BaseHTTPRequestHandler) -> None:
        if self.product_manager is None:
            self._json(handler, 404, {"error": "setup_not_available"})
            return
        try:
            data = self._read_json(handler)
            source = data.get("source")
            if not isinstance(source, str):
                raise ValueError("camera URL is required")
            result = self.product_manager.test_camera(source)
            self._json(handler, 200, {key: value for key, value in result.items() if key != "jpeg"})
        except (ValueError, RuntimeError) as exc:
            self._json(handler, 400, {"error": str(exc)})

    def _setup_preview(self, handler: BaseHTTPRequestHandler, token: str) -> None:
        if self.product_manager is None or not token or "/" in token or "\\" in token:
            self._json(handler, 404, {"error": "preview_not_found"})
            return
        payload = self.product_manager.preview_jpeg(token)
        if payload is None:
            self._json(handler, 404, {"error": "preview_not_found"})
            return
        self._send(handler, 200, payload, "image/jpeg")

    def _setup_save(self, handler: BaseHTTPRequestHandler) -> None:
        if self.product_manager is None:
            self._json(handler, 404, {"error": "setup_not_available"})
            return
        try:
            data = self._read_json(handler)
            required_strings = ("preview_token", "shop_name", "camera_name", "camera_id")
            if any(not isinstance(data.get(key), str) for key in required_strings):
                raise ValueError("setup details are incomplete")
            result = self.product_manager.save_setup(
                preview_token=data["preview_token"], shop_name=data["shop_name"],
                camera_name=data["camera_name"], camera_id=data["camera_id"],
                pickup_zone=data.get("pickup_zone"), customer_zone=data.get("customer_zone"),
            )
            self._json(handler, 200, result)
        except (ValueError, RuntimeError) as exc:
            self._json(handler, 400, {"error": str(exc)})

    def _monitoring(self, handler: BaseHTTPRequestHandler, action: str) -> None:
        if self.product_manager is None:
            self._json(handler, 404, {"error": "product_mode_not_enabled"})
            return
        try:
            self._read_json(handler)
            result = (
                self.product_manager.start_monitoring()
                if action == "start"
                else self.product_manager.stop_monitoring()
            )
            self._json(handler, 200, result)
        except (ValueError, RuntimeError, OSError) as exc:
            self._json(handler, 400, {"error": str(exc)})


def run_dashboard(
    database_path: str | Path,
    evidence_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
) -> int:
    server = DashboardServer(database_path, evidence_root, host=host, port=port).start()
    address = f"http://{server.address[0]}:{server.address[1]}"
    print(f"Scoop AI dashboard running at {address}")
    if open_browser:
        import webbrowser

        webbrowser.open(address)
    try:
        server.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def run_product(
    product_root: str | Path,
    checkpoint_manifest: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
    open_browser: bool = True,
) -> int:
    manager = ProductManager(product_root, checkpoint_manifest)
    manager.paths.root.mkdir(parents=True, exist_ok=True)
    manager.paths.evidence.mkdir(parents=True, exist_ok=True)
    manager.paths.database.parent.mkdir(parents=True, exist_ok=True)
    server = DashboardServer(
        manager.paths.database,
        manager.paths.evidence,
        host=host,
        port=port,
        product_manager=manager,
    ).start()
    address = f"http://{server.address[0]}:{server.address[1]}"
    print(f"Scoop AI client running at {address}")
    if manager.configured:
        manager.auto_start()
    if open_browser:
        import webbrowser

        webbrowser.open(address)
    try:
        server.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.stop()
    return 0
