"""The desktop shell must bind a private port and refuse off-box navigation."""

from __future__ import annotations

import errno
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoop_ai.desktop.app import PORT_SCAN_LIMIT, DesktopLaunchError, _bind_dashboard  # noqa: E402


class _StubServer:
    def __init__(self, port: int) -> None:
        self.port = port

    def start(self) -> "_StubServer":
        return self


class BindDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = mock.Mock()
        self.manager.paths.database = Path("events.sqlite3")
        self.manager.paths.evidence = Path("evidence")
        # Never probe the real loopback interface; a developer machine may well
        # have the installed client listening on the default port.
        free = mock.patch("scoop_ai.desktop.app._in_use", return_value=False)
        self.free_ports = free.start()
        self.addCleanup(free.stop)

    def test_uses_the_requested_port_when_it_is_free(self) -> None:
        with mock.patch(
            "scoop_ai.desktop.app.DashboardServer",
            side_effect=lambda *a, port, **k: _StubServer(port),
        ):
            server = _bind_dashboard(self.manager, "127.0.0.1", 8090)
        self.assertEqual(server.port, 8090)

    def test_a_live_server_is_stepped_over_even_though_binding_would_succeed(self) -> None:
        """Windows lets SO_REUSEADDR steal a listening port, so probe first."""

        with mock.patch(
            "scoop_ai.desktop.app.DashboardServer",
            side_effect=lambda *a, port, **k: _StubServer(port),
        ), mock.patch(
            "scoop_ai.desktop.app._in_use", side_effect=lambda _host, port: port == 8090
        ):
            server = _bind_dashboard(self.manager, "127.0.0.1", 8090)
        self.assertEqual(server.port, 8091)

    def test_every_port_answering_is_reported_as_a_launch_failure(self) -> None:
        with mock.patch(
            "scoop_ai.desktop.app.DashboardServer",
            side_effect=lambda *a, port, **k: _StubServer(port),
        ), mock.patch("scoop_ai.desktop.app._in_use", return_value=True):
            with self.assertRaises(DesktopLaunchError):
                _bind_dashboard(self.manager, "127.0.0.1", 8090)

    def test_steps_past_a_port_another_program_holds(self) -> None:
        def build(*_args, port: int, **_kwargs):
            if port < 8092:
                raise OSError(errno.EADDRINUSE, "address in use")
            return _StubServer(port)

        with mock.patch("scoop_ai.desktop.app.DashboardServer", side_effect=build):
            server = _bind_dashboard(self.manager, "127.0.0.1", 8090)
        self.assertEqual(server.port, 8092)

    def test_gives_up_with_a_clear_message_when_the_range_is_full(self) -> None:
        def build(*_args, **_kwargs):
            raise OSError(errno.EADDRINUSE, "address in use")

        with mock.patch("scoop_ai.desktop.app.DashboardServer", side_effect=build):
            with self.assertRaises(DesktopLaunchError) as caught:
                _bind_dashboard(self.manager, "127.0.0.1", 8090)
        self.assertIn(str(8090 + PORT_SCAN_LIMIT - 1), str(caught.exception))

    def test_an_unrelated_socket_error_is_not_retried(self) -> None:
        attempts = []

        def build(*_args, port: int, **_kwargs):
            attempts.append(port)
            raise OSError(errno.EAFNOSUPPORT, "address family not supported")

        with mock.patch("scoop_ai.desktop.app.DashboardServer", side_effect=build):
            with self.assertRaises(OSError):
                _bind_dashboard(self.manager, "127.0.0.1", 8090)
        self.assertEqual(attempts, [8090])


class ClientEntryPointTests(unittest.TestCase):
    """The installed shortcut opens a window; support flags keep the old paths."""

    def test_default_launch_opens_the_desktop_window(self) -> None:
        from scoop_ai import client

        with mock.patch("scoop_ai.desktop.run_desktop", return_value=0) as run_desktop:
            self.assertEqual(client.main([]), 0)
        run_desktop.assert_called_once()
        self.assertFalse(run_desktop.call_args.kwargs["start_hidden"])

    def test_start_hidden_is_forwarded(self) -> None:
        from scoop_ai import client

        with mock.patch("scoop_ai.desktop.run_desktop", return_value=0) as run_desktop:
            client.main(["--start-hidden"])
        self.assertTrue(run_desktop.call_args.kwargs["start_hidden"])

    def test_browser_flag_never_builds_a_window(self) -> None:
        from scoop_ai import client

        with mock.patch("scoop_ai.desktop.run_desktop") as run_desktop, \
                mock.patch.object(client, "_reachable", return_value=True), \
                mock.patch.object(client.webbrowser, "open") as open_browser:
            self.assertEqual(client.main(["--browser"]), 0)
        run_desktop.assert_not_called()
        open_browser.assert_called_once()

    def test_a_machine_without_qt_falls_back_to_the_browser(self) -> None:
        from scoop_ai import client

        with mock.patch(
            "scoop_ai.desktop.run_desktop",
            side_effect=DesktopLaunchError("no QtWebEngine here."),
        ), mock.patch.object(client, "_reachable", return_value=True), \
                mock.patch.object(client.webbrowser, "open") as open_browser:
            self.assertEqual(client.main([]), 0)
        open_browser.assert_called_once()


if __name__ == "__main__":
    unittest.main()
