"""Native desktop window that hosts the local client without a web browser.

The dashboard HTTP server still runs, but it binds a loopback port inside this
process and is rendered by an embedded Chromium view. Nothing is exposed to the
network, and the operator never sees a browser, address bar, or URL.
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import sys
from pathlib import Path

from ..compute import describe_compute
from ..dashboard import DashboardServer, ProductManager

WINDOW_TITLE = "Scoop AI"
ORGANISATION = "ScoopAI"
DEFAULT_PORT = 8090
PORT_SCAN_LIMIT = 12

LOGGER = logging.getLogger("scoop-ai.desktop")
_CONSOLE_LEVELS = {
    "InfoMessageLevel": logging.INFO,
    "WarningMessageLevel": logging.WARNING,
    "ErrorMessageLevel": logging.ERROR,
}


class DesktopLaunchError(RuntimeError):
    """Raised when the desktop shell cannot start on this machine."""


def _in_use(host: str, port: int, timeout: float = 0.4) -> bool:
    """Report whether something already answers on this loopback port.

    Binding cannot be relied on to detect this. Python's HTTP server sets
    SO_REUSEADDR, which on Windows lets a second socket take a port another
    process is already listening on: both binds succeed and connections are
    split between them unpredictably. Connecting first is what actually
    distinguishes a live server from a free port, and unlike an exclusive bind
    it still treats a lingering TIME_WAIT socket as free.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _bind_dashboard(
    manager: ProductManager, host: str, port: int
) -> DashboardServer:
    """Start the dashboard, stepping past ports another program already holds."""

    last_error: OSError | None = None
    for candidate in range(port, port + PORT_SCAN_LIMIT):
        if _in_use(host, candidate):
            continue
        try:
            return DashboardServer(
                manager.paths.database,
                manager.paths.evidence,
                host=host,
                port=candidate,
                product_manager=manager,
            ).start()
        except OSError as exc:
            if exc.errno not in {errno.EADDRINUSE, errno.EACCES}:
                raise
            last_error = exc
    raise DesktopLaunchError(
        f"No free loopback port between {port} and {port + PORT_SCAN_LIMIT - 1}."
    ) from last_error


def run_desktop(
    product_root: str | Path,
    checkpoint_manifest: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    software_rendering: bool = False,
    start_hidden: bool = False,
) -> int:
    """Run the client as a desktop application and return a process exit code."""

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )

    if software_rendering:
        # Shop machines without a usable display driver (and RDP sessions) cannot
        # composite on the GPU; Chromium otherwise renders a blank window.
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} --disable-gpu".strip()

    try:
        # Importing the WebEngine module before QApplication exists is required
        # for Chromium to share the application's OpenGL context.
        from PySide6.QtWebEngineCore import (
            QWebEnginePage,
            QWebEngineProfile,
            QWebEngineSettings,
        )
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtCore import QUrl, Qt, QTimer
        from PySide6.QtGui import (
            QAction,
            QColor,
            QGuiApplication,
            QIcon,
            QKeySequence,
            QPainter,
            QPainterPath,
            QPalette,
            QPixmap,
        )
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
        from PySide6.QtWidgets import (
            QApplication,
            QMainWindow,
            QMenu,
            QMessageBox,
            QSystemTrayIcon,
        )
    except ImportError as exc:
        raise DesktopLaunchError(
            "The desktop window needs PySide6 with QtWebEngine. "
            "Reinstall Scoop AI, or start the browser dashboard instead."
        ) from exc

    def _dark_desktop() -> bool:
        """Follow the operating system's light/dark preference, as the page does."""

        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
        window = QGuiApplication.palette().color(QPalette.ColorRole.Window)
        return window.lightness() < 128

    def _page_background() -> QColor:
        # Matches --bg in static/app.css for both themes.
        return QColor("#14181a") if _dark_desktop() else QColor("#f7f8f7")

    def _app_icon() -> QIcon:
        """Build the rounded-square mark used by the taskbar, tray, and window.

        Rendering several sizes keeps the small taskbar and tray variants sharp
        instead of letting Qt downscale one large bitmap.
        """

        icon = QIcon()
        for size in (16, 24, 32, 48, 64, 128, 256):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            path = QPainterPath()
            inset = size * 0.06
            path.addRoundedRect(
                inset, inset, size - 2 * inset, size - 2 * inset, size * 0.24, size * 0.24
            )
            painter.fillPath(path, QColor("#0f7a53"))
            painter.setPen(QColor("#ffffff"))
            font = painter.font()
            font.setPixelSize(max(8, int(size * 0.56)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S")
            painter.end()
            icon.addPixmap(pixmap)
        return icon

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName(WINDOW_TITLE)
    application.setOrganizationName(ORGANISATION)
    # Closing the last window hides to the tray; the process must survive it so
    # monitoring keeps running.
    application.setQuitOnLastWindowClosed(False)

    # A second launch (desktop shortcut, startup entry) must raise the running
    # window rather than fight it for the loopback port.
    guard_name = f"scoop-ai-desktop-{socket.gethostname()}-{port}"
    probe = QLocalSocket()
    probe.connectToServer(guard_name)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return 0
    QLocalServer.removeServer(guard_name)
    guard = QLocalServer()
    guard.listen(guard_name)

    manager = ProductManager(product_root, checkpoint_manifest)
    manager.paths.root.mkdir(parents=True, exist_ok=True)
    manager.paths.evidence.mkdir(parents=True, exist_ok=True)
    manager.paths.database.parent.mkdir(parents=True, exist_ok=True)
    server = _bind_dashboard(manager, host, port)
    address = f"http://{server.address[0]}:{server.address[1]}"
    compute = describe_compute()

    profile = QWebEngineProfile("scoop-ai-desktop", application)
    profile.setPersistentStoragePath(str(manager.paths.root / "desktop" / "web"))
    profile.setCachePath(str(manager.paths.root / "desktop" / "cache"))

    class LocalOnlyPage(QWebEnginePage):
        """Refuse every navigation that is not the local client itself.

        The client only ever serves loopback, so a future page bug cannot steer
        the shell to an external site or leak the evidence it is displaying.
        """

        def acceptNavigationRequest(  # noqa: N802 - Qt override
            self, url: QUrl, _type: object, _is_main_frame: bool
        ) -> bool:
            return url.scheme() == "http" and url.host() in {
                "127.0.0.1",
                "localhost",
                "::1",
            }

        def javaScriptConsoleMessage(  # noqa: N802 - Qt override
            self, level: object, message: str, line: int, source: str
        ) -> None:
            # Without this the page fails silently: a blocked request or a script
            # error leaves no trace anywhere the operator or a log can reach.
            LOGGER.log(
                _CONSOLE_LEVELS.get(getattr(level, "name", str(level)), logging.INFO),
                "page: %s (%s:%s)",
                message,
                source or "inline",
                line,
            )

    class ClientView(QWebEngineView):
        """A view pinned to the local client; external links never open here."""

        def createWindow(self, _type: object) -> None:  # noqa: N802 - Qt override
            return None

    class DesktopWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(WINDOW_TITLE)
            self.setWindowIcon(_app_icon())
            self.resize(1280, 860)
            self.setMinimumSize(900, 600)
            self._quitting = False

            # The page owns the entire window. A native toolbar on top of a styled
            # web application reads as two competing navigation systems, so the
            # sidebar in the page is the only navigation.
            self.view = ClientView(self)
            self.view.setPage(LocalOnlyPage(profile, self.view))
            settings = self.view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
            )
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False
            )
            # Chromium paints white before the first frame, which flashes against
            # a dark page. Painting the page's own background removes it.
            self.view.page().setBackgroundColor(_page_background())
            self.setCentralWidget(self.view)

            self._devtools: QWebEngineView | None = None

            # Keyboard shortcuts an operator expects, without visible chrome.
            for sequence, slot in (
                ("F5", self.view.reload),
                ("Ctrl+R", self.view.reload),
                ("Ctrl+1", lambda: self.go("/")),
                ("Ctrl+2", lambda: self.go("/camera")),
                ("Ctrl+3", lambda: self.go("/system")),
                ("F12", self.toggle_devtools),
                ("Ctrl+Shift+I", self.toggle_devtools),
            ):
                action = QAction(self)
                action.setShortcut(QKeySequence(sequence))
                action.triggered.connect(slot)
                self.addAction(action)

            self.view.loadFinished.connect(self._load_finished)
            self.view.load(QUrl(address))

        def _load_finished(self, ok: bool) -> None:
            if not ok:
                LOGGER.error("page failed to load: %s", self.view.url().toString())

        def toggle_devtools(self) -> None:
            """Open Chromium's inspector in its own window.

            The shell has no menu bar, so without an explicit shortcut there is no
            way to inspect a misbehaving page on a customer machine.
            """

            if self._devtools is not None and self._devtools.isVisible():
                self._devtools.close()
                return
            if self._devtools is None:
                self._devtools = QWebEngineView()
                self._devtools.setWindowTitle("Scoop AI - developer tools")
                self._devtools.resize(1100, 700)
                self.view.page().setDevToolsPage(self._devtools.page())
            self._devtools.show()
            self._devtools.raise_()

        def go(self, path: str) -> None:
            self.view.load(QUrl(address + path))

        def request_quit(self) -> None:
            if manager.monitoring:
                confirm = QMessageBox.question(
                    self,
                    "Quit Scoop AI?",
                    "Monitoring is running and will stop. Quit anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if confirm != QMessageBox.StandardButton.Yes:
                    return
            self._quitting = True
            self.close()
            application.quit()

        def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
            if self._quitting:
                event.accept()
                return
            # Hiding instead of exiting is what makes background monitoring work
            # the way an operator expects from a tray application.
            event.ignore()
            self.hide()
            tray.showMessage(
                WINDOW_TITLE,
                "Scoop AI keeps monitoring in the background. Use the tray icon to reopen it.",
                _app_icon(),
                4000,
            )

        def reveal(self) -> None:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    window = DesktopWindow()

    tray = QSystemTrayIcon(_app_icon(), application)
    tray.setToolTip(f"{WINDOW_TITLE} — {compute.headline}")
    menu = QMenu()
    open_action = menu.addAction("Open Scoop AI")
    open_action.triggered.connect(window.reveal)
    monitor_action = menu.addAction("Start monitoring")
    menu.addSeparator()
    quit_action = menu.addAction("Quit")
    quit_action.triggered.connect(window.request_quit)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: window.reveal()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )
    tray.show()

    def toggle_monitoring() -> None:
        try:
            if manager.monitoring:
                manager.stop_monitoring()
            else:
                manager.start_monitoring()
        except (ValueError, RuntimeError, OSError) as exc:
            QMessageBox.warning(window, "Monitoring", str(exc))
        refresh_monitor_action()

    def refresh_monitor_action() -> None:
        running = manager.monitoring
        monitor_action.setText("Stop monitoring" if running else "Start monitoring")
        monitor_action.setEnabled(manager.configured)
        suffix = "monitoring" if running else "idle"
        tray.setToolTip(f"{WINDOW_TITLE} — {compute.headline} — {suffix}")

    monitor_action.triggered.connect(toggle_monitoring)
    refresh_monitor_action()

    poll = QTimer(application)
    poll.timeout.connect(refresh_monitor_action)
    poll.start(4000)

    guard.newConnection.connect(lambda: (guard.nextPendingConnection(), window.reveal()))

    if manager.configured:
        manager.auto_start()
        refresh_monitor_action()
    if not start_hidden:
        window.show()

    try:
        return int(application.exec())
    finally:
        poll.stop()
        manager.close()
        server.stop()
        guard.close()
