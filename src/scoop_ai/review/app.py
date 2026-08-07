"""PySide6 desktop queue for immutable scoop-event review decisions."""

from __future__ import annotations

import getpass
import sys
import uuid
from pathlib import Path

from ..storage import GroundTruthRecord, ReviewRecord, SQLiteEventRepository
from ..storage.database import utc_now_iso


def run_review_app(
    database_path: str | Path,
    evidence_root: str | Path | None = None,
) -> int:
    """Launch the local reviewer; PySide6 is imported only for this command."""

    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QInputDialog,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print(
            "PySide6 is required for the review app. Install the 'review' extra.",
            file=sys.stderr,
        )
        return 2

    repository = SQLiteEventRepository(database_path)
    root = Path(evidence_root or Path(database_path).parent / "evidence").resolve()
    reviewer_id = getpass.getuser() or "local-operator"

    class ReviewWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Scoop AI — Silent Pilot Review")
            self.resize(1200, 720)
            self._events = []

            central = QWidget()
            layout = QVBoxLayout(central)
            controls = QHBoxLayout()
            controls.addWidget(QLabel("Review state:"))
            self.state_filter = QComboBox()
            self.state_filter.addItems(
                ["unreviewed", "needs_review", "accepted", "rejected", "all"]
            )
            controls.addWidget(self.state_filter)
            refresh = QPushButton("Refresh")
            refresh.clicked.connect(self.refresh)
            controls.addWidget(refresh)
            controls.addStretch(1)
            layout.addLayout(controls)

            self.table = QTableWidget(0, 8)
            self.table.setHorizontalHeaderLabels(
                [
                    "Time",
                    "Camera",
                    "Event",
                    "Container",
                    "Scoop",
                    "Confidence",
                    "State",
                    "Evidence",
                ]
            )
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            self.table.setSelectionMode(QTableWidget.SingleSelection)
            self.table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.table)

            decisions = QHBoxLayout()
            for label, decision in (
                ("Accept", "accepted"),
                ("Reject", "rejected"),
                ("Needs review", "needs_review"),
            ):
                button = QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, value=decision: self.review(value)
                )
                decisions.addWidget(button)
            open_evidence = QPushButton("Open evidence")
            open_evidence.clicked.connect(self.open_evidence)
            decisions.addWidget(open_evidence)
            correct_container = QPushButton("Correct container")
            correct_container.clicked.connect(self.correct_container)
            decisions.addWidget(correct_container)
            ground_truth = QPushButton("Record ground truth")
            ground_truth.clicked.connect(self.record_ground_truth)
            decisions.addWidget(ground_truth)
            decisions.addStretch(1)
            layout.addLayout(decisions)
            self.setCentralWidget(central)

            self.state_filter.currentTextChanged.connect(self.refresh)
            self.refresh()

        def selected_event(self):
            row = self.table.currentRow()
            return self._events[row] if 0 <= row < len(self._events) else None

        def refresh(self, *_: object) -> None:
            selected_state = self.state_filter.currentText()
            state = None if selected_state == "all" else selected_state
            self._events = repository.list_events(review_state=state, limit=1000)
            self.table.setRowCount(len(self._events))
            for row, event in enumerate(self._events):
                values = (
                    event.occurred_at,
                    event.camera_id,
                    event.event_type,
                    "" if event.container_track_id is None else str(event.container_track_id),
                    "" if event.scoop_track_id is None else str(event.scoop_track_id),
                    "" if event.confidence is None else f"{event.confidence:.1%}",
                    event.review_state,
                    event.evidence_path or "",
                )
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(value))
            self.table.resizeColumnsToContents()

        def review(self, decision: str) -> None:
            event = self.selected_event()
            if event is None:
                QMessageBox.information(self, "Review", "Select one event first.")
                return
            repository.add_review(
                ReviewRecord(
                    review_id=str(uuid.uuid4()),
                    event_id=event.event_id,
                    decision=decision,
                    reviewer_id=reviewer_id,
                    reviewed_at=utc_now_iso(),
                    metadata={"application": "pyside6-review"},
                )
            )
            self.refresh()

        def open_evidence(self) -> None:
            event = self.selected_event()
            if event is None or not event.evidence_path:
                QMessageBox.information(self, "Evidence", "No evidence is attached.")
                return
            path = (root / event.evidence_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                QMessageBox.warning(self, "Evidence", "Evidence file is unavailable.")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

        def correct_container(self) -> None:
            event = self.selected_event()
            if event is None:
                QMessageBox.information(self, "Correction", "Select one event first.")
                return
            value, accepted = QInputDialog.getInt(
                self,
                "Correct container",
                "Container track ID:",
                value=event.container_track_id or 0,
                minValue=0,
            )
            if not accepted:
                return
            repository.add_review(
                ReviewRecord(
                    review_id=str(uuid.uuid4()),
                    event_id=event.event_id,
                    decision="accepted",
                    reviewer_id=reviewer_id,
                    reviewed_at=utc_now_iso(),
                    notes="container corrected by reviewer",
                    metadata={"corrected_container_track_id": value},
                )
            )
            self.refresh()

        def record_ground_truth(self) -> None:
            event = self.selected_event()
            if event is None:
                QMessageBox.information(self, "Ground truth", "Select one event first.")
                return
            repository.add_ground_truth(
                GroundTruthRecord(
                    ground_truth_id=str(uuid.uuid4()),
                    session_id=event.session_id,
                    camera_id=event.camera_id,
                    occurred_at=event.occurred_at,
                    is_completed_scoop=True,
                    reviewer_id=reviewer_id,
                    container_track_id=event.container_track_id,
                    metadata={"source_event_id": event.event_id},
                )
            )
            QMessageBox.information(self, "Ground truth", "Ground-truth scoop recorded.")

        def closeEvent(self, event) -> None:  # noqa: N802, ANN001
            repository.close()
            super().closeEvent(event)

    application = QApplication.instance() or QApplication(sys.argv)
    window = ReviewWindow()
    window.show()
    try:
        return int(application.exec())
    finally:
        repository.close()
