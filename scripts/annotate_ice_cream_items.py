"""Small local bounding-box annotator for the ice-cream handover dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


CLASS_NAME = "ice_cream_item"


class ImageCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(720, 560)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.pixmap = QPixmap()
        self.boxes: list[list[int]] = []
        self.image_rect = QRect()
        self.drag_start: QPoint | None = None
        self.drag_end: QPoint | None = None

    def set_image(self, path: Path, boxes: list[list[int]]) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self.pixmap = pixmap
        self.boxes = [list(map(int, box)) for box in boxes]
        self.drag_start = None
        self.drag_end = None
        self.update()
        return True

    def _display_rect(self) -> QRect:
        if self.pixmap.isNull():
            return QRect()
        scaled = self.pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left = (self.width() - scaled.width()) // 2
        top = (self.height() - scaled.height()) // 2
        return QRect(left, top, scaled.width(), scaled.height())

    def _to_image(self, point: QPoint) -> QPoint:
        rect = self._display_rect()
        if rect.isEmpty() or not rect.contains(point):
            return QPoint(-1, -1)
        x = round((point.x() - rect.left()) * self.pixmap.width() / rect.width())
        y = round((point.y() - rect.top()) * self.pixmap.height() / rect.height())
        return QPoint(
            max(0, min(self.pixmap.width() - 1, x)),
            max(0, min(self.pixmap.height() - 1, y)),
        )

    def _to_display_box(self, box: list[int]) -> QRect:
        rect = self._display_rect()
        x, y, width, height = box
        return QRect(
            rect.left() + round(x * rect.width() / self.pixmap.width()),
            rect.top() + round(y * rect.height() / self.pixmap.height()),
            max(1, round(width * rect.width() / self.pixmap.width())),
            max(1, round(height * rect.height() / self.pixmap.height())),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15171b"))
        if self.pixmap.isNull():
            return
        self.image_rect = self._display_rect()
        painter.drawPixmap(self.image_rect, self.pixmap)
        painter.setPen(QPen(QColor("#00ff66"), 3))
        for box in self.boxes:
            display_box = self._to_display_box(box)
            painter.drawRect(display_box)
            painter.fillRect(
                QRect(display_box.left(), max(0, display_box.top() - 22), 150, 22),
                QColor(0, 0, 0, 180),
            )
            painter.drawText(display_box.left() + 4, max(17, display_box.top() - 5), CLASS_NAME)
        if self.drag_start is not None and self.drag_end is not None:
            painter.setPen(QPen(QColor("#ffcc00"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.drag_start, self.drag_end).normalized())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._to_image(event.position().toPoint())
            if point.x() >= 0:
                self.drag_start = event.position().toPoint()
                self.drag_end = self.drag_start
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.delete_last_box()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.drag_start is not None:
            point = event.position().toPoint()
            rect = self._display_rect()
            self.drag_end = QPoint(
                max(rect.left(), min(rect.right(), point.x())),
                max(rect.top(), min(rect.bottom(), point.y())),
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.drag_start is None:
            return
        start = self._to_image(self.drag_start)
        end = self._to_image(event.position().toPoint())
        self.drag_start = None
        self.drag_end = None
        if start.x() < 0 or end.x() < 0:
            self.update()
            return
        left, right = sorted((start.x(), end.x()))
        top, bottom = sorted((start.y(), end.y()))
        if right - left >= 5 and bottom - top >= 5:
            self.boxes.append([left, top, right - left, bottom - top])
        self.update()

    def delete_last_box(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self.update()


class AnnotatorWindow(QMainWindow):
    def __init__(
        self,
        dataset_root: Path,
        output_path: Path,
        review_status: str | None = None,
        review_paths: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root.resolve()
        self.output_path = output_path.resolve()
        self.records = self._load_manifest()
        self.annotations = self._load_annotations()
        self.dirty_keys: set[str] = set()
        if review_status:
            self.records = [
                record
                for record in self.records
                if self.annotations.get(str(record["image"]), {}).get("status") == review_status
            ]
            if not self.records:
                raise ValueError(f"No frames have status {review_status!r}")
        if review_paths is not None:
            self.records = [record for record in self.records if str(record["image"]) in review_paths]
            if not self.records:
                raise ValueError("The review list does not contain any dataset frames")
        self.index = self._resume_index()

        title = "Ice Cream Item Annotation"
        if review_status:
            title += f" — Review {review_status}"
        elif review_paths is not None:
            title += " — Focused review"
        self.setWindowTitle(title)
        self.resize(1100, 850)
        self.canvas = ImageCanvas()
        self.progress = QLabel()
        self.filename = QLabel()
        self.help_text = QLabel(
            "LEFT-DRAG a box around each active item • Right-click/Backspace: delete last box • "
            "Enter: Save & Next • N: Negative • K: Skip • ←/→: Previous/Next"
        )
        self.help_text.setWordWrap(True)
        self.help_text.setStyleSheet("color: #d6d8dc; padding: 7px;")
        self.progress.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.filename.setStyleSheet("color: #aeb4be;")

        previous = QPushButton("← Previous")
        delete = QPushButton("Delete last box")
        skip = QPushButton("Skip (K)")
        negative = QPushButton("No active item (N)")
        save = QPushButton("Save boxes & Next (Enter)")
        save.setStyleSheet("font-weight: bold; background: #18794e; color: white; padding: 8px;")
        negative.setStyleSheet("background: #334155; color: white; padding: 8px;")

        previous.clicked.connect(self.previous_image)
        delete.clicked.connect(self.canvas.delete_last_box)
        skip.clicked.connect(self.mark_skipped)
        negative.clicked.connect(self.mark_negative)
        save.clicked.connect(self.mark_annotated)

        buttons = QHBoxLayout()
        for button in (previous, delete, skip, negative, save):
            buttons.addWidget(button)

        layout = QVBoxLayout()
        layout.addWidget(self.progress)
        layout.addWidget(self.filename)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.help_text)
        layout.addLayout(buttons)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self.mark_annotated)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.mark_annotated)
        QShortcut(QKeySequence("N"), self, activated=self.mark_negative)
        QShortcut(QKeySequence("K"), self, activated=self.mark_skipped)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self.canvas.delete_last_box)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self.previous_image)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self.next_image)
        self.load_current()

    def _load_manifest(self) -> list[dict[str, object]]:
        manifest = self.dataset_root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        records.sort(key=lambda item: (str(item.get("source_session", "")), float(item.get("timestamp_seconds", 0))))
        if not records:
            raise ValueError("Dataset manifest is empty")
        return records

    def _load_annotations(self) -> dict[str, dict[str, object]]:
        if not self.output_path.is_file():
            return {}
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        return dict(payload.get("items", {}))

    def _resume_index(self) -> int:
        for index, record in enumerate(self.records):
            if str(record["image"]) not in self.annotations:
                return index
        return 0

    def _current_key(self) -> str:
        return str(self.records[self.index]["image"])

    def load_current(self) -> None:
        record = self.records[self.index]
        key = self._current_key()
        saved = self.annotations.get(key, {})
        image_path = self.dataset_root / Path(key)
        if not self.canvas.set_image(image_path, list(saved.get("boxes", []))):
            QMessageBox.critical(self, "Image error", f"Could not open {image_path}")
            return
        reviewed = sum(1 for item in self.annotations.values() if item.get("status") in {"annotated", "negative", "skipped"})
        positives = sum(1 for item in self.annotations.values() if item.get("status") == "annotated")
        negatives = sum(1 for item in self.annotations.values() if item.get("status") == "negative")
        self.progress.setText(
            f"Frame {self.index + 1}/{len(self.records)}   |   Reviewed {reviewed}   |   "
            f"Positive {positives}   |   Negative {negatives}"
        )
        self.filename.setText(
            f"{record.get('source_video')}  •  {float(record.get('timestamp_seconds', 0)):.1f}s  •  "
            f"status: {saved.get('status', 'unreviewed')}"
        )

    def _save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.is_file() and not self.dirty_keys:
            return
        merged_items = dict(self.annotations)
        if self.output_path.is_file():
            disk_payload = json.loads(self.output_path.read_text(encoding="utf-8"))
            merged_items = dict(disk_payload.get("items", {}))
            for key in self.dirty_keys:
                merged_items[key] = self.annotations[key]
        payload = {
            "schema_version": 1,
            "class_names": [CLASS_NAME],
            "dataset_root": str(self.dataset_root),
            "source_manifest": "manifest.jsonl",
            "items": merged_items,
        }
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.output_path)
        self.annotations = merged_items
        self.dirty_keys.clear()

    def _record(self, status: str, boxes: list[list[int]]) -> None:
        record = self.records[self.index]
        key = self._current_key()
        self.annotations[key] = {
            "status": status,
            "boxes": boxes,
            "width": int(record["width"]),
            "height": int(record["height"]),
            "source_session": record.get("source_session"),
            "timestamp_seconds": record.get("timestamp_seconds"),
        }
        self.dirty_keys.add(key)
        self._save()
        self.next_image()

    def mark_annotated(self) -> None:
        if not self.canvas.boxes:
            self.statusBar().showMessage("Draw at least one box around the active ice-cream item first.", 4000)
            return
        self._record("annotated", self.canvas.boxes)

    def mark_negative(self) -> None:
        if self.canvas.boxes:
            self.statusBar().showMessage("Delete the boxes before marking this frame negative.", 4000)
            return
        self._record("negative", [])

    def mark_skipped(self) -> None:
        self._record("skipped", [])

    def previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def next_image(self) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
            self.load_current()
        else:
            self.statusBar().showMessage("All frames are done.", 5000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--review-status", choices=("annotated", "negative", "skipped"))
    review.add_argument("--review-list", type=Path, help="Text file containing one manifest image path per line")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    review_paths = None
    if args.review_list:
        review_paths = {
            line.strip().replace("\\", "/")
            for line in args.review_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    window = AnnotatorWindow(args.dataset, args.output, args.review_status, review_paths)
    if args.smoke_test:
        print(f"Loaded {len(window.records)} frames; first={window._current_key()}")
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
