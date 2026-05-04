from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yolo_trainer.annotations import (
    CANONICAL_CLASSES,
    AnnotationStore,
    PixelBox,
    ReviewStateStore,
)
from yolo_trainer.image_import import ImportResult, import_stem_zc_images
from yolo_trainer.project import (
    ImportedImage,
    InvalidProjectError,
    YOLOTrainingProject,
    create_project,
    open_project,
)


APP_NAME = "YOLO Trainer"
MINIMUM_WINDOW_SIZE = QSize(960, 640)


class AnnotationImageLabel(QLabel):
    box_drawn = Signal(int, int, int, int)

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._drag_start: QPoint | None = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            drag_end = event.position().toPoint()
            x_min = min(self._drag_start.x(), drag_end.x())
            y_min = min(self._drag_start.y(), drag_end.y())
            x_max = max(self._drag_start.x(), drag_end.x())
            y_max = max(self._drag_start.y(), drag_end.y())
            self._drag_start = None
            if x_max > x_min and y_max > y_min:
                self.box_drawn.emit(x_min, y_min, x_max, y_max)
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        dm3_reader: Callable[[Path], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self._current_project: YOLOTrainingProject | None = None
        self._selected_image: ImportedImage | None = None
        self._annotation_image_text = "No image selected for annotation."
        self._dm3_reader = dm3_reader

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(MINIMUM_WINDOW_SIZE)

        self._status_label = QLabel("No YOLO Training Project loaded.")
        self._status_label.setObjectName("projectStatusLabel")

        self._queue_label = QLabel("Create or open a project to start.")
        self._queue_label.setObjectName("imageQueueLabel")

        self._error_label = QLabel("")
        self._error_label.setObjectName("projectErrorLabel")

        self._class_selector = QComboBox()
        self._class_selector.setObjectName("annotationClassSelector")
        self._class_selector.addItems(CANONICAL_CLASSES)

        self._annotation_image_label = AnnotationImageLabel(
            "No image selected for annotation."
        )
        self._annotation_image_label.setObjectName("annotationImageLabel")
        self._annotation_image_label.box_drawn.connect(self._draw_selected_class_box)

        self._annotation_status_label = QLabel("No boxes saved.")
        self._annotation_status_label.setObjectName("annotationStatusLabel")

        self._review_state_label = QLabel("Review state: none")
        self._review_state_label.setObjectName("reviewStateLabel")

        self._review_progress_label = QLabel(
            "Review progress: 0/0 reviewed, 0 unreviewed"
        )
        self._review_progress_label.setObjectName("reviewProgressLabel")

        create_button = QPushButton("Create Project")
        create_button.setObjectName("createProjectButton")
        create_button.clicked.connect(self._choose_project_to_create)

        open_button = QPushButton("Open Project")
        open_button.setObjectName("openProjectButton")
        open_button.clicked.connect(self._choose_project_to_open)

        import_button = QPushButton("Import Images")
        import_button.setObjectName("importImagesButton")
        import_button.clicked.connect(self._choose_images_to_import)

        reviewed_empty_button = QPushButton("Mark Reviewed Empty")
        reviewed_empty_button.setObjectName("markReviewedEmptyButton")
        reviewed_empty_button.clicked.connect(self.mark_selected_image_reviewed_empty)

        layout = QVBoxLayout()
        layout.addWidget(create_button)
        layout.addWidget(open_button)
        layout.addWidget(import_button)
        layout.addWidget(self._status_label)
        layout.addWidget(self._queue_label)
        layout.addWidget(self._error_label)
        layout.addWidget(self._class_selector)
        layout.addWidget(self._annotation_image_label)
        layout.addWidget(self._annotation_status_label)
        layout.addWidget(self._review_state_label)
        layout.addWidget(self._review_progress_label)
        layout.addWidget(reviewed_empty_button)
        layout.addStretch()

        root = QWidget()
        root.setLayout(layout)
        root.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(root)

    def create_project_at(self, path: Path | str, name: str | None = None) -> None:
        try:
            self._show_project(create_project(path, name=name))
        except OSError as error:
            self._show_project_error(f"Could not create YOLO Training Project: {error}")

    def import_images(self, paths: list[Path] | tuple[Path, ...]) -> ImportResult | None:
        if self._current_project is None:
            self._show_project_error(
                "Open a YOLO Training Project before importing images."
            )
            return None

        result = import_stem_zc_images(
            self._current_project,
            paths,
            dm3_reader=self._dm3_reader,
        )
        self._show_project(open_project(self._current_project.path))
        if result.failed:
            self._error_label.setText(_format_import_failures(result))
        return result

    def select_image(self, image_id: str) -> None:
        if self._current_project is None:
            self._show_project_error(
                "Open a YOLO Training Project before selecting images."
            )
            return

        for imported_image in self._current_project.imported_images:
            if imported_image.image_id == image_id:
                self._show_annotation_image(imported_image)
                return

        self._show_project_error(f"Imported image not found: {image_id}")

    def selected_image_id(self) -> str | None:
        if self._selected_image is None:
            return None
        return self._selected_image.image_id

    def draw_annotation_box(
        self,
        *,
        class_name: str,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> None:
        if self._current_project is None or self._selected_image is None:
            self._show_project_error("Select an imported image before drawing boxes.")
            return

        AnnotationStore(self._current_project).add_box(
            self._selected_image,
            class_name=class_name,
            pixel_box=PixelBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        )
        self._refresh_annotation_status()

    def undo_last_annotation(self) -> None:
        if self._current_project is None or self._selected_image is None:
            return
        AnnotationStore(self._current_project).undo_last(self._selected_image)
        self._refresh_annotation_status()

    def delete_annotation(self, *, index: int) -> None:
        if self._current_project is None or self._selected_image is None:
            return
        AnnotationStore(self._current_project).delete_box(
            self._selected_image,
            index=index,
        )
        self._refresh_annotation_status()

    def mark_selected_image_reviewed_empty(self) -> None:
        if self._current_project is None or self._selected_image is None:
            return
        AnnotationStore(self._current_project).save(self._selected_image, [])
        ReviewStateStore(self._current_project).mark_reviewed_empty(
            self._selected_image
        )
        self._refresh_annotation_status()

    def open_project_at(self, path: Path | str) -> None:
        try:
            self._show_project(open_project(path))
        except InvalidProjectError as error:
            self._current_project = None
            self._show_project_error(str(error))

    def current_project_name(self) -> str | None:
        if self._current_project is None:
            return None
        return self._current_project.name

    def project_status_text(self) -> str:
        return self._status_label.text()

    def image_queue_text(self) -> str:
        return self._queue_label.text()

    def project_error_text(self) -> str:
        return self._error_label.text()

    def annotation_image_text(self) -> str:
        return self._annotation_image_text

    def annotation_status_text(self) -> str:
        return self._annotation_status_label.text()

    def review_state_text(self) -> str:
        return self._review_state_label.text()

    def review_progress_text(self) -> str:
        return self._review_progress_label.text()

    def annotation_canvas(self) -> AnnotationImageLabel:
        return self._annotation_image_label

    def _choose_project_to_create(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Create YOLO Training Project",
        )
        if path:
            self.create_project_at(path)

    def _choose_project_to_open(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Open YOLO Training Project",
        )
        if path:
            self.open_project_at(path)

    def _choose_images_to_import(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import STEM ZC Images",
            filter="STEM ZC Images (*.tif *.tiff *.dm3)",
        )
        if paths:
            self.import_images([Path(path) for path in paths])

    def _show_project(self, project: YOLOTrainingProject) -> None:
        self._current_project = project
        self._status_label.setText(f"Project loaded: {project.name}")
        self._queue_label.setText(_format_image_queue(project))
        self._error_label.setText("")
        if project.imported_images:
            self._show_annotation_image(project.imported_images[0])
        else:
            self._selected_image = None
            self._annotation_image_text = "No image selected for annotation."
            self._annotation_image_label.setText(self._annotation_image_text)
            self._annotation_image_label.setPixmap(QPixmap())
            self._annotation_status_label.setText("No boxes saved.")
            self._refresh_review_status()

    def _show_project_error(self, message: str) -> None:
        self._status_label.setText("No YOLO Training Project loaded.")
        self._queue_label.setText("Create or open a project to start.")
        self._error_label.setText(message)
        self._refresh_review_status()

    def _show_annotation_image(self, imported_image: ImportedImage) -> None:
        self._selected_image = imported_image
        self._annotation_image_text = f"Annotating {imported_image.display_name}"
        self._annotation_image_label.setText(self._annotation_image_text)
        self._annotation_image_label.setPixmap(
            QPixmap(str(imported_image.normalized_image_path))
        )
        self._refresh_annotation_status()

    def _draw_selected_class_box(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> None:
        self.draw_annotation_box(
            class_name=self._class_selector.currentText(),
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    def _refresh_annotation_status(self) -> None:
        if self._current_project is None or self._selected_image is None:
            self._annotation_status_label.setText("No boxes saved.")
            return
        annotations = AnnotationStore(self._current_project).load(self._selected_image)
        self._annotation_status_label.setText(
            _format_annotation_status(len(annotations))
        )
        self._refresh_review_status()

    def _refresh_review_status(self) -> None:
        if self._current_project is None:
            self._review_state_label.setText("Review state: none")
            self._review_progress_label.setText(
                "Review progress: 0/0 reviewed, 0 unreviewed"
            )
            return

        review_states = ReviewStateStore(self._current_project)
        if self._selected_image is None:
            self._review_state_label.setText("Review state: none")
        else:
            self._review_state_label.setText(
                f"Review state: {review_states.load(self._selected_image)}"
            )

        progress = review_states.progress(self._current_project.imported_images)
        self._review_progress_label.setText(
            "Review progress: "
            f"{progress.reviewed_count}/{progress.total_count} reviewed, "
            f"{progress.unreviewed_count} unreviewed"
        )


def build_main_window() -> MainWindow:
    return MainWindow()


def _format_image_queue(project: YOLOTrainingProject) -> str:
    if not project.imported_images:
        return "Image queue is empty."

    names = ", ".join(image.display_name for image in project.imported_images)
    if project.image_count == 1:
        return f"1 image imported: {names}"
    return f"{project.image_count} images imported: {names}"


def _format_import_failures(result: ImportResult) -> str:
    return "; ".join(
        f"{failure.source_path.name}: {failure.message}" for failure in result.failed
    )


def _format_annotation_status(count: int) -> str:
    if count == 0:
        return "No boxes saved."
    if count == 1:
        return "1 box saved."
    return f"{count} boxes saved."


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-trainer")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Start the Qt app in smoke-test mode and exit immediately.",
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = build_main_window()
    window.show()
    app.processEvents()

    if args.smoke:
        print("YOLO Trainer app smoke OK")
        window.close()
        app.processEvents()
        return 0

    return app.exec()


def main() -> int:
    return run()
