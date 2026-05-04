from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from PySide6.QtCore import QPoint, QProcess, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yolo_trainer.annotations import (
    CANONICAL_CLASSES,
    AnnotationStore,
    PixelBox,
    ReviewStateStore,
)
from yolo_trainer.dataset_export import DatasetExportResult, export_dataset
from yolo_trainer.image_import import ImportResult, import_stem_zc_images
from yolo_trainer.project import (
    ImportedImage,
    InvalidProjectError,
    YOLOTrainingProject,
    create_project,
    open_project,
)
from yolo_trainer.training import (
    TRAINING_STATUS_CANCELED,
    TRAINING_STATUS_COMPLETED,
    TRAINING_STATUS_FAILED,
    TrainingProcessCallbacks,
    TrainingProcessFactory,
    TrainingRunStore,
    TrainingSettings,
    build_training_command,
    format_training_run_history,
    format_training_settings,
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
        training_process_factory: TrainingProcessFactory | None = None,
    ) -> None:
        super().__init__()
        self._current_project: YOLOTrainingProject | None = None
        self._selected_image: ImportedImage | None = None
        self._annotation_image_text = "No image selected for annotation."
        self._dm3_reader = dm3_reader
        self._training_settings = TrainingSettings()
        self._last_dataset_export_path: Path | None = None
        self._training_process_factory = (
            training_process_factory or qt_training_process_factory
        )
        self._training_process = None
        self._active_training_run_id: str | None = None

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

        self._dataset_export_status_label = QLabel("Dataset export: not exported.")
        self._dataset_export_status_label.setObjectName("datasetExportStatusLabel")

        self._pretrained_model_input = QLineEdit(
            self._training_settings.pretrained_model_path
        )
        self._pretrained_model_input.setObjectName("pretrainedModelInput")

        self._epochs_input = QSpinBox()
        self._epochs_input.setObjectName("trainingEpochsInput")
        self._epochs_input.setRange(1, 1_000_000)
        self._epochs_input.setValue(self._training_settings.epochs)

        self._imgsz_input = QSpinBox()
        self._imgsz_input.setObjectName("trainingImgszInput")
        self._imgsz_input.setRange(1, 1_000_000)
        self._imgsz_input.setValue(self._training_settings.imgsz)

        self._batch_input = QLineEdit(self._training_settings.batch)
        self._batch_input.setObjectName("trainingBatchInput")

        self._device_input = QLineEdit(self._training_settings.device)
        self._device_input.setObjectName("trainingDeviceInput")

        self._output_name_input = QLineEdit(self._training_settings.output_name)
        self._output_name_input.setObjectName("trainingOutputNameInput")

        self._training_settings_label = QLabel(
            format_training_settings(self._training_settings)
        )
        self._training_settings_label.setObjectName("trainingSettingsLabel")

        self._training_status_label = QLabel("Training: not started.")
        self._training_status_label.setObjectName("trainingStatusLabel")

        self._training_run_history_label = QLabel("Training runs: none")
        self._training_run_history_label.setObjectName("trainingRunHistoryLabel")

        self._training_log = QPlainTextEdit()
        self._training_log.setObjectName("trainingLog")
        self._training_log.setReadOnly(True)

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

        export_dataset_button = QPushButton("Export Dataset")
        export_dataset_button.setObjectName("exportDatasetButton")
        export_dataset_button.clicked.connect(self._choose_dataset_export_directory)

        choose_model_button = QPushButton("Choose Model")
        choose_model_button.setObjectName("chooseModelButton")
        choose_model_button.clicked.connect(self._choose_pretrained_model)

        start_training_button = QPushButton("Start Training")
        start_training_button.setObjectName("startTrainingButton")
        start_training_button.clicked.connect(self.start_training)

        cancel_training_button = QPushButton("Cancel Training")
        cancel_training_button.setObjectName("cancelTrainingButton")
        cancel_training_button.clicked.connect(self.cancel_training)

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
        layout.addWidget(export_dataset_button)
        layout.addWidget(self._dataset_export_status_label)
        layout.addWidget(QLabel("Pretrained model"))
        layout.addWidget(self._pretrained_model_input)
        layout.addWidget(choose_model_button)
        layout.addWidget(QLabel("Epochs"))
        layout.addWidget(self._epochs_input)
        layout.addWidget(QLabel("Image size"))
        layout.addWidget(self._imgsz_input)
        layout.addWidget(QLabel("Batch"))
        layout.addWidget(self._batch_input)
        layout.addWidget(QLabel("Device"))
        layout.addWidget(self._device_input)
        layout.addWidget(QLabel("Output name"))
        layout.addWidget(self._output_name_input)
        layout.addWidget(self._training_settings_label)
        layout.addWidget(start_training_button)
        layout.addWidget(cancel_training_button)
        layout.addWidget(self._training_status_label)
        layout.addWidget(self._training_run_history_label)
        layout.addWidget(self._training_log)
        layout.addStretch()

        root = QWidget()
        root.setLayout(layout)
        root.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(root)

        self._pretrained_model_input.textChanged.connect(
            self._sync_training_settings_from_controls
        )
        self._epochs_input.valueChanged.connect(self._sync_training_settings_from_controls)
        self._imgsz_input.valueChanged.connect(self._sync_training_settings_from_controls)
        self._batch_input.textChanged.connect(self._sync_training_settings_from_controls)
        self._device_input.textChanged.connect(self._sync_training_settings_from_controls)
        self._output_name_input.textChanged.connect(
            self._sync_training_settings_from_controls
        )

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

    def export_dataset_to(self, path: Path | str) -> None:
        if self._current_project is None:
            self._show_project_error("Open a YOLO Training Project before exporting.")
            return
        result = export_dataset(self._current_project, path)
        self._last_dataset_export_path = result.export_path
        self._dataset_export_status_label.setText(
            _format_dataset_export_result(result)
        )

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

    def dataset_export_status_text(self) -> str:
        return self._dataset_export_status_label.text()

    def training_settings_text(self) -> str:
        return self._training_settings_label.text()

    def training_status_text(self) -> str:
        return self._training_status_label.text()

    def training_log_text(self) -> str:
        return self._training_log.toPlainText()

    def training_run_history_text(self) -> str:
        return self._training_run_history_label.text()

    def start_training(self) -> None:
        if self._current_project is None:
            self._show_project_error("Open a YOLO Training Project before training.")
            return
        if self._last_dataset_export_path is None:
            self._training_status_label.setText("Training: export dataset first.")
            return

        dataset_yaml_path = self._last_dataset_export_path / "dataset.yaml"
        if not dataset_yaml_path.exists():
            self._training_status_label.setText("Training: exported dataset is missing.")
            return

        self._training_log.clear()
        self._training_status_label.setText(
            f"Training: running {self._training_settings.output_name}"
        )
        run_record = TrainingRunStore(self._current_project.path).create(
            self._training_settings,
            dataset_yaml_path=dataset_yaml_path,
        )
        self._active_training_run_id = run_record.run_id
        self._refresh_training_run_history()
        callbacks = TrainingProcessCallbacks(
            log=self._append_training_log,
            finished=self._finish_training,
            failed_to_start=self._fail_training_start,
        )
        command = build_training_command(
            self._training_settings,
            dataset_yaml_path=dataset_yaml_path,
            project_path=self._current_project.path,
        )
        self._training_process = self._training_process_factory(
            command,
            self._current_project.path,
            callbacks,
        )
        self._training_process.start()

    def cancel_training(self) -> None:
        if self._training_process is None or self._current_project is None:
            return
        self._training_process.cancel()
        if self._active_training_run_id is not None:
            TrainingRunStore(self._current_project.path).mark_status(
                self._active_training_run_id,
                TRAINING_STATUS_CANCELED,
            )
        self._training_status_label.setText(
            f"Training: canceled {self._training_settings.output_name}"
        )
        self._active_training_run_id = None
        self._training_process = None
        self._refresh_training_run_history()

    def select_pretrained_model(self, path: Path | str) -> None:
        model_path = Path(path)
        if model_path.suffix.lower() != ".pt":
            self._show_project_error("Select an Ultralytics-compatible .pt file.")
            return
        self._pretrained_model_input.setText(str(model_path))
        self._sync_training_settings_from_controls()

    def set_training_settings(
        self,
        *,
        epochs: int | None = None,
        imgsz: int | None = None,
        batch: str | None = None,
        device: str | None = None,
        output_name: str | None = None,
    ) -> None:
        if epochs is not None:
            self._epochs_input.setValue(epochs)
        if imgsz is not None:
            self._imgsz_input.setValue(imgsz)
        if batch is not None:
            self._batch_input.setText(batch)
        if device is not None:
            self._device_input.setText(device)
        if output_name is not None:
            self._output_name_input.setText(output_name)
        self._sync_training_settings_from_controls()

    def annotation_canvas(self) -> AnnotationImageLabel:
        return self._annotation_image_label

    def _choose_dataset_export_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Export Ultralytics Dataset",
        )
        if path:
            self.export_dataset_to(path)

    def _choose_pretrained_model(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Pretrained YOLO Weights",
            filter="Ultralytics weights (*.pt)",
        )
        if path:
            self.select_pretrained_model(path)

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

    def _refresh_training_settings(self) -> None:
        self._training_settings_label.setText(
            format_training_settings(self._training_settings)
        )

    def _sync_training_settings_from_controls(self) -> None:
        self._training_settings = TrainingSettings(
            pretrained_model_path=self._pretrained_model_input.text(),
            epochs=self._epochs_input.value(),
            imgsz=self._imgsz_input.value(),
            batch=self._batch_input.text(),
            device=self._device_input.text(),
            output_name=self._output_name_input.text(),
        )
        self._refresh_training_settings()

    def _append_training_log(self, text: str) -> None:
        cursor = self._training_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._training_log.setTextCursor(cursor)
        self._training_log.insertPlainText(text)

    def _finish_training(self, exit_code: int) -> None:
        if self._active_training_run_id is None:
            return
        if exit_code == 0:
            status = TRAINING_STATUS_COMPLETED
            self._training_status_label.setText(
                f"Training: completed {self._training_settings.output_name}"
            )
        else:
            status = TRAINING_STATUS_FAILED
            self._training_status_label.setText(
                f"Training: failed {self._training_settings.output_name}"
            )
        if self._current_project is not None:
            TrainingRunStore(self._current_project.path).mark_status(
                self._active_training_run_id,
                status,
            )
        self._training_process = None
        self._active_training_run_id = None
        self._refresh_training_run_history()

    def _fail_training_start(self, message: str) -> None:
        if self._current_project is not None and self._active_training_run_id:
            TrainingRunStore(self._current_project.path).mark_status(
                self._active_training_run_id,
                TRAINING_STATUS_FAILED,
            )
        self._training_status_label.setText(f"Training: failed to start: {message}")
        self._active_training_run_id = None
        self._training_process = None
        self._refresh_training_run_history()

    def _show_project(self, project: YOLOTrainingProject) -> None:
        previous_project_path = (
            self._current_project.path if self._current_project is not None else None
        )
        if previous_project_path != project.path:
            self._last_dataset_export_path = None
        self._current_project = project
        self._status_label.setText(f"Project loaded: {project.name}")
        self._queue_label.setText(_format_image_queue(project))
        self._error_label.setText("")
        self._refresh_training_run_history()
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
        self._refresh_training_run_history()

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

    def _refresh_training_run_history(self) -> None:
        if self._current_project is None:
            self._training_run_history_label.setText("Training runs: none")
            return
        records = TrainingRunStore(self._current_project.path).list()
        self._training_run_history_label.setText(format_training_run_history(records))


class QtTrainingProcess:
    def __init__(
        self,
        command: list[str],
        working_directory: Path,
        callbacks: TrainingProcessCallbacks,
    ) -> None:
        self._callbacks = callbacks
        self._process = QProcess()
        self._process.setProgram(command[0])
        self._process.setArguments(command[1:])
        self._process.setWorkingDirectory(str(working_directory))
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.errorOccurred.connect(self._handle_error)
        self._process.finished.connect(self._handle_finished)

    def start(self) -> None:
        self._process.start()

    def cancel(self) -> None:
        self._process.terminate()

    def _read_output(self) -> None:
        output = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        if output:
            self._callbacks.log(output)

    def _handle_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._callbacks.failed_to_start(self._process.errorString())

    def _handle_finished(self, exit_code: int, _exit_status) -> None:
        self._callbacks.finished(exit_code)


def qt_training_process_factory(
    command: list[str],
    working_directory: Path,
    callbacks: TrainingProcessCallbacks,
) -> QtTrainingProcess:
    return QtTrainingProcess(command, working_directory, callbacks)


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


def _format_dataset_export_result(result: DatasetExportResult) -> str:
    return (
        "Dataset export: "
        f"{result.included_count} images exported "
        f"({result.train_count} train, {result.val_count} val), "
        f"{result.skipped_unreviewed_count} unreviewed skipped."
    )


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
