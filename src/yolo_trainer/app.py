from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yolo_trainer.project import (
    InvalidProjectError,
    YOLOTrainingProject,
    create_project,
    open_project,
)


APP_NAME = "YOLO Trainer"
MINIMUM_WINDOW_SIZE = QSize(960, 640)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._current_project: YOLOTrainingProject | None = None

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(MINIMUM_WINDOW_SIZE)

        self._status_label = QLabel("No YOLO Training Project loaded.")
        self._status_label.setObjectName("projectStatusLabel")

        self._queue_label = QLabel("Create or open a project to start.")
        self._queue_label.setObjectName("imageQueueLabel")

        self._error_label = QLabel("")
        self._error_label.setObjectName("projectErrorLabel")

        create_button = QPushButton("Create Project")
        create_button.setObjectName("createProjectButton")
        create_button.clicked.connect(self._choose_project_to_create)

        open_button = QPushButton("Open Project")
        open_button.setObjectName("openProjectButton")
        open_button.clicked.connect(self._choose_project_to_open)

        layout = QVBoxLayout()
        layout.addWidget(create_button)
        layout.addWidget(open_button)
        layout.addWidget(self._status_label)
        layout.addWidget(self._queue_label)
        layout.addWidget(self._error_label)
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

    def _show_project(self, project: YOLOTrainingProject) -> None:
        self._current_project = project
        self._status_label.setText(f"Project loaded: {project.name}")
        self._queue_label.setText("Image queue is empty.")
        self._error_label.setText("")

    def _show_project_error(self, message: str) -> None:
        self._status_label.setText("No YOLO Training Project loaded.")
        self._queue_label.setText("Create or open a project to start.")
        self._error_label.setText(message)


def build_main_window() -> MainWindow:
    return MainWindow()


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
