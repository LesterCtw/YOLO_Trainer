import os

from PySide6.QtWidgets import QApplication

from yolo_trainer.app import MainWindow
from yolo_trainer.project import create_project


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_gui_creates_project_and_shows_empty_image_queue(tmp_path) -> None:
    _app()
    window = MainWindow()

    window.create_project_at(tmp_path / "new-project", name="New Project")

    assert window.current_project_name() == "New Project"
    assert window.project_status_text() == "Project loaded: New Project"
    assert window.image_queue_text() == "Image queue is empty."


def test_gui_opens_existing_project_after_window_reconstruction(tmp_path) -> None:
    _app()
    project_path = tmp_path / "existing-project"
    create_project(project_path, name="Existing Project")

    window = MainWindow()
    window.open_project_at(project_path)

    assert window.current_project_name() == "Existing Project"
    assert window.project_status_text() == "Project loaded: Existing Project"
    assert window.image_queue_text() == "Image queue is empty."


def test_gui_rejects_invalid_project_folder_with_message(tmp_path) -> None:
    _app()
    invalid_path = tmp_path / "plain-folder"
    invalid_path.mkdir()
    window = MainWindow()

    window.open_project_at(invalid_path)

    assert window.current_project_name() is None
    assert "not a YOLO Training Project" in window.project_error_text()
