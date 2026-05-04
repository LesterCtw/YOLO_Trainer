import os

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import tifffile

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


def test_gui_imports_images_into_active_project_queue(tmp_path) -> None:
    _app()
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(6, dtype=np.uint16).reshape(2, 3))
    window = MainWindow()
    window.create_project_at(tmp_path / "project", name="Image Project")

    window.import_images([source])

    assert window.project_error_text() == ""
    assert window.image_queue_text() == "1 image imported: source.tif"

    reopened = MainWindow()
    reopened.open_project_at(tmp_path / "project")

    assert reopened.image_queue_text() == "1 image imported: source.tif"


def test_gui_surfaces_failed_import_without_clearing_existing_queue(tmp_path) -> None:
    _app()
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(6, dtype=np.uint16).reshape(2, 3))
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    window = MainWindow()
    window.create_project_at(tmp_path / "project", name="Image Project")
    window.import_images([source])

    window.import_images([unsupported])

    assert window.image_queue_text() == "1 image imported: source.tif"
    assert "notes.txt" in window.project_error_text()
    assert "Unsupported STEM ZC Image format" in window.project_error_text()


def test_gui_imports_dm3_with_controlled_reader(tmp_path) -> None:
    _app()
    source = tmp_path / "source.dm3"
    source.write_bytes(b"controlled dm3 fixture")
    window = MainWindow(
        dm3_reader=lambda path: np.arange(6, dtype=np.uint16).reshape(2, 3),
    )
    window.create_project_at(tmp_path / "project", name="DM3 Project")

    window.import_images([source])

    assert window.project_error_text() == ""
    assert window.image_queue_text() == "1 image imported: source.dm3"


def test_gui_annotation_smoke_autosaves_and_restores_boxes(tmp_path) -> None:
    _app()
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    window = MainWindow()
    window.create_project_at(tmp_path / "project", name="Annotation Project")
    window.import_images([source])

    image_id = window.selected_image_id()
    assert image_id is not None
    assert window.annotation_image_text() == "Annotating source.tif"

    window.draw_annotation_box(
        class_name="xsection_metal",
        x_min=0,
        y_min=0,
        x_max=2,
        y_max=2,
    )
    window.draw_annotation_box(
        class_name="alongline_via",
        x_min=1,
        y_min=1,
        x_max=4,
        y_max=4,
    )

    assert window.annotation_status_text() == "2 boxes saved."
    window.undo_last_annotation()
    assert window.annotation_status_text() == "1 box saved."

    reopened = MainWindow()
    reopened.open_project_at(tmp_path / "project")
    reopened.select_image(image_id)

    assert reopened.annotation_image_text() == "Annotating source.tif"
    assert reopened.annotation_status_text() == "1 box saved."

    reopened.delete_annotation(index=0)
    assert reopened.annotation_status_text() == "No boxes saved."


def test_gui_shows_review_state_progress_and_reviewed_empty_workflow(tmp_path) -> None:
    _app()
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    window = MainWindow()
    window.create_project_at(tmp_path / "project", name="Review Project")
    window.import_images([source])

    image_id = window.selected_image_id()
    assert image_id is not None
    assert window.review_state_text() == "Review state: unreviewed"
    assert (
        window.review_progress_text()
        == "Review progress: 0/1 reviewed, 1 unreviewed"
    )

    window.draw_annotation_box(
        class_name="xsection_metal",
        x_min=0,
        y_min=0,
        x_max=2,
        y_max=2,
    )

    assert window.review_state_text() == "Review state: labeled"
    assert (
        window.review_progress_text()
        == "Review progress: 1/1 reviewed, 0 unreviewed"
    )

    window.undo_last_annotation()

    assert window.review_state_text() == "Review state: unreviewed"
    assert (
        window.review_progress_text()
        == "Review progress: 0/1 reviewed, 1 unreviewed"
    )

    window.mark_selected_image_reviewed_empty()

    assert window.review_state_text() == "Review state: reviewed_empty"
    assert (
        window.review_progress_text()
        == "Review progress: 1/1 reviewed, 0 unreviewed"
    )

    reopened = MainWindow()
    reopened.open_project_at(tmp_path / "project")
    reopened.select_image(image_id)

    assert reopened.review_state_text() == "Review state: reviewed_empty"
    assert (
        reopened.review_progress_text()
        == "Review progress: 1/1 reviewed, 0 unreviewed"
    )


def test_gui_mouse_drag_draws_annotation_box(tmp_path) -> None:
    app = _app()
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    window = MainWindow()
    window.create_project_at(tmp_path / "project", name="Annotation Project")
    window.import_images([source])
    window.show()
    app.processEvents()

    canvas = window.annotation_canvas()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(0, 0))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))

    assert window.annotation_status_text() == "1 box saved."
