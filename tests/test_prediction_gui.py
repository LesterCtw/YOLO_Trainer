import os

import numpy as np
from PySide6.QtWidgets import QApplication
import tifffile

from yolo_trainer.annotations import AnnotationStore, PixelBox, ReviewStateStore
from yolo_trainer.app import MainWindow
from yolo_trainer.prediction import RawPredictionBox
from yolo_trainer.project import open_project


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_gui_previews_predictions_without_mutating_manual_annotations(tmp_path) -> None:
    _app()
    runner = FakePredictionRunner(
        [
            RawPredictionBox(
                class_id=0,
                confidence=0.876,
                x_min=1,
                y_min=1,
                x_max=3,
                y_max=3,
            )
        ]
    )
    window = MainWindow(prediction_runner=runner)
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    project_path = tmp_path / "project"
    weights_path = tmp_path / "best.pt"
    weights_path.write_text("fake weights", encoding="utf-8")
    window.create_project_at(project_path, name="Prediction Project")
    window.import_images([source])
    image_id = window.selected_image_id()
    assert image_id is not None
    window.draw_annotation_box(
        class_name="xsection_metal",
        x_min=0,
        y_min=0,
        x_max=2,
        y_max=2,
    )
    project = open_project(project_path)
    imported_image = project.imported_images[0]
    original_annotations = AnnotationStore(project).load(imported_image)
    original_review_state = ReviewStateStore(project).load(imported_image)

    window.select_prediction_weights(weights_path)
    window.run_prediction_preview()

    project_after_preview = open_project(project_path)
    imported_after_preview = project_after_preview.imported_images[0]
    assert runner.calls == [(weights_path, imported_image.normalized_image_path)]
    assert window.prediction_status_text() == "Prediction: 1 box on source.tif"
    assert "xsection_metal 87.6%" in window.prediction_preview_text()
    assert "normalized=(1,1)-(3,3)" in window.prediction_preview_text()
    assert AnnotationStore(project_after_preview).load(imported_after_preview) == (
        original_annotations
    )
    assert ReviewStateStore(project_after_preview).load(imported_after_preview) == (
        original_review_state
    )


def test_gui_can_use_latest_successful_run_best_weights_for_prediction(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    runner = FakePredictionRunner([])
    window = MainWindow(
        training_process_factory=_fake_training_factory(processes),
        prediction_runner=runner,
    )
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    project_path = tmp_path / "project"
    window.create_project_at(project_path, name="Prediction Project")
    window.import_images([source])
    window.draw_annotation_box(
        class_name="xsection_metal",
        x_min=0,
        y_min=0,
        x_max=2,
        y_max=2,
    )
    window.export_dataset_to(tmp_path / "export")
    window.start_training()
    best_model_path = (
        project_path / "training" / "runs" / "yolo-trainer-run" / "weights" / "best.pt"
    )
    best_model_path.parent.mkdir(parents=True)
    best_model_path.write_text("fake best weights", encoding="utf-8")
    processes[0].finish(exit_code=0)

    window.select_latest_successful_run_weights()
    window.run_prediction_preview()

    imported_image = open_project(project_path).imported_images[0]
    assert runner.calls == [(best_model_path, imported_image.normalized_image_path)]
    assert window.prediction_status_text() == "Prediction: 0 boxes on source.tif"


def test_gui_surfaces_prediction_weight_errors(tmp_path) -> None:
    _app()
    runner = FakePredictionRunner([])
    window = MainWindow(prediction_runner=runner)
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    valid_weights = tmp_path / "valid.pt"
    valid_weights.write_text("fake weights", encoding="utf-8")
    window.create_project_at(tmp_path / "project", name="Prediction Project")
    window.import_images([source])

    window.select_prediction_weights(valid_weights)
    window.select_prediction_weights(tmp_path / "weights.txt")

    assert window.prediction_status_text() == (
        "Prediction: select an Ultralytics-compatible .pt file."
    )
    window.run_prediction_preview()

    assert window.prediction_status_text() == "Prediction: select weights before preview."
    assert runner.calls == []

    window.select_prediction_weights(tmp_path / "missing.pt")
    window.run_prediction_preview()

    assert window.prediction_status_text() == "Prediction: weights file missing."


def test_gui_surfaces_prediction_runner_failures(tmp_path) -> None:
    _app()
    window = MainWindow(prediction_runner=FailingPredictionRunner())
    source = tmp_path / "source.tif"
    weights_path = tmp_path / "broken.pt"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    weights_path.write_text("not real weights", encoding="utf-8")
    window.create_project_at(tmp_path / "project", name="Prediction Project")
    window.import_images([source])
    window.select_prediction_weights(weights_path)

    window.run_prediction_preview()

    assert window.prediction_status_text() == (
        "Prediction: failed: incompatible weights"
    )


class FakePredictionRunner:
    def __init__(self, boxes: list[RawPredictionBox]) -> None:
        self._boxes = boxes
        self.calls: list[tuple] = []

    def predict(self, weights_path, image_path):
        self.calls.append((weights_path, image_path))
        return self._boxes


class FailingPredictionRunner:
    def predict(self, weights_path, image_path):
        raise RuntimeError("incompatible weights")


class FakeTrainingProcess:
    def __init__(self, command, working_directory, callbacks) -> None:
        self.command = command
        self.working_directory = working_directory
        self.callbacks = callbacks

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    def finish(self, *, exit_code: int) -> None:
        self.callbacks.finished(exit_code)


def _fake_training_factory(processes: list[FakeTrainingProcess]):
    def factory(command, working_directory, callbacks):
        process = FakeTrainingProcess(command, working_directory, callbacks)
        processes.append(process)
        return process

    return factory
