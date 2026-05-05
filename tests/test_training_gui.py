import os

import numpy as np
from PySide6.QtWidgets import QApplication
import tifffile

from yolo_trainer.annotations import AnnotationStore, PixelBox
from yolo_trainer.app import MainWindow
from yolo_trainer.project import open_project


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_gui_shows_default_training_settings() -> None:
    _app()
    window = MainWindow()

    assert window.training_settings_text() == (
        "Training settings: model=YOLO11m.pt, epochs=100, imgsz=1024, "
        "batch=auto, device=0, output=yolo-trainer-run"
    )


def test_gui_updates_training_settings(tmp_path) -> None:
    _app()
    model_path = tmp_path / "custom-model.pt"
    model_path.write_text("fake weights", encoding="utf-8")
    window = MainWindow()

    window.select_pretrained_model(model_path)
    window.set_training_settings(
        epochs=12,
        imgsz=640,
        batch="4",
        device="cpu",
        output_name="trial-run",
    )

    assert window.training_settings_text() == (
        f"Training settings: model={model_path}, epochs=12, imgsz=640, "
        "batch=4, device=cpu, output=trial-run"
    )


def test_gui_starts_training_from_exported_dataset_and_streams_logs(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()

    assert len(processes) == 1
    assert processes[0].started
    assert str(tmp_path / "export" / "dataset.yaml") in processes[0].command
    assert window.training_status_text() == "Training: running yolo-trainer-run"

    processes[0].emit_log("epoch 1/100\n")

    assert "epoch 1/100" in window.training_log_text()


def test_gui_shows_training_completion_and_failure(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    processes[0].finish(exit_code=0)

    assert window.training_status_text() == "Training: completed yolo-trainer-run"

    window.start_training()
    processes[1].finish(exit_code=2)

    assert window.training_status_text() == "Training: failed yolo-trainer-run"


def test_gui_summarizes_failed_training_run_with_log_context(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    processes[0].emit_log("loading dataset\n")
    processes[0].emit_log("RuntimeError: CUDA out of memory\n")
    processes[0].finish(exit_code=2)

    summary = window.training_run_summary_text()

    assert "Run yolo-trainer-run: failed" in summary
    assert "Log context:" in summary
    assert "loading dataset" in summary
    assert "RuntimeError: CUDA out of memory" in summary

    reopened = MainWindow(training_process_factory=_fake_training_factory([]))
    reopened.open_project_at(tmp_path / "project")

    assert "RuntimeError: CUDA out of memory" in reopened.training_run_summary_text()


def test_gui_summarizes_completed_training_run_outputs(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    run_output_path = tmp_path / "project" / "training" / "runs" / "yolo-trainer-run"
    best_model_path = run_output_path / "weights" / "best.pt"
    best_model_path.parent.mkdir(parents=True)
    best_model_path.write_text("fake best weights", encoding="utf-8")
    processes[0].finish(exit_code=0)

    summary = window.training_run_summary_text()

    assert "Run yolo-trainer-run: completed" in summary
    assert f"Official best.pt: {best_model_path}" in summary
    assert f"Dataset: {tmp_path / 'export' / 'dataset.yaml'}" in summary


def test_gui_summarizes_completed_training_metrics_and_plots(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    run_output_path = tmp_path / "project" / "training" / "runs" / "yolo-trainer-run"
    (run_output_path / "weights").mkdir(parents=True)
    (run_output_path / "weights" / "best.pt").write_text(
        "fake best weights",
        encoding="utf-8",
    )
    metrics_path = run_output_path / "results.csv"
    plot_path = run_output_path / "results.png"
    confusion_matrix_path = run_output_path / "confusion_matrix.png"
    metrics_path.write_text("epoch,metrics/mAP50(B)\n1,0.9\n", encoding="utf-8")
    plot_path.write_text("fake plot", encoding="utf-8")
    confusion_matrix_path.write_text("fake matrix", encoding="utf-8")
    processes[0].finish(exit_code=0)

    summary = window.training_run_summary_text()

    assert f"Metrics: {metrics_path}" in summary
    assert f"Plots: {confusion_matrix_path}, {plot_path}" in summary
    assert "quality pass" not in summary.lower()
    assert "quality fail" not in summary.lower()


def test_gui_shows_failed_process_start(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    processes[0].fail_to_start("program not found")

    assert window.training_status_text() == (
        "Training: failed to start: program not found"
    )


def test_gui_cancels_training_and_retains_canceled_run(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    project_path = _create_exported_labeled_project(window, tmp_path)
    window.set_training_settings(output_name="cancel-me")

    window.start_training()
    window.cancel_training()

    assert processes[0].canceled
    assert window.training_status_text() == "Training: canceled cancel-me"
    assert "cancel-me: canceled" in window.training_run_history_text()

    reopened = MainWindow(training_process_factory=_fake_training_factory([]))
    reopened.open_project_at(project_path)

    assert "cancel-me: canceled" in reopened.training_run_history_text()


def test_gui_summarizes_canceled_run_without_official_outputs(tmp_path) -> None:
    _app()
    processes: list[FakeTrainingProcess] = []
    window = MainWindow(training_process_factory=_fake_training_factory(processes))
    _create_exported_labeled_project(window, tmp_path)

    window.start_training()
    partial_best_model_path = (
        tmp_path
        / "project"
        / "training"
        / "runs"
        / "yolo-trainer-run"
        / "weights"
        / "best.pt"
    )
    partial_best_model_path.parent.mkdir(parents=True)
    partial_best_model_path.write_text("partial weights", encoding="utf-8")
    window.cancel_training()

    summary = window.training_run_summary_text()

    assert "Run yolo-trainer-run: canceled" in summary
    assert "Official best.pt:" not in summary
    assert str(partial_best_model_path) not in summary


class FakeTrainingProcess:
    def __init__(self, command, working_directory, callbacks) -> None:
        self.command = command
        self.working_directory = working_directory
        self.callbacks = callbacks
        self.started = False
        self.canceled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.canceled = True

    def emit_log(self, text: str) -> None:
        self.callbacks.log(text)

    def finish(self, *, exit_code: int) -> None:
        self.callbacks.finished(exit_code)

    def fail_to_start(self, message: str) -> None:
        self.callbacks.failed_to_start(message)


def _fake_training_factory(processes: list[FakeTrainingProcess]):
    def factory(command, working_directory, callbacks):
        process = FakeTrainingProcess(command, working_directory, callbacks)
        processes.append(process)
        return process

    return factory


def _create_exported_labeled_project(window: MainWindow, tmp_path):
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    project_path = tmp_path / "project"
    window.create_project_at(project_path, name="Training Project")
    window.import_images([source])
    project = open_project(project_path)
    imported_image = project.imported_images[0]
    AnnotationStore(project).add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )
    window.open_project_at(project_path)
    window.export_dataset_to(tmp_path / "export")
    return project_path
