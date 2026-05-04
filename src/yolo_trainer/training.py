from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable, Protocol


DEFAULT_PRETRAINED_MODEL = "YOLO11m.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = "auto"
DEFAULT_DEVICE = "0"
DEFAULT_OUTPUT_NAME = "yolo-trainer-run"
TRAINING_STATUS_RUNNING = "running"
TRAINING_STATUS_COMPLETED = "completed"
TRAINING_STATUS_FAILED = "failed"
TRAINING_STATUS_CANCELED = "canceled"


@dataclass(frozen=True)
class TrainingSettings:
    pretrained_model_path: str = DEFAULT_PRETRAINED_MODEL
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: str = DEFAULT_BATCH
    device: str = DEFAULT_DEVICE
    output_name: str = DEFAULT_OUTPUT_NAME


@dataclass(frozen=True)
class TrainingProcessCallbacks:
    log: Callable[[str], None]
    finished: Callable[[int], None]
    failed_to_start: Callable[[str], None]


class TrainingProcess(Protocol):
    def start(self) -> None: ...

    def cancel(self) -> None: ...


TrainingProcessFactory = Callable[
    [list[str], Path, TrainingProcessCallbacks],
    TrainingProcess,
]


@dataclass(frozen=True)
class TrainingRunRecord:
    run_id: str
    output_name: str
    status: str
    dataset_yaml_path: str
    pretrained_model_path: str
    epochs: int
    imgsz: int
    batch: str
    device: str


class TrainingRunStore:
    def __init__(self, project_path: Path) -> None:
        self._path = project_path / "training" / "runs.json"

    def create(
        self,
        settings: TrainingSettings,
        *,
        dataset_yaml_path: Path,
    ) -> TrainingRunRecord:
        record = TrainingRunRecord(
            run_id=_new_run_id(),
            output_name=settings.output_name,
            status=TRAINING_STATUS_RUNNING,
            dataset_yaml_path=str(dataset_yaml_path),
            pretrained_model_path=settings.pretrained_model_path,
            epochs=settings.epochs,
            imgsz=settings.imgsz,
            batch=settings.batch,
            device=settings.device,
        )
        records = self.list()
        records.append(record)
        self._write(records)
        return record

    def list(self) -> list[TrainingRunRecord]:
        if not self._path.exists():
            return []
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return [TrainingRunRecord(**item) for item in data]

    def mark_status(self, run_id: str, status: str) -> None:
        records = [
            record if record.run_id != run_id else _replace_status(record, status)
            for record in self.list()
        ]
        self._write(records)

    def _write(self, records: list[TrainingRunRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(record) for record in records], indent=2) + "\n",
            encoding="utf-8",
        )


def format_training_settings(settings: TrainingSettings) -> str:
    return (
        "Training settings: "
        f"model={settings.pretrained_model_path}, "
        f"epochs={settings.epochs}, "
        f"imgsz={settings.imgsz}, "
        f"batch={settings.batch}, "
        f"device={settings.device}, "
        f"output={settings.output_name}"
    )


def build_training_command(
    settings: TrainingSettings,
    *,
    dataset_yaml_path: Path,
    project_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "yolo_trainer.training_worker",
        "--model",
        settings.pretrained_model_path,
        "--data",
        str(dataset_yaml_path),
        "--epochs",
        str(settings.epochs),
        "--imgsz",
        str(settings.imgsz),
        "--batch",
        settings.batch,
        "--device",
        settings.device,
        "--project",
        str(project_path / "training" / "runs"),
        "--name",
        settings.output_name,
    ]


def format_training_run_history(records: list[TrainingRunRecord]) -> str:
    if not records:
        return "Training runs: none"
    return "Training runs: " + ", ".join(
        f"{record.output_name}: {record.status}" for record in records
    )


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


def _replace_status(record: TrainingRunRecord, status: str) -> TrainingRunRecord:
    return TrainingRunRecord(
        run_id=record.run_id,
        output_name=record.output_name,
        status=status,
        dataset_yaml_path=record.dataset_yaml_path,
        pretrained_model_path=record.pretrained_model_path,
        epochs=record.epochs,
        imgsz=record.imgsz,
        batch=record.batch,
        device=record.device,
    )
