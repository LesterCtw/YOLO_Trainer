from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from yolo_trainer.annotations import CANONICAL_CLASSES, PixelBox
from yolo_trainer.project import ImportedImage


@dataclass(frozen=True)
class RawPredictionBox:
    class_id: int
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class PredictionBox:
    class_id: int
    class_name: str
    confidence: float
    normalized_box: PixelBox
    original_box: PixelBox


@dataclass(frozen=True)
class PredictionResult:
    image_id: str
    boxes: list[PredictionBox]


class PredictionRunner(Protocol):
    def predict(
        self,
        weights_path: Path,
        image_path: Path,
    ) -> list[RawPredictionBox]: ...


class UltralyticsPredictionRunner:
    def __init__(self, model_factory=None) -> None:
        self._model_factory = model_factory

    def predict(
        self,
        weights_path: Path,
        image_path: Path,
    ) -> list[RawPredictionBox]:
        model_factory = self._model_factory or _load_ultralytics_model_factory()

        model = model_factory(str(weights_path))
        results = model.predict(source=str(image_path), verbose=False)
        if not results:
            return []
        return _raw_boxes_from_ultralytics_result(results[0])


def predict_project_image(
    imported_image: ImportedImage,
    *,
    weights_path: Path | str,
    prediction_runner: PredictionRunner,
) -> PredictionResult:
    weights = Path(weights_path)
    raw_boxes = prediction_runner.predict(weights, imported_image.normalized_image_path)
    metadata = json.loads(imported_image.metadata_path.read_text(encoding="utf-8"))
    boxes = [
        PredictionBox(
            class_id=raw_box.class_id,
            class_name=CANONICAL_CLASSES[raw_box.class_id],
            confidence=raw_box.confidence,
            normalized_box=_normalized_box(raw_box),
            original_box=_map_to_original(_normalized_box(raw_box), metadata),
        )
        for raw_box in raw_boxes
    ]
    return PredictionResult(image_id=imported_image.image_id, boxes=boxes)


def _normalized_box(raw_box: RawPredictionBox) -> PixelBox:
    return PixelBox(
        x_min=round(raw_box.x_min),
        y_min=round(raw_box.y_min),
        x_max=round(raw_box.x_max),
        y_max=round(raw_box.y_max),
    )


def _map_to_original(box: PixelBox, metadata: dict) -> PixelBox:
    mapping = metadata["coordinate_mapping"]
    return PixelBox(
        x_min=_map_coordinate(box.x_min, mapping["scale_x"], mapping["offset_x"]),
        y_min=_map_coordinate(box.y_min, mapping["scale_y"], mapping["offset_y"]),
        x_max=_map_coordinate(box.x_max, mapping["scale_x"], mapping["offset_x"]),
        y_max=_map_coordinate(box.y_max, mapping["scale_y"], mapping["offset_y"]),
    )


def _map_coordinate(value: int, scale: float, offset: float) -> int:
    return round(value * scale + offset)


def _load_ultralytics_model_factory():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install the training environment "
            "dependencies before running prediction preview."
        ) from error
    return YOLO


def _raw_boxes_from_ultralytics_result(result: Any) -> list[RawPredictionBox]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    raw_boxes: list[RawPredictionBox] = []
    for box in boxes:
        xyxy_values = _as_list(box.xyxy[0])
        confidence_values = _as_list(box.conf)
        class_values = _as_list(box.cls)
        raw_boxes.append(
            RawPredictionBox(
                class_id=int(class_values[0]),
                confidence=float(confidence_values[0]),
                x_min=float(xyxy_values[0]),
                y_min=float(xyxy_values[1]),
                x_max=float(xyxy_values[2]),
                y_max=float(xyxy_values[3]),
            )
        )
    return raw_boxes


def _as_list(value: Any) -> list:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value
