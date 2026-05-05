import json

import numpy as np
import tifffile

from yolo_trainer.annotations import PixelBox
from yolo_trainer.prediction import (
    RawPredictionBox,
    UltralyticsPredictionRunner,
    predict_project_image,
)
from yolo_trainer.project import create_project, open_project
from yolo_trainer.image_import import import_stem_zc_images


def test_prediction_uses_normalized_image_and_maps_boxes_to_original_coordinates(
    tmp_path,
) -> None:
    project = create_project(tmp_path / "project", name="Prediction Project")
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    import_stem_zc_images(project, [source])
    imported_image = open_project(project.path).imported_images[0]
    metadata = json.loads(imported_image.metadata_path.read_text(encoding="utf-8"))
    metadata["original_size"] = {"width": 40, "height": 80}
    metadata["coordinate_mapping"] = {
        "scale_x": 2.0,
        "scale_y": 3.0,
        "offset_x": 10.0,
        "offset_y": 20.0,
    }
    imported_image.metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    weights_path = tmp_path / "best.pt"
    weights_path.write_text("fake weights", encoding="utf-8")
    runner = FakePredictionRunner(
        [
            RawPredictionBox(
                class_id=2,
                confidence=0.8764,
                x_min=1.2,
                y_min=2.1,
                x_max=5.4,
                y_max=6.4,
            )
        ]
    )

    result = predict_project_image(
        imported_image,
        weights_path=weights_path,
        prediction_runner=runner,
    )

    assert runner.calls == [(weights_path, imported_image.normalized_image_path)]
    assert result.image_id == imported_image.image_id
    assert len(result.boxes) == 1
    box = result.boxes[0]
    assert box.class_id == 2
    assert box.class_name == "xsection_via"
    assert box.confidence == 0.8764
    assert box.normalized_box == PixelBox(x_min=1, y_min=2, x_max=5, y_max=6)
    assert box.original_box == PixelBox(x_min=12, y_min=26, x_max=20, y_max=38)


def test_ultralytics_prediction_runner_parses_model_boxes(tmp_path) -> None:
    weights_path = tmp_path / "best.pt"
    image_path = tmp_path / "normalized.png"
    weights_path.write_text("fake weights", encoding="utf-8")
    image_path.write_text("fake image", encoding="utf-8")
    model_factory = FakeModelFactory()
    runner = UltralyticsPredictionRunner(model_factory=model_factory)

    boxes = runner.predict(weights_path, image_path)

    assert model_factory.created_with == str(weights_path)
    assert model_factory.model.predicted_source == str(image_path)
    assert boxes == [
        RawPredictionBox(
            class_id=1,
            confidence=0.91,
            x_min=1.5,
            y_min=2.5,
            x_max=7.5,
            y_max=8.5,
        )
    ]


class FakePredictionRunner:
    def __init__(self, boxes: list[RawPredictionBox]) -> None:
        self._boxes = boxes
        self.calls: list[tuple] = []

    def predict(self, weights_path, image_path):
        self.calls.append((weights_path, image_path))
        return self._boxes


class FakeModelFactory:
    def __init__(self) -> None:
        self.created_with = ""
        self.model = FakeUltralyticsModel()

    def __call__(self, weights_path: str):
        self.created_with = weights_path
        return self.model


class FakeUltralyticsModel:
    def __init__(self) -> None:
        self.predicted_source = ""

    def predict(self, *, source: str, verbose: bool):
        self.predicted_source = source
        return [FakeUltralyticsResult()]


class FakeUltralyticsResult:
    boxes = [
        type(
            "FakeUltralyticsBox",
            (),
            {"xyxy": [[1.5, 2.5, 7.5, 8.5]], "conf": [0.91], "cls": [1]},
        )()
    ]
