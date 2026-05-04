from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from yolo_trainer.project import ImportedImage, YOLOTrainingProject


CANONICAL_CLASSES = (
    "xsection_metal",
    "alongline_metal",
    "xsection_via",
    "alongline_via",
)
REVIEW_STATE_UNREVIEWED = "unreviewed"
REVIEW_STATE_LABELED = "labeled"
REVIEW_STATE_REVIEWED_EMPTY = "reviewed_empty"
REVIEW_STATES = (
    REVIEW_STATE_UNREVIEWED,
    REVIEW_STATE_LABELED,
    REVIEW_STATE_REVIEWED_EMPTY,
)


@dataclass(frozen=True)
class PixelBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class MetalDetectionBox:
    class_id: int
    class_name: str
    pixel_box: PixelBox


@dataclass(frozen=True)
class ReviewProgress:
    total_count: int
    reviewed_count: int
    unreviewed_count: int


class ReviewStateStore:
    def __init__(self, project: YOLOTrainingProject) -> None:
        self._project = project

    def load(self, imported_image: ImportedImage) -> str:
        metadata = _read_image_metadata(imported_image)
        review_state = str(metadata.get("review_state", REVIEW_STATE_UNREVIEWED))
        if review_state not in REVIEW_STATES:
            return REVIEW_STATE_UNREVIEWED
        return review_state

    def mark_labeled(self, imported_image: ImportedImage) -> None:
        self._save(imported_image, REVIEW_STATE_LABELED)

    def mark_unreviewed(self, imported_image: ImportedImage) -> None:
        self._save(imported_image, REVIEW_STATE_UNREVIEWED)

    def mark_reviewed_empty(self, imported_image: ImportedImage) -> None:
        self._save(imported_image, REVIEW_STATE_REVIEWED_EMPTY)

    def progress(self, imported_images: tuple[ImportedImage, ...]) -> ReviewProgress:
        total_count = len(imported_images)
        unreviewed_count = sum(
            1
            for imported_image in imported_images
            if self.load(imported_image) == REVIEW_STATE_UNREVIEWED
        )
        return ReviewProgress(
            total_count=total_count,
            reviewed_count=total_count - unreviewed_count,
            unreviewed_count=unreviewed_count,
        )

    def _save(self, imported_image: ImportedImage, review_state: str) -> None:
        metadata = _read_image_metadata(imported_image)
        metadata["review_state"] = review_state
        _write_image_metadata(imported_image, metadata)


class AnnotationStore:
    def __init__(self, project: YOLOTrainingProject) -> None:
        self._project = project

    def add_box(
        self,
        imported_image: ImportedImage,
        *,
        class_name: str,
        pixel_box: PixelBox,
    ) -> MetalDetectionBox:
        annotations = self.load(imported_image)
        annotation = MetalDetectionBox(
            class_id=_class_id(class_name),
            class_name=class_name,
            pixel_box=pixel_box,
        )
        annotations.append(annotation)
        self.save(imported_image, annotations)
        return annotation

    def undo_last(self, imported_image: ImportedImage) -> MetalDetectionBox | None:
        annotations = self.load(imported_image)
        if not annotations:
            return None
        removed = annotations.pop()
        self.save(imported_image, annotations)
        return removed

    def delete_box(
        self,
        imported_image: ImportedImage,
        *,
        index: int,
    ) -> MetalDetectionBox:
        annotations = self.load(imported_image)
        removed = annotations.pop(index)
        self.save(imported_image, annotations)
        return removed

    def load(self, imported_image: ImportedImage) -> list[MetalDetectionBox]:
        label_path = self._label_path(imported_image)
        if not label_path.exists():
            return []

        width, height = _normalized_size(imported_image)
        annotations: list[MetalDetectionBox] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id_text, x_center_text, y_center_text, width_text, height_text = (
                line.split()
            )
            class_id = int(class_id_text)
            annotations.append(
                MetalDetectionBox(
                    class_id=class_id,
                    class_name=CANONICAL_CLASSES[class_id],
                    pixel_box=_pixel_box_from_yolo(
                        image_width=width,
                        image_height=height,
                        x_center=float(x_center_text),
                        y_center=float(y_center_text),
                        box_width=float(width_text),
                        box_height=float(height_text),
                    ),
                )
            )
        return annotations

    def save(
        self,
        imported_image: ImportedImage,
        annotations: list[MetalDetectionBox],
    ) -> None:
        label_path = self._label_path(imported_image)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = _normalized_size(imported_image)
        label_text = "".join(
            _to_yolo_label_line(annotation, image_width=width, image_height=height)
            for annotation in annotations
        )
        label_path.write_text(label_text, encoding="utf-8")
        review_states = ReviewStateStore(self._project)
        if annotations:
            review_states.mark_labeled(imported_image)
        elif review_states.load(imported_image) != REVIEW_STATE_REVIEWED_EMPTY:
            review_states.mark_unreviewed(imported_image)

    def _label_path(self, imported_image: ImportedImage) -> Path:
        return self._project.path / "labels" / f"{imported_image.image_id}.txt"


def _class_id(class_name: str) -> int:
    try:
        return CANONICAL_CLASSES.index(class_name)
    except ValueError as error:
        raise ValueError(f"Unsupported Metal Detection Box class: {class_name}") from error


def _normalized_size(imported_image: ImportedImage) -> tuple[int, int]:
    metadata = _read_image_metadata(imported_image)
    normalized_size = metadata["normalized_size"]
    return int(normalized_size["width"]), int(normalized_size["height"])


def _read_image_metadata(imported_image: ImportedImage) -> dict:
    return json.loads(imported_image.metadata_path.read_text(encoding="utf-8"))


def _write_image_metadata(imported_image: ImportedImage, metadata: dict) -> None:
    imported_image.metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _to_yolo_label_line(
    annotation: MetalDetectionBox,
    *,
    image_width: int,
    image_height: int,
) -> str:
    box = annotation.pixel_box
    box_width = box.x_max - box.x_min
    box_height = box.y_max - box.y_min
    x_center = box.x_min + box_width / 2
    y_center = box.y_min + box_height / 2
    return (
        f"{annotation.class_id} "
        f"{x_center / image_width:.6f} "
        f"{y_center / image_height:.6f} "
        f"{box_width / image_width:.6f} "
        f"{box_height / image_height:.6f}\n"
    )


def _pixel_box_from_yolo(
    *,
    image_width: int,
    image_height: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
) -> PixelBox:
    width = box_width * image_width
    height = box_height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return PixelBox(
        x_min=round(center_x - width / 2),
        y_min=round(center_y - height / 2),
        x_max=round(center_x + width / 2),
        y_max=round(center_y + height / 2),
    )
