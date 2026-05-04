from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from yolo_trainer.annotations import (
    CANONICAL_CLASSES,
    REVIEW_STATE_LABELED,
    REVIEW_STATE_REVIEWED_EMPTY,
    AnnotationStore,
    ReviewStateStore,
)
from yolo_trainer.project import ImportedImage, YOLOTrainingProject


DATASET_SPLIT_SEED = 20260504


@dataclass(frozen=True)
class DatasetExportResult:
    export_path: Path
    included_count: int
    skipped_unreviewed_count: int
    train_count: int
    val_count: int


def export_dataset(
    project: YOLOTrainingProject,
    export_path: Path | str,
    *,
    seed: int = DATASET_SPLIT_SEED,
) -> DatasetExportResult:
    destination = Path(export_path)
    _create_dataset_dirs(destination)

    review_states = ReviewStateStore(project)
    included_images = [
        image
        for image in project.imported_images
        if review_states.load(image)
        in {REVIEW_STATE_LABELED, REVIEW_STATE_REVIEWED_EMPTY}
    ]
    skipped_unreviewed_count = len(project.imported_images) - len(included_images)

    train_images, val_images = _split_train_val(included_images, seed=seed)
    for image in train_images:
        _export_image(project, image, destination=destination, split="train")
    for image in val_images:
        _export_image(project, image, destination=destination, split="val")

    _write_dataset_yaml(destination)
    return DatasetExportResult(
        export_path=destination,
        included_count=len(included_images),
        skipped_unreviewed_count=skipped_unreviewed_count,
        train_count=len(train_images),
        val_count=len(val_images),
    )


def _create_dataset_dirs(destination: Path) -> None:
    for relative_path in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        managed_dir = destination / relative_path
        if managed_dir.exists():
            shutil.rmtree(managed_dir)
        managed_dir.mkdir(parents=True, exist_ok=True)


def _split_train_val(
    images: list[ImportedImage],
    *,
    seed: int,
) -> tuple[list[ImportedImage], list[ImportedImage]]:
    shuffled_images = list(images)
    random.Random(seed).shuffle(shuffled_images)
    if len(shuffled_images) <= 1:
        return shuffled_images, []

    val_count = max(1, int(len(shuffled_images) * 0.2))
    train_count = len(shuffled_images) - val_count
    return shuffled_images[:train_count], shuffled_images[train_count:]


def _export_image(
    project: YOLOTrainingProject,
    imported_image: ImportedImage,
    *,
    destination: Path,
    split: str,
) -> None:
    target_stem = imported_image.image_id
    shutil.copy2(
        imported_image.normalized_image_path,
        destination / "images" / split / f"{target_stem}.png",
    )
    label_text = _label_text(project, imported_image)
    (destination / "labels" / split / f"{target_stem}.txt").write_text(
        label_text,
        encoding="utf-8",
    )


def _label_text(project: YOLOTrainingProject, imported_image: ImportedImage) -> str:
    if ReviewStateStore(project).load(imported_image) == REVIEW_STATE_REVIEWED_EMPTY:
        return ""

    source_label_path = project.path / "labels" / f"{imported_image.image_id}.txt"
    if source_label_path.exists():
        return source_label_path.read_text(encoding="utf-8")
    annotations = AnnotationStore(project).load(imported_image)
    if annotations:
        raise ValueError(f"Missing saved labels for image: {imported_image.image_id}")
    return ""


def _write_dataset_yaml(destination: Path) -> None:
    names = "".join(
        f"  {class_id}: {class_name}\n"
        for class_id, class_name in enumerate(CANONICAL_CLASSES)
    )
    (destination / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}",
        encoding="utf-8",
    )
