from pathlib import Path

import numpy as np
import tifffile

from yolo_trainer.annotations import AnnotationStore, PixelBox, ReviewStateStore
from yolo_trainer.dataset_export import export_dataset
from yolo_trainer.image_import import import_stem_zc_images
from yolo_trainer.project import create_project, open_project


def _import_image(tmp_path, project, name: str, start: int = 0):
    source = tmp_path / name
    pixels = np.arange(start, start + 16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(source, pixels)
    import_stem_zc_images(project, [source])
    return open_project(project.path).imported_images[-1]


def test_export_writes_ultralytics_dataset_for_labeled_images(tmp_path) -> None:
    project = create_project(tmp_path / "project", name="Dataset Project")
    imported_image = _import_image(tmp_path, project, "labeled.tif")
    project = open_project(project.path)
    imported_image = project.imported_images[0]
    AnnotationStore(project).add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )

    result = export_dataset(project, tmp_path / "export")

    assert result.included_count == 1
    assert result.skipped_unreviewed_count == 0
    assert result.train_count == 1
    assert result.val_count == 0
    assert (tmp_path / "export" / "images" / "train").is_dir()
    assert (tmp_path / "export" / "images" / "val").is_dir()
    assert (tmp_path / "export" / "labels" / "train").is_dir()
    assert (tmp_path / "export" / "labels" / "val").is_dir()
    assert len(list((tmp_path / "export" / "images" / "train").glob("*.png"))) == 1
    assert len(list((tmp_path / "export" / "labels" / "train").glob("*.txt"))) == 1

    label_text = next(
        (tmp_path / "export" / "labels" / "train").glob("*.txt")
    ).read_text(encoding="utf-8")
    assert label_text == "0 0.250000 0.250000 0.500000 0.500000\n"

    assert (tmp_path / "export" / "dataset.yaml").read_text(encoding="utf-8") == (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: xsection_metal\n"
        "  1: alongline_metal\n"
        "  2: xsection_via\n"
        "  3: alongline_via\n"
    )


def test_export_skips_unreviewed_images_and_writes_reviewed_empty_negative_examples(
    tmp_path,
) -> None:
    project = create_project(tmp_path / "project", name="Dataset Project")
    labeled_image = _import_image(tmp_path, project, "labeled.tif", start=0)
    reviewed_empty_image = _import_image(tmp_path, project, "empty.tif", start=16)
    unreviewed_image = _import_image(tmp_path, project, "unreviewed.tif", start=32)
    project = open_project(project.path)
    labeled_image, reviewed_empty_image, unreviewed_image = project.imported_images
    AnnotationStore(project).add_box(
        labeled_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )
    AnnotationStore(project).add_box(
        reviewed_empty_image,
        class_name="xsection_via",
        pixel_box=PixelBox(x_min=1, y_min=1, x_max=3, y_max=3),
    )
    ReviewStateStore(project).mark_reviewed_empty(reviewed_empty_image)

    result = export_dataset(project, tmp_path / "export")

    assert result.included_count == 2
    assert result.skipped_unreviewed_count == 1
    exported_image_ids = _exported_stems(tmp_path / "export" / "images")
    exported_label_ids = _exported_stems(tmp_path / "export" / "labels")
    assert exported_image_ids == {labeled_image.image_id, reviewed_empty_image.image_id}
    assert exported_label_ids == {labeled_image.image_id, reviewed_empty_image.image_id}
    assert unreviewed_image.image_id not in exported_image_ids

    reviewed_empty_label = next(
        (tmp_path / "export" / "labels").glob(f"*/{reviewed_empty_image.image_id}.txt")
    )
    assert reviewed_empty_label.read_text(encoding="utf-8") == ""


def test_export_uses_deterministic_80_20_image_level_train_val_split(tmp_path) -> None:
    project = create_project(tmp_path / "project", name="Dataset Project")
    for index in range(5):
        _import_image(tmp_path, project, f"reviewed-{index}.tif", start=index * 16)
    project = open_project(project.path)
    review_states = ReviewStateStore(project)
    for imported_image in project.imported_images:
        review_states.mark_reviewed_empty(imported_image)

    first_result = export_dataset(project, tmp_path / "first-export")
    second_result = export_dataset(project, tmp_path / "second-export")

    assert first_result.train_count == 4
    assert first_result.val_count == 1
    assert second_result.train_count == 4
    assert second_result.val_count == 1
    assert _exported_stems(tmp_path / "first-export" / "images" / "train") == (
        _exported_stems(tmp_path / "second-export" / "images" / "train")
    )
    assert _exported_stems(tmp_path / "first-export" / "images" / "val") == (
        _exported_stems(tmp_path / "second-export" / "images" / "val")
    )


def test_export_replaces_stale_dataset_files_when_reusing_export_directory(
    tmp_path,
) -> None:
    project = create_project(tmp_path / "project", name="Dataset Project")
    imported_image = _import_image(tmp_path, project, "labeled.tif")
    project = open_project(project.path)
    imported_image = project.imported_images[0]
    AnnotationStore(project).add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )
    export_path = tmp_path / "export"

    first_result = export_dataset(project, export_path)
    ReviewStateStore(project).mark_unreviewed(imported_image)
    second_result = export_dataset(project, export_path)

    assert first_result.included_count == 1
    assert second_result.included_count == 0
    assert _exported_stems(export_path / "images") == set()
    assert _exported_stems(export_path / "labels") == set()


def _exported_stems(root: Path) -> set[str]:
    return {path.stem for path in root.rglob("*") if path.is_file()}
