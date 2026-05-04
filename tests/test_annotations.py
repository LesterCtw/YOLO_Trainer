import numpy as np
import tifffile

from yolo_trainer.annotations import (
    CANONICAL_CLASSES,
    AnnotationStore,
    PixelBox,
    ReviewStateStore,
)
from yolo_trainer.image_import import import_stem_zc_images
from yolo_trainer.project import create_project, open_project


def _import_one_image(tmp_path):
    project = create_project(tmp_path / "project", name="Annotation Project")
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.arange(16, dtype=np.uint16).reshape(4, 4))
    import_stem_zc_images(project, [source])
    return open_project(project.path).imported_images[0], project


def _import_two_images(tmp_path):
    project = create_project(tmp_path / "project", name="Annotation Project")
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    tifffile.imwrite(first, np.arange(16, dtype=np.uint16).reshape(4, 4))
    tifffile.imwrite(second, np.arange(16, 32, dtype=np.uint16).reshape(4, 4))
    import_stem_zc_images(project, [first, second])
    reopened = open_project(project.path)
    return reopened.imported_images, reopened


def test_imported_images_start_as_unreviewed_and_persist_review_state(tmp_path) -> None:
    imported_image, project = _import_one_image(tmp_path)

    assert ReviewStateStore(project).load(imported_image) == "unreviewed"

    reopened_project = open_project(project.path)
    reopened_image = reopened_project.imported_images[0]

    assert ReviewStateStore(reopened_project).load(reopened_image) == "unreviewed"


def test_annotation_store_writes_and_reads_yolo_labels_with_canonical_class_ids(
    tmp_path,
) -> None:
    imported_image, project = _import_one_image(tmp_path)
    store = AnnotationStore(project)

    store.add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )

    assert CANONICAL_CLASSES == (
        "xsection_metal",
        "alongline_metal",
        "xsection_via",
        "alongline_via",
    )

    label_path = project.path / "labels" / f"{imported_image.image_id}.txt"
    assert (
        label_path.read_text(encoding="utf-8")
        == "0 0.250000 0.250000 0.500000 0.500000\n"
    )

    reloaded_annotations = AnnotationStore(project).load(imported_image)
    assert len(reloaded_annotations) == 1
    assert reloaded_annotations[0].class_id == 0
    assert reloaded_annotations[0].class_name == "xsection_metal"
    assert reloaded_annotations[0].pixel_box == PixelBox(0, 0, 2, 2)


def test_annotation_changes_update_labeled_state_without_implicit_reviewed_empty(
    tmp_path,
) -> None:
    imported_image, project = _import_one_image(tmp_path)
    annotations = AnnotationStore(project)
    review_states = ReviewStateStore(project)

    annotations.add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )

    assert review_states.load(imported_image) == "labeled"

    annotations.undo_last(imported_image)

    assert annotations.load(imported_image) == []
    assert review_states.load(imported_image) == "unreviewed"


def test_user_can_mark_an_image_as_reviewed_empty(tmp_path) -> None:
    imported_image, project = _import_one_image(tmp_path)

    ReviewStateStore(project).mark_reviewed_empty(imported_image)

    assert ReviewStateStore(project).load(imported_image) == "reviewed_empty"

    reopened_project = open_project(project.path)
    reopened_image = reopened_project.imported_images[0]

    assert ReviewStateStore(reopened_project).load(reopened_image) == "reviewed_empty"


def test_empty_annotation_autosave_preserves_reviewed_empty_state(tmp_path) -> None:
    imported_image, project = _import_one_image(tmp_path)
    review_states = ReviewStateStore(project)

    review_states.mark_reviewed_empty(imported_image)
    AnnotationStore(project).save(imported_image, [])

    assert review_states.load(imported_image) == "reviewed_empty"


def test_review_progress_counts_reviewed_and_unreviewed_images(tmp_path) -> None:
    imported_images, project = _import_two_images(tmp_path)
    first, second = imported_images
    review_states = ReviewStateStore(project)

    review_states.mark_reviewed_empty(first)
    AnnotationStore(project).add_box(
        second,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=2, y_max=2),
    )

    progress = review_states.progress(imported_images)

    assert progress.total_count == 2
    assert progress.reviewed_count == 2
    assert progress.unreviewed_count == 0


def test_overlapping_boxes_undo_and_delete_autosave(tmp_path) -> None:
    imported_image, project = _import_one_image(tmp_path)
    store = AnnotationStore(project)

    first = store.add_box(
        imported_image,
        class_name="xsection_metal",
        pixel_box=PixelBox(x_min=0, y_min=0, x_max=3, y_max=3),
    )
    second = store.add_box(
        imported_image,
        class_name="alongline_via",
        pixel_box=PixelBox(x_min=1, y_min=1, x_max=4, y_max=4),
    )

    assert [box.pixel_box for box in AnnotationStore(project).load(imported_image)] == [
        first.pixel_box,
        second.pixel_box,
    ]

    undone = store.undo_last(imported_image)
    assert undone == second
    assert [box.pixel_box for box in AnnotationStore(project).load(imported_image)] == [
        first.pixel_box,
    ]

    third = store.add_box(
        imported_image,
        class_name="xsection_via",
        pixel_box=PixelBox(x_min=1, y_min=0, x_max=4, y_max=2),
    )
    deleted = store.delete_box(imported_image, index=0)

    assert deleted == first
    assert AnnotationStore(project).load(imported_image) == [third]
