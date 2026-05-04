import json

import numpy as np
from PIL import Image
import tifffile

from yolo_trainer.image_import import import_stem_zc_images
from yolo_trainer.project import create_project, open_project


def test_tiff_import_preserves_source_creates_normalized_png_and_persists_queue(
    tmp_path,
) -> None:
    project = create_project(tmp_path / "project", name="Import Project")
    source = tmp_path / "source.tif"
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(source, pixels)

    result = import_stem_zc_images(project, [source])
    reloaded = open_project(project.path)

    assert result.imported_count == 1
    assert result.failed == []
    imported = reloaded.imported_images[0]

    assert imported.display_name == "source.tif"
    assert imported.source_path.exists()
    assert imported.source_path.read_bytes() == source.read_bytes()
    assert imported.normalized_image_path.exists()

    normalized = Image.open(imported.normalized_image_path)
    assert normalized.mode == "L"
    assert normalized.size == (4, 3)

    metadata = json.loads(imported.metadata_path.read_text(encoding="utf-8"))
    assert metadata["display_name"] == "source.tif"
    assert metadata["original_size"] == {"width": 4, "height": 3}
    assert metadata["normalized_size"] == {"width": 4, "height": 3}


def test_import_uses_deterministic_normalization_and_coordinate_mapping(
    tmp_path,
) -> None:
    pixels = np.array(
        [
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 65535],
        ],
        dtype=np.uint16,
    )
    source = tmp_path / "deterministic.tif"
    tifffile.imwrite(source, pixels)

    first_project = create_project(tmp_path / "first", name="First")
    second_project = create_project(tmp_path / "second", name="Second")

    import_stem_zc_images(first_project, [source])
    import_stem_zc_images(second_project, [source])
    first_import = open_project(first_project.path).imported_images[0]
    second_import = open_project(second_project.path).imported_images[0]

    assert first_import.normalized_image_path.read_bytes() == (
        second_import.normalized_image_path.read_bytes()
    )

    metadata = json.loads(first_import.metadata_path.read_text(encoding="utf-8"))
    assert metadata["normalization"] == {
        "method": "percentile",
        "lower_percentile": 1,
        "upper_percentile": 99,
        "lower_value": 0.8,
        "upper_value": 60297.8,
    }
    assert metadata["coordinate_mapping"] == {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }


def test_dm3_import_uses_controlled_reader(tmp_path) -> None:
    project = create_project(tmp_path / "project", name="DM3 Project")
    source = tmp_path / "controlled.dm3"
    source.write_bytes(b"controlled dm3 fixture")

    result = import_stem_zc_images(
        project,
        [source],
        dm3_reader=lambda path: np.arange(6, dtype=np.uint16).reshape(2, 3),
    )
    imported = open_project(project.path).imported_images[0]

    assert result.imported_count == 1
    assert result.failed == []
    assert imported.display_name == "controlled.dm3"
    assert imported.source_path.read_bytes() == source.read_bytes()
    assert Image.open(imported.normalized_image_path).size == (3, 2)


def test_unsupported_and_non_2d_inputs_fail_without_corrupting_project(tmp_path) -> None:
    project = create_project(tmp_path / "project", name="Import Project")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    volume = tmp_path / "volume.tif"
    tifffile.imwrite(volume, np.zeros((2, 3, 4), dtype=np.uint16))

    result = import_stem_zc_images(project, [unsupported, volume])
    reloaded = open_project(project.path)

    assert result.imported_count == 0
    assert [failure.source_path.name for failure in result.failed] == [
        "notes.txt",
        "volume.tif",
    ]
    assert "Unsupported STEM ZC Image format" in result.failed[0].message
    assert "Only 2D STEM ZC Images are supported" in result.failed[1].message
    assert reloaded.imported_images == ()
