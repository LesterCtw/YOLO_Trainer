import pytest

from yolo_trainer.project import InvalidProjectError, create_project, open_project


def test_yolo_training_project_metadata_persists_across_loads(tmp_path) -> None:
    project_path = tmp_path / "sample-project"

    created = create_project(project_path, name="Sample Project")
    loaded = open_project(project_path)

    assert created.path == project_path
    assert loaded.path == project_path
    assert loaded.name == "Sample Project"
    assert loaded.image_count == 0


def test_invalid_project_folder_is_rejected_with_clear_message(tmp_path) -> None:
    invalid_project_path = tmp_path / "plain-folder"
    invalid_project_path.mkdir()

    with pytest.raises(InvalidProjectError, match="not a YOLO Training Project"):
        open_project(invalid_project_path)
