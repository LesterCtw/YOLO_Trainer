from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_METADATA_FILE = "yolo-trainer-project.json"
PROJECT_FORMAT = "yolo-trainer-project"
PROJECT_SCHEMA_VERSION = 1


class InvalidProjectError(ValueError):
    """Raised when a folder is not a supported YOLO Training Project."""


@dataclass(frozen=True)
class ImportedImage:
    image_id: str
    display_name: str
    source_path: Path
    normalized_image_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class YOLOTrainingProject:
    path: Path
    name: str
    image_count: int = 0
    imported_images: tuple[ImportedImage, ...] = ()


def create_project(path: Path | str, name: str | None = None) -> YOLOTrainingProject:
    project_path = Path(path)
    project_path.mkdir(parents=True, exist_ok=True)

    project_name = name or project_path.name
    metadata = {
        "format": PROJECT_FORMAT,
        "schema_version": PROJECT_SCHEMA_VERSION,
        "name": project_name,
    }
    _metadata_path(project_path).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return YOLOTrainingProject(path=project_path, name=project_name)


def open_project(path: Path | str) -> YOLOTrainingProject:
    project_path = Path(path)
    metadata = _read_metadata(project_path)
    imported_images = _read_imported_images(project_path)

    return YOLOTrainingProject(
        path=project_path,
        name=str(metadata["name"]),
        image_count=len(imported_images),
        imported_images=tuple(imported_images),
    )


def _metadata_path(project_path: Path) -> Path:
    return project_path / PROJECT_METADATA_FILE


def _read_metadata(project_path: Path) -> dict[str, Any]:
    metadata_path = _metadata_path(project_path)
    if not metadata_path.exists():
        raise InvalidProjectError(
            "Selected folder is not a YOLO Training Project: metadata file is missing."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InvalidProjectError(
            "Selected folder is not a valid YOLO Training Project: metadata is unreadable."
        ) from error

    if metadata.get("format") != PROJECT_FORMAT:
        raise InvalidProjectError("Unsupported YOLO Training Project folder.")
    if metadata.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise InvalidProjectError("Unsupported YOLO Training Project schema version.")
    if not metadata.get("name"):
        raise InvalidProjectError("YOLO Training Project metadata is missing a name.")
    return metadata


def _read_imported_images(project_path: Path) -> list[ImportedImage]:
    metadata_dir = project_path / "images" / "metadata"
    if not metadata_dir.exists():
        return []

    imported_images: list[ImportedImage] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        imported_images.append(
            ImportedImage(
                image_id=str(metadata["image_id"]),
                display_name=str(metadata["display_name"]),
                source_path=project_path / str(metadata["source_project_path"]),
                normalized_image_path=project_path
                / str(metadata["normalized_project_path"]),
                metadata_path=metadata_path,
            )
        )
    return imported_images
