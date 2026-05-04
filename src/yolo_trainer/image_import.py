from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
from PIL import Image
import tifffile

from yolo_trainer.project import YOLOTrainingProject


@dataclass(frozen=True)
class ImportFailure:
    source_path: Path
    message: str


@dataclass(frozen=True)
class ImportResult:
    imported_count: int
    failed: list[ImportFailure]


def import_stem_zc_images(
    project: YOLOTrainingProject,
    source_paths: list[Path] | tuple[Path, ...],
    *,
    dm3_reader: Callable[[Path], np.ndarray] | None = None,
) -> ImportResult:
    imported_count = 0
    failures: list[ImportFailure] = []

    for source_path in source_paths:
        source = Path(source_path)
        try:
            pixels = _read_supported_image(source, dm3_reader=dm3_reader)
            _import_one_image(project, source, pixels)
        except (OSError, ValueError) as error:
            failures.append(ImportFailure(source_path=source, message=str(error)))
            continue
        imported_count += 1

    return ImportResult(imported_count=imported_count, failed=failures)


def _read_supported_image(
    source_path: Path,
    *,
    dm3_reader: Callable[[Path], np.ndarray] | None,
) -> np.ndarray:
    if source_path.suffix.lower() in {".tif", ".tiff"}:
        pixels = tifffile.imread(source_path)
    elif source_path.suffix.lower() == ".dm3":
        if dm3_reader is None:
            raise ValueError("DM3 import requires a configured DM3 reader.")
        pixels = dm3_reader(source_path)
    else:
        raise ValueError(f"Unsupported STEM ZC Image format: {source_path.suffix}")

    pixels = np.asarray(pixels)
    if pixels.ndim != 2:
        raise ValueError("Only 2D STEM ZC Images are supported.")
    return pixels


def _import_one_image(
    project: YOLOTrainingProject,
    source_path: Path,
    pixels: np.ndarray,
) -> None:
    image_id = _image_id(source_path)
    image_root = project.path / "images"
    source_dir = image_root / "sources"
    normalized_dir = image_root / "normalized"
    metadata_dir = image_root / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_project_path = source_dir / f"{image_id}{source_path.suffix.lower()}"
    normalized_project_path = normalized_dir / f"{image_id}.png"
    metadata_path = metadata_dir / f"{image_id}.json"

    shutil.copy2(source_path, source_project_path)
    lower_value, upper_value = normalization_bounds(pixels)
    normalized_pixels = normalize_to_uint8(
        pixels,
        lower_value=lower_value,
        upper_value=upper_value,
    )
    Image.fromarray(normalized_pixels).save(normalized_project_path)

    height, width = pixels.shape
    metadata = {
        "image_id": image_id,
        "display_name": source_path.name,
        "source_project_path": str(source_project_path.relative_to(project.path)),
        "normalized_project_path": str(
            normalized_project_path.relative_to(project.path)
        ),
        "original_size": {"width": width, "height": height},
        "normalized_size": {"width": width, "height": height},
        "normalization": {
            "method": "percentile",
            "lower_percentile": 1,
            "upper_percentile": 99,
            "lower_value": lower_value,
            "upper_value": upper_value,
        },
        "coordinate_mapping": {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def normalization_bounds(pixels: np.ndarray) -> tuple[float, float]:
    values = pixels.astype(np.float64)
    return (
        round(float(np.percentile(values, 1)), 6),
        round(float(np.percentile(values, 99)), 6),
    )


def normalize_to_uint8(
    pixels: np.ndarray,
    *,
    lower_value: float | None = None,
    upper_value: float | None = None,
) -> np.ndarray:
    values = pixels.astype(np.float64)
    lower = lower_value
    upper = upper_value
    if lower is None or upper is None:
        lower, upper = normalization_bounds(pixels)
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)

    normalized = (values - lower) / (upper - lower)
    clipped = np.clip(normalized, 0.0, 1.0)
    return np.rint(clipped * 255).astype(np.uint8)


def _image_id(source_path: Path) -> str:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return digest[:16]
