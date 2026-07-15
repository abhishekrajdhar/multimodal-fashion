"""Dataset utilities for discovering fashion images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Represents a discovered image in the dataset."""

    image_id: int
    image_path: Path
    filename: str


class FashionDataset:
    """Discovers image files for the indexing pipeline."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def load_records(self) -> list[ImageRecord]:
        """Recursively discover valid images under the dataset directory."""
        dataset_dir = self.data_dir.expanduser().resolve()
        self._validate_dataset_dir(dataset_dir=dataset_dir)

        candidate_paths = self._discover_image_paths(dataset_dir=dataset_dir)
        LOGGER.info(
            "Discovered %d candidate image files in '%s'.",
            len(candidate_paths),
            dataset_dir,
        )

        valid_paths: list[Path] = []
        for image_path in tqdm(candidate_paths, desc="Validating images", unit="image"):
            if self._is_valid_image(image_path=image_path):
                valid_paths.append(image_path)

        records = [
            ImageRecord(
                image_id=image_id,
                image_path=image_path,
                filename=image_path.name,
            )
            for image_id, image_path in enumerate(valid_paths)
        ]

        LOGGER.info(
            "Loaded %d valid images from '%s' after filtering corrupted files.",
            len(records),
            dataset_dir,
        )
        return records

    def _validate_dataset_dir(self, dataset_dir: Path) -> None:
        """Ensure the dataset directory exists and is readable."""
        if not dataset_dir.exists():
            message = f"Dataset directory does not exist: {dataset_dir}"
            LOGGER.error(message)
            raise FileNotFoundError(message)

        if not dataset_dir.is_dir():
            message = f"Dataset path is not a directory: {dataset_dir}"
            LOGGER.error(message)
            raise NotADirectoryError(message)

    def _discover_image_paths(self, dataset_dir: Path) -> list[Path]:
        """Return sorted candidate image paths under the dataset directory."""
        image_paths = [
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return sorted(image_paths)

    def _is_valid_image(self, image_path: Path) -> bool:
        """Return whether the file is a readable image."""
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            LOGGER.warning("Skipping corrupted image '%s': %s", image_path, error)
            return False

        return True
