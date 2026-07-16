"""Dataset utilities for discovering fashion images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import get_context
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


def _validate_image_worker(image_path_str: str) -> tuple[str, bool, str | None]:
    """Validate an image path in a worker process."""
    image_path = Path(image_path_str)
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        return (image_path_str, False, str(error))

    return (image_path_str, True, None)


class FashionDataset:
    """Discovers image files for the indexing pipeline."""

    def __init__(self, data_dir: Path, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than 0.")

        self.data_dir = data_dir
        self.num_workers = num_workers

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

        valid_paths = self._validate_candidate_paths(candidate_paths=candidate_paths)

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

    def _validate_candidate_paths(self, candidate_paths: list[Path]) -> list[Path]:
        """Validate candidate image paths, optionally using multiprocessing."""
        if self.num_workers == 1 or len(candidate_paths) <= 1:
            valid_paths: list[Path] = []
            for image_path in tqdm(candidate_paths, desc="Validating images", unit="image"):
                if self._is_valid_image(image_path=image_path):
                    valid_paths.append(image_path)

            return valid_paths

        valid_paths = []
        with get_context("spawn").Pool(processes=self.num_workers) as pool:
            results = pool.imap(
                _validate_image_worker,
                (str(image_path) for image_path in candidate_paths),
                chunksize=max(1, len(candidate_paths) // (self.num_workers * 4)),
            )
            for image_path_str, is_valid, error_message in tqdm(
                results,
                total=len(candidate_paths),
                desc="Validating images",
                unit="image",
            ):
                image_path = Path(image_path_str)
                if is_valid:
                    valid_paths.append(image_path)
                    continue

                LOGGER.warning("Skipping corrupted image '%s': %s", image_path, error_message)

        return valid_paths

    def _is_valid_image(self, image_path: Path) -> bool:
        """Return whether the file is a readable image."""
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            LOGGER.warning("Skipping corrupted image '%s': %s", image_path, error)
            return False

        return True
