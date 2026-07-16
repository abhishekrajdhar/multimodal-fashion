"""Image embedding generation with OpenCLIP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Final, Protocol, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME: Final[str] = "ViT-H-14"
DEFAULT_PRETRAINED: Final[str] = "laion2b_s32b_b79k"
EXPECTED_EMBEDDING_DIM: Final[int] = 1024


class OpenClipModel(Protocol):
    """Protocol describing the OpenCLIP image encoding surface."""

    def eval(self) -> torch.nn.Module: ...

    def requires_grad_(self, requires_grad: bool) -> torch.nn.Module: ...

    def encode_image(self, image: torch.Tensor) -> torch.Tensor: ...


class ImageEncoder:
    """Generates normalized OpenCLIP image embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: OpenClipModel | None = None
        self.preprocess: Callable[[Image.Image], torch.Tensor] | None = None

    def load_model(self) -> None:
        """Load the OpenCLIP model and preprocessing pipeline."""
        if self.model is not None and self.preprocess is not None:
            LOGGER.debug(
                "OpenCLIP model '%s' with weights '%s' is already loaded.",
                self.model_name,
                self.pretrained,
            )
            return

        LOGGER.info(
            "Loading OpenCLIP model '%s' with weights '%s' on device '%s'.",
            self.model_name,
            self.pretrained,
            self.device,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.preprocess = preprocess
        LOGGER.info("Image encoder loaded successfully.")

    def encode_image(self, image_path: Path, output_path: Path | None = None) -> np.ndarray:
        """Encode a single image and optionally save the embedding to disk."""
        embeddings = self.encode_batch(image_paths=[image_path], output_path=output_path)
        return embeddings[0]

    def encode_batch(
        self,
        image_paths: Sequence[Path],
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Encode a batch of images into normalized float32 embeddings."""
        self.load_model()

        if self.model is None or self.preprocess is None:
            raise RuntimeError("Image encoder failed to initialize.")

        resolved_paths = [self._validate_image_path(image_path=image_path) for image_path in image_paths]
        if not resolved_paths:
            empty_embeddings = np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
            if output_path is not None:
                self._save_embeddings(embeddings=empty_embeddings, output_path=output_path)
            return empty_embeddings

        LOGGER.info("Encoding %d images with batch size %d.", len(resolved_paths), self.batch_size)
        all_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for batch_start in range(0, len(resolved_paths), self.batch_size):
                batch_paths = resolved_paths[batch_start : batch_start + self.batch_size]
                image_tensors = [self._preprocess_image(image_path=path) for path in batch_paths]
                batch_tensor = torch.stack(image_tensors, dim=0).to(self.device)

                batch_embeddings = self.model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True).clamp(
                    min=1e-12,
                )

                batch_array = batch_embeddings.detach().cpu().to(torch.float32).numpy()
                self._validate_embedding_shape(embeddings=batch_array)
                all_embeddings.append(batch_array)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)

        if output_path is not None:
            self._save_embeddings(embeddings=embeddings, output_path=output_path)

        LOGGER.info("Generated embeddings with shape %s.", embeddings.shape)
        return embeddings

    def _validate_image_path(self, image_path: Path) -> Path:
        """Validate that the given image path exists and is a file."""
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.exists():
            message = f"Image file does not exist: {resolved_path}"
            LOGGER.error(message)
            raise FileNotFoundError(message)

        if not resolved_path.is_file():
            message = f"Image path is not a file: {resolved_path}"
            LOGGER.error(message)
            raise FileNotFoundError(message)

        return resolved_path

    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image for OpenCLIP inference."""
        if self.preprocess is None:
            raise RuntimeError("Preprocessing pipeline is not loaded.")

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            preprocessed_image = self.preprocess(rgb_image)

        return preprocessed_image

    def _validate_embedding_shape(self, embeddings: np.ndarray) -> None:
        """Validate the expected embedding dimension."""
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            message = (
                "Unexpected embedding shape "
                f"{embeddings.shape}; expected (*, {EXPECTED_EMBEDDING_DIM})."
            )
            LOGGER.error(message)
            raise ValueError(message)

    def _save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Persist embeddings as a float32 NumPy array."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings.astype(np.float32, copy=False))
        LOGGER.info("Saved embeddings to '%s'.", output_path)
