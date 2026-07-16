"""Text embedding generation with OpenCLIP."""

from __future__ import annotations

import logging
from typing import Any, Final, Protocol, Sequence

import numpy as np
import open_clip
import torch

from indexer.image_encoder import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    EXPECTED_EMBEDDING_DIM,
)

LOGGER = logging.getLogger(__name__)


class OpenClipTextModel(Protocol):
    """Protocol describing the OpenCLIP text encoding surface."""

    def eval(self) -> torch.nn.Module: ...

    def requires_grad_(self, requires_grad: bool) -> torch.nn.Module: ...

    def encode_text(self, text: torch.Tensor) -> torch.Tensor: ...


class TextEncoder:
    """Generates normalized OpenCLIP text embeddings."""

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
        self.model: OpenClipTextModel | None = None
        self.tokenizer: Any | None = None

    def load_model(self) -> None:
        """Load the shared OpenCLIP model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            LOGGER.debug(
                "OpenCLIP text model '%s' with weights '%s' is already loaded.",
                self.model_name,
                self.pretrained,
            )
            return

        LOGGER.info(
            "Loading OpenCLIP text model '%s' with weights '%s' on device '%s'.",
            self.model_name,
            self.pretrained,
            self.device,
        )
        model = open_clip.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.tokenizer = tokenizer
        LOGGER.info("Text encoder loaded successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single query string into a normalized float32 embedding."""
        embeddings = self.encode_batch([text])
        return embeddings[0]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of natural-language queries into normalized embeddings."""
        self.load_model()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text encoder failed to initialize.")

        normalized_texts = [self._normalize_text(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)

        LOGGER.info("Encoding %d text queries with batch size %d.", len(normalized_texts), self.batch_size)
        all_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for batch_start in range(0, len(normalized_texts), self.batch_size):
                batch_texts = normalized_texts[batch_start : batch_start + self.batch_size]
                tokenized_text = self.tokenizer(batch_texts).to(self.device)

                batch_embeddings = self.model.encode_text(tokenized_text)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True).clamp(
                    min=1e-12,
                )

                batch_array = batch_embeddings.detach().cpu().to(torch.float32).numpy()
                self._validate_embedding_shape(batch_array)
                all_embeddings.append(batch_array)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
        LOGGER.info("Generated text embeddings with shape %s.", embeddings.shape)
        return embeddings

    def _normalize_text(self, text: str) -> str:
        """Validate and normalize a query string."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Text queries must be non-empty strings.")

        return normalized_text

    def _validate_embedding_shape(self, embeddings: np.ndarray) -> None:
        """Validate the expected embedding dimension."""
        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            message = (
                "Unexpected embedding shape "
                f"{embeddings.shape}; expected (*, {EXPECTED_EMBEDDING_DIM})."
            )
            LOGGER.error(message)
            raise ValueError(message)
