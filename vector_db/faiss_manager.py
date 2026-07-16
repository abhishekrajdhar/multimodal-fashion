"""FAISS index management for multimodal retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import faiss
import numpy as np

from metadata import ImageMetadata

LOGGER = logging.getLogger(__name__)
DEFAULT_METADATA_CACHE_SIZE: Final[int] = 4096
INDEX_FILENAME: Final[str] = "faiss.index"
METADATA_MANIFEST_FILENAME: Final[str] = "metadata.json"
VECTOR_MAPPING_FILENAME: Final[str] = "mapping.json"
METADATA_DIRNAME: Final[str] = "metadata"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represents a single vector search hit."""

    vector_id: int
    image_id: int
    score: float
    metadata: ImageMetadata


class FaissManager:
    """Manages a FAISS IndexFlatIP and external metadata storage."""

    def __init__(
        self,
        index_dir: Path,
        embedding_dim: int | None = None,
        metadata_cache_size: int = DEFAULT_METADATA_CACHE_SIZE,
    ) -> None:
        if embedding_dim is not None and embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0 when provided.")

        if metadata_cache_size <= 0:
            raise ValueError("metadata_cache_size must be greater than 0.")

        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.metadata_cache_size = metadata_cache_size
        self.index: faiss.IndexFlatIP | None = (
            faiss.IndexFlatIP(embedding_dim) if embedding_dim is not None else None
        )
        self.vector_id_to_image_id: list[int] = []
        self.known_image_ids: set[int] = set()
        self.metadata_cache: dict[int, ImageMetadata] = {}

    def add(self, embeddings: np.ndarray, metadata_items: Sequence[ImageMetadata]) -> list[int]:
        """Add normalized embeddings and external metadata to the index."""
        validated_embeddings = self._prepare_embeddings(embeddings=embeddings, normalize=True)
        if validated_embeddings.shape[0] != len(metadata_items):
            raise ValueError("The number of embeddings must match the number of metadata items.")

        if validated_embeddings.shape[0] == 0:
            return []

        self._ensure_index(embedding_dim=validated_embeddings.shape[1])
        if self.index is None:
            raise RuntimeError("FAISS index failed to initialize.")

        starting_vector_id = self.index.ntotal
        vector_ids = list(range(starting_vector_id, starting_vector_id + validated_embeddings.shape[0]))

        normalized_metadata_items = self._normalize_metadata_items(
            metadata_items=metadata_items,
            normalized_embeddings=validated_embeddings,
        )
        self._validate_new_image_ids(metadata_items=normalized_metadata_items)

        self.index.add(validated_embeddings)
        self.vector_id_to_image_id.extend(item.image_id for item in normalized_metadata_items)
        self.known_image_ids.update(item.image_id for item in normalized_metadata_items)

        for metadata_item in normalized_metadata_items:
            self._write_metadata(metadata=metadata_item)
            self._update_metadata_cache(metadata=metadata_item)

        LOGGER.info(
            "Added %d embeddings to IndexFlatIP. Index now contains %d vectors.",
            validated_embeddings.shape[0],
            self.index.ntotal,
        )
        return vector_ids

    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[SearchResult]]:
        """Search the index with normalized query embeddings."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        if self.index.ntotal == 0:
            LOGGER.warning("Search requested on an empty index.")
            return []

        prepared_queries = self._prepare_embeddings(embeddings=query_embeddings, normalize=True)
        if prepared_queries.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension {prepared_queries.shape[1]} does not match index "
                f"dimension {self.index.d}.",
            )

        search_k = min(top_k, self.index.ntotal)
        scores, vector_ids = self.index.search(prepared_queries, search_k)
        results: list[list[SearchResult]] = []

        for query_scores, query_vector_ids in zip(scores, vector_ids, strict=True):
            query_results: list[SearchResult] = []
            for score, vector_id in zip(query_scores, query_vector_ids, strict=True):
                if vector_id < 0:
                    continue

                image_id = self.vector_id_to_image_id[vector_id]
                metadata = self._get_metadata(image_id=image_id)
                query_results.append(
                    SearchResult(
                        vector_id=int(vector_id),
                        image_id=image_id,
                        score=float(score),
                        metadata=metadata,
                    ),
                )

            results.append(query_results)

        LOGGER.info(
            "Search completed for %d queries with top_k=%d.",
            prepared_queries.shape[0],
            top_k,
        )
        return results

    def save(self) -> None:
        """Persist the index and external mappings to disk."""
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path()))
        self._write_vector_mapping()
        self._write_manifest()
        LOGGER.info("Saved FAISS index state to '%s'.", self.index_dir)

    def load(self) -> None:
        """Load the index and external mappings from disk."""
        index_path = self._index_path()
        mapping_path = self._vector_mapping_path()
        metadata_manifest_path = self._metadata_manifest_path()

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file does not exist: {index_path}")

        if not mapping_path.exists():
            raise FileNotFoundError(f"Vector mapping file does not exist: {mapping_path}")

        if not metadata_manifest_path.exists():
            raise FileNotFoundError(f"Metadata manifest file does not exist: {metadata_manifest_path}")

        with metadata_manifest_path.open("r", encoding="utf-8") as file_handle:
            metadata_manifest = json.load(file_handle)

        if not isinstance(metadata_manifest, dict):
            raise ValueError("Metadata manifest JSON must contain an object.")

        index = faiss.read_index(str(index_path))
        if not isinstance(index, faiss.IndexFlatIP):
            raise TypeError(
                f"Expected a FAISS IndexFlatIP at '{index_path}', got '{type(index).__name__}'.",
            )

        with mapping_path.open("r", encoding="utf-8") as file_handle:
            mapping_payload = json.load(file_handle)

        if not isinstance(mapping_payload, dict):
            raise ValueError("Vector mapping JSON must contain an object.")

        image_ids = mapping_payload.get("image_ids")
        if not isinstance(image_ids, list):
            raise ValueError("Vector mapping JSON must contain an 'image_ids' list.")

        vector_id_to_image_id = [int(image_id) for image_id in image_ids]
        if index.ntotal != len(vector_id_to_image_id):
            raise ValueError(
                "Loaded FAISS index size does not match vector-to-image mapping length: "
                f"{index.ntotal} != {len(vector_id_to_image_id)}."
            )

        manifest_embedding_dim = metadata_manifest.get("embedding_dim")
        if manifest_embedding_dim is not None and int(manifest_embedding_dim) != index.d:
            raise ValueError(
                "Metadata manifest embedding dimension does not match loaded FAISS index dimension: "
                f"{manifest_embedding_dim} != {index.d}.",
            )

        self.index = index
        self.embedding_dim = index.d
        self.vector_id_to_image_id = vector_id_to_image_id
        self.known_image_ids = set(vector_id_to_image_id)
        self.metadata_cache.clear()
        LOGGER.info(
            "Loaded IndexFlatIP with %d vectors and dimension %d from '%s'.",
            self.index.ntotal,
            self.index.d,
            self.index_dir,
        )

    def _ensure_index(self, embedding_dim: int) -> None:
        """Initialize the FAISS index if needed and validate dimensions."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.embedding_dim = embedding_dim
            LOGGER.info("Initialized IndexFlatIP with dimension %d.", embedding_dim)
            return

        if self.index.d != embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding_dim} does not match index dimension {self.index.d}.",
            )

    def _prepare_embeddings(self, embeddings: np.ndarray, normalize: bool) -> np.ndarray:
        """Validate embeddings and return a contiguous float32 matrix."""
        prepared_embeddings = np.asarray(embeddings, dtype=np.float32)
        if prepared_embeddings.ndim == 1:
            prepared_embeddings = prepared_embeddings.reshape(1, -1)

        if prepared_embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 1D or 2D array.")

        if prepared_embeddings.shape[1] == 0:
            raise ValueError("Embeddings must have a non-zero dimension.")

        prepared_embeddings = np.ascontiguousarray(prepared_embeddings)
        if normalize and prepared_embeddings.shape[0] > 0:
            faiss.normalize_L2(prepared_embeddings)

        return prepared_embeddings

    def _normalize_metadata_items(
        self,
        metadata_items: Sequence[ImageMetadata],
        normalized_embeddings: np.ndarray,
    ) -> list[ImageMetadata]:
        """Return metadata records with normalized CLIP embeddings."""
        normalized_items: list[ImageMetadata] = []
        for metadata_item, normalized_embedding in zip(
            metadata_items,
            normalized_embeddings,
            strict=True,
        ):
            normalized_items.append(
                ImageMetadata(
                    image_id=metadata_item.image_id,
                    image_path=metadata_item.image_path,
                    attributes=metadata_item.attributes,
                    clip_embedding=normalized_embedding.astype(np.float32, copy=False).tolist(),
                ),
            )

        return normalized_items

    def _validate_new_image_ids(self, metadata_items: Sequence[ImageMetadata]) -> None:
        """Ensure image identifiers are unique across the index."""
        batch_image_ids: set[int] = set()
        for metadata_item in metadata_items:
            image_id = metadata_item.image_id
            if image_id in batch_image_ids:
                raise ValueError(f"Duplicate image_id in add batch: {image_id}")

            if image_id in self.known_image_ids:
                raise ValueError(f"image_id already exists in the index: {image_id}")

            batch_image_ids.add(image_id)

    def _get_metadata(self, image_id: int) -> ImageMetadata:
        """Load metadata for an image, using a small in-memory cache."""
        cached_metadata = self.metadata_cache.get(image_id)
        if cached_metadata is not None:
            return cached_metadata

        metadata = ImageMetadata.load_json(self._metadata_path_for_image_id(image_id=image_id))
        self._update_metadata_cache(metadata=metadata)
        return metadata

    def _update_metadata_cache(self, metadata: ImageMetadata) -> None:
        """Maintain a bounded metadata cache."""
        if len(self.metadata_cache) >= self.metadata_cache_size:
            self.metadata_cache.clear()

        self.metadata_cache[metadata.image_id] = metadata

    def _write_metadata(self, metadata: ImageMetadata) -> None:
        """Persist metadata for a single image to a sharded JSON file."""
        metadata_path = self._metadata_path_for_image_id(image_id=metadata.image_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata.to_dict()
        payload.pop("clip_embedding", None)
        with metadata_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)

    def _write_vector_mapping(self) -> None:
        """Persist the vector-id to image-id mapping."""
        payload = {
            "version": 1,
            "count": len(self.vector_id_to_image_id),
            "image_ids": self.vector_id_to_image_id,
        }
        with self._vector_mapping_path().open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def _write_manifest(self) -> None:
        """Persist metadata storage details needed to validate future loads."""
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")

        payload = {
            "version": 1,
            "embedding_dim": self.index.d,
            "count": len(self.known_image_ids),
            "layout": "sharded-per-image",
            "mapping": "vector_id -> image_id -> metadata",
            "metadata_root": METADATA_DIRNAME,
            "file_template": "metadata/{image_id // 1000:09d}/{image_id}.json",
            "clip_embedding_stored_in_metadata": False,
        }
        with self._metadata_manifest_path().open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def _index_path(self) -> Path:
        """Return the FAISS index file path."""
        return self.index_dir / INDEX_FILENAME

    def _metadata_manifest_path(self) -> Path:
        """Return the metadata manifest file path."""
        return self.index_dir / METADATA_MANIFEST_FILENAME

    def _vector_mapping_path(self) -> Path:
        """Return the vector mapping file path."""
        return self.index_dir / VECTOR_MAPPING_FILENAME

    def _metadata_root(self) -> Path:
        """Return the metadata root directory."""
        return self.index_dir / METADATA_DIRNAME

    def _metadata_path_for_image_id(self, image_id: int) -> Path:
        """Return the sharded metadata path for an image identifier."""
        shard_name = f"{image_id // 1000:09d}"
        return self._metadata_root() / shard_name / f"{image_id}.json"
