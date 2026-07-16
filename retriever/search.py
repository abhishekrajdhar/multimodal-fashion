"""First-stage semantic search for multimodal fashion retrieval."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping

# FAISS and PyTorch can load separate OpenMP runtimes on macOS; allow coexistence.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from metadata import ImageMetadata
from retriever.query_parser import QueryParser
from retriever.text_encoder import TextEncoder
from utils.config import load_config
from utils.logger import setup_logging

if TYPE_CHECKING:
    from vector_db.faiss_manager import FaissManager, SearchResult

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/config.yaml")
DEFAULT_TOP_K: Final[int] = 100


@dataclass(frozen=True, slots=True)
class FirstStageCandidate:
    """Represents a first-stage nearest-neighbor retrieval result."""

    image_id: int
    similarity_score: float
    metadata: ImageMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate payload."""
        return {
            "image_id": self.image_id,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Holds intermediate search artifacts for downstream reranking."""

    query_text: str
    parsed_query: dict[str, Any]
    query_embedding: np.ndarray
    candidates: list[FirstStageCandidate]


class SearchEngine:
    """Performs first-stage retrieval over the vector index."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        query_parser: QueryParser | None = None,
        text_encoder: TextEncoder | None = None,
        faiss_manager: FaissManager | None = None,
    ) -> None:
        self.config_path = config_path
        self.project_root = config_path.expanduser().resolve().parent.parent
        self.config = load_config(config_path=self.config_path)
        setup_logging(self.config.get("logging", {}))

        self.paths_config = self.config.get("paths", {})
        self.indexing_config = self.config.get("indexing", {})
        self.models_config = self.config.get("models", {})

        self.output_dir = self._resolve_path(
            self.paths_config.get("faiss_index_dir")
            or self.paths_config.get("output_dir")
            or "outputs",
        )
        self.query_parser = query_parser or QueryParser(config_path=self.config_path)
        self.text_encoder = text_encoder or self._create_text_encoder()
        self.text_encoder.load_model()
        self.faiss_manager = faiss_manager
        self._index_loaded = False

    def search(
        self,
        query: str | Mapping[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[FirstStageCandidate]:
        """Parse a query, embed it, run FAISS search, and return first-stage matches."""
        return self.search_with_context(query=query, top_k=top_k).candidates

    def search_with_context(
        self,
        query: str | Mapping[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> SearchContext:
        """Run first-stage retrieval and return both results and intermediate artifacts."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_text = self._extract_query_text(query)
        parsed_query = self.query_parser.parse(query_text)
        embedding_text = self._build_embedding_text(
            original_query=query_text,
            parsed_query=parsed_query,
        )
        query_embedding = self.text_encoder.encode_text(embedding_text)

        self._ensure_index_loaded()
        search_batches = self.faiss_manager.search(query_embeddings=query_embedding, top_k=top_k)
        if not search_batches:
            return SearchContext(
                query_text=query_text,
                parsed_query=parsed_query,
                query_embedding=query_embedding,
                candidates=[],
            )

        first_stage_candidates = [
            self._to_first_stage_candidate(search_result=search_result)
            for search_result in search_batches[0]
        ]
        LOGGER.info(
            "Retrieved %d first-stage candidates for query '%s'.",
            len(first_stage_candidates),
            query_text,
        )
        return SearchContext(
            query_text=query_text,
            parsed_query=parsed_query,
            query_embedding=query_embedding,
            candidates=first_stage_candidates,
        )

    def _resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a project-relative or absolute path."""
        candidate_path = Path(path_value).expanduser()
        if candidate_path.is_absolute():
            return candidate_path

        return (self.project_root / candidate_path).resolve()

    def _create_text_encoder(self) -> TextEncoder:
        """Create the OpenCLIP text encoder from configuration."""
        image_encoder_config = self.models_config.get("image_encoder", {})
        return TextEncoder(
            model_name=str(image_encoder_config.get("name", "ViT-H-14")),
            pretrained=str(image_encoder_config.get("checkpoint", "laion2b_s32b_b79k")),
            batch_size=int(self.indexing_config.get("batch_size", 32)),
        )

    def _get_faiss_manager(self) -> "FaissManager":
        """Create the FAISS manager lazily after the text encoder is loaded."""
        if self.faiss_manager is not None:
            return self.faiss_manager

        from vector_db.faiss_manager import FaissManager

        self.faiss_manager = FaissManager(
            index_dir=self.output_dir,
            embedding_dim=int(self.indexing_config.get("embedding_dim", 1024)),
        )
        return self.faiss_manager

    def _ensure_index_loaded(self) -> None:
        """Load the FAISS index lazily on first use."""
        if self._index_loaded:
            return

        self._get_faiss_manager().load()
        self._index_loaded = True

    def warmup(self) -> None:
        """Load the FAISS index once so subsequent queries are fast."""
        self._ensure_index_loaded()

    def _extract_query_text(self, query: str | Mapping[str, Any]) -> str:
        """Normalize supported search inputs into raw text."""
        if isinstance(query, str):
            query_text = query.strip()
        elif "text" in query:
            query_text = str(query["text"]).strip()
        elif "query" in query:
            query_text = str(query["query"]).strip()
        else:
            raise ValueError("Query input must be a string or a mapping with 'text' or 'query'.")

        if not query_text:
            raise ValueError("Query text must be non-empty.")

        return query_text

    def _build_embedding_text(
        self,
        original_query: str,
        parsed_query: Mapping[str, Any],
    ) -> str:
        """Create a normalized text query for OpenCLIP encoding after parsing."""
        ordered_fields = (
            "scene",
            "style",
            "upper_color",
            "upper_garment",
            "lower_color",
            "lower_garment",
            "outerwear_color",
            "outerwear",
            "dress",
            "tie",
            "hat",
            "bag",
            "footwear",
        )
        structured_parts = [
            normalized_value
            for field_name in ordered_fields
            if (normalized_value := self._normalize_optional_string(parsed_query.get(field_name))) is not None
        ]

        keyword_values = parsed_query.get("keywords", [])
        if isinstance(keyword_values, list):
            structured_parts.extend(
                keyword.strip().lower()
                for keyword in keyword_values
                if isinstance(keyword, str) and keyword.strip()
            )

        if structured_parts:
            return " ".join(structured_parts)

        return original_query

    def _normalize_optional_string(self, value: Any) -> str | None:
        """Normalize nullable string-like values."""
        if value is None:
            return None

        normalized_value = str(value).strip().lower()
        if not normalized_value or normalized_value == "null":
            return None

        return normalized_value

    def _to_first_stage_candidate(self, search_result: "SearchResult") -> FirstStageCandidate:
        """Convert a FAISS search hit into the public first-stage candidate shape."""
        return FirstStageCandidate(
            image_id=search_result.image_id,
            similarity_score=search_result.score,
            metadata=search_result.metadata,
        )
