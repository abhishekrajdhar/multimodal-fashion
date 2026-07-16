"""Candidate reranking for multimodal fashion retrieval."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

import numpy as np

from metadata import FashionAttributes, ImageMetadata

if TYPE_CHECKING:
    from vector_db.faiss_manager import SearchResult

LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
ACCESSORY_FIELDS: Final[tuple[str, ...]] = ("outerwear", "tie", "hat", "bag", "footwear", "dress")
GARMENT_FIELDS: Final[tuple[str, ...]] = ("upper_garment", "lower_garment")
COLOR_FIELDS: Final[tuple[str, ...]] = ("upper_color", "lower_color", "outerwear_color")


@dataclass(frozen=True, slots=True)
class RerankScores:
    """Stores all component scores for a reranked candidate."""

    clip_similarity: float
    caption_similarity: float
    attribute_score: float
    scene_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """Represents a candidate after score fusion and reranking."""

    vector_id: int | None
    image_id: int
    metadata: ImageMetadata
    scores: RerankScores

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the reranked result."""
        return {
            "vector_id": self.vector_id,
            "image_id": self.image_id,
            "metadata": self.metadata.to_dict(),
            "scores": {
                "clip_similarity": self.scores.clip_similarity,
                "caption_similarity": self.scores.caption_similarity,
                "attribute_score": self.scores.attribute_score,
                "scene_score": self.scores.scene_score,
                "final_score": self.scores.final_score,
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Internal normalized candidate representation for reranking."""

    vector_id: int
    image_id: int
    score: float
    metadata: ImageMetadata


class Reranker:
    """Refines initial retrieval results using richer signals."""

    def rerank(
        self,
        parsed_query: Mapping[str, Any],
        candidates: Sequence["SearchResult" | Mapping[str, Any]],
        top_k: int,
        query_embedding: np.ndarray | Sequence[float] | None = None,
    ) -> list[RerankedCandidate]:
        """Score, sort, and truncate retrieval candidates."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not candidates:
            return []

        normalized_query = self._normalize_query(parsed_query)
        prepared_query_embedding = self._prepare_query_embedding(query_embedding)

        reranked_candidates: list[RerankedCandidate] = []
        for candidate in candidates:
            normalized_candidate = self._normalize_candidate(candidate)
            scores = RerankScores(
                clip_similarity=self._compute_clip_similarity(
                    query_embedding=prepared_query_embedding,
                    candidate=normalized_candidate,
                ),
                caption_similarity=self._compute_caption_similarity(
                    parsed_query=normalized_query,
                    metadata=normalized_candidate.metadata,
                ),
                attribute_score=self._compute_attribute_score(
                    parsed_query=normalized_query,
                    metadata=normalized_candidate.metadata,
                ),
                scene_score=self._compute_scene_score(
                    parsed_query=normalized_query,
                    metadata=normalized_candidate.metadata,
                ),
                final_score=0.0,
            )
            final_score = (
                0.45 * scores.clip_similarity
                + 0.20 * scores.caption_similarity
                + 0.20 * scores.attribute_score
                + 0.15 * scores.scene_score
            )
            reranked_candidates.append(
                RerankedCandidate(
                    vector_id=normalized_candidate.vector_id,
                    image_id=normalized_candidate.image_id,
                    metadata=normalized_candidate.metadata,
                    scores=RerankScores(
                        clip_similarity=scores.clip_similarity,
                        caption_similarity=scores.caption_similarity,
                        attribute_score=scores.attribute_score,
                        scene_score=scores.scene_score,
                        final_score=final_score,
                    ),
                ),
            )

        reranked_candidates.sort(
            key=lambda candidate: (
                candidate.scores.final_score,
                candidate.scores.clip_similarity,
                candidate.image_id,
            ),
            reverse=True,
        )
        LOGGER.info("Reranked %d candidates and returned top %d.", len(candidates), min(top_k, len(candidates)))
        return reranked_candidates[:top_k]

    # CLIP similarity captures the strongest cross-modal semantic match signal.
    # If we have the text query embedding and the candidate's CLIP embedding, we
    # compute a cosine-style similarity via dot product on normalized vectors.
    # When the candidate embedding is unavailable, we fall back to the first-stage
    # retrieval score, which is already an inner-product similarity from FAISS.
    def _compute_clip_similarity(
        self,
        query_embedding: np.ndarray | None,
        candidate: CandidateRecord,
    ) -> float:
        """Compute a normalized CLIP similarity score in [0, 1]."""
        candidate_embedding = np.asarray(candidate.metadata.clip_embedding, dtype=np.float32)
        if query_embedding is not None and candidate_embedding.size > 0:
            if candidate_embedding.ndim != 1:
                candidate_embedding = candidate_embedding.reshape(-1)

            denominator = np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            if denominator <= 0.0:
                return 0.0

            cosine_similarity = float(np.dot(query_embedding, candidate_embedding) / denominator)
            return self._clamp_01((cosine_similarity + 1.0) / 2.0)

        return self._clamp_01((float(candidate.score) + 1.0) / 2.0)

    # Caption similarity checks whether the structured request and the generated
    # image caption describe the same objects, colors, actions, and scene tokens.
    # We compare token sets with Sørensen-Dice overlap so partial matches still
    # receive credit while unrelated captions are pushed down.
    def _compute_caption_similarity(
        self,
        parsed_query: Mapping[str, Any],
        metadata: ImageMetadata,
    ) -> float:
        """Compute token-overlap similarity between the parsed query and caption."""
        query_caption_text = self._build_caption_proxy(parsed_query)
        caption_text = metadata.attributes.caption
        if not query_caption_text or not caption_text:
            return 0.0

        query_tokens = set(self._tokenize(query_caption_text))
        caption_tokens = set(self._tokenize(caption_text))
        if not query_tokens or not caption_tokens:
            return 0.0

        overlap = len(query_tokens & caption_tokens)
        return (2.0 * overlap) / (len(query_tokens) + len(caption_tokens))

    # Attribute scoring rewards agreement on the most retrieval-critical fashion
    # facts: garment types, garment colors, and accessories. Garment type matches
    # matter the most, then colors, then accessories. Missing requested fields do
    # not count against the candidate; only explicitly requested attributes do.
    def _compute_attribute_score(
        self,
        parsed_query: Mapping[str, Any],
        metadata: ImageMetadata,
    ) -> float:
        """Compute a weighted score for garment, color, and accessory matches."""
        attributes = metadata.attributes
        garment_score = self._average_field_match(parsed_query, attributes, GARMENT_FIELDS)
        color_score = self._average_field_match(parsed_query, attributes, COLOR_FIELDS)
        accessory_score = self._average_field_match(parsed_query, attributes, ACCESSORY_FIELDS)

        weighted_sum = (0.45 * garment_score) + (0.35 * color_score) + (0.20 * accessory_score)
        return self._clamp_01(weighted_sum)

    # Scene scoring handles location or environment intent, such as park, beach,
    # or office. Exact matches receive full credit, while substring containment
    # such as "sports court" vs "court" receives partial credit.
    def _compute_scene_score(
        self,
        parsed_query: Mapping[str, Any],
        metadata: ImageMetadata,
    ) -> float:
        """Compute scene agreement between the parsed query and metadata scene."""
        requested_scene = self._normalize_optional_string(parsed_query.get("scene"))
        candidate_scene = self._normalize_optional_string(metadata.attributes.scene)
        if requested_scene is None or candidate_scene is None:
            return 0.0

        if requested_scene == candidate_scene:
            return 1.0

        if requested_scene in candidate_scene or candidate_scene in requested_scene:
            return 0.5

        return 0.0

    def _prepare_query_embedding(
        self,
        query_embedding: np.ndarray | Sequence[float] | None,
    ) -> np.ndarray | None:
        """Normalize the text query embedding if one is available."""
        if query_embedding is None:
            return None

        prepared_embedding = np.asarray(query_embedding, dtype=np.float32)
        if prepared_embedding.ndim != 1:
            prepared_embedding = prepared_embedding.reshape(-1)

        denominator = np.linalg.norm(prepared_embedding)
        if denominator <= 0.0:
            return None

        return prepared_embedding / denominator

    def _normalize_query(self, parsed_query: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the parsed query into the expected reranker schema."""
        normalized_query: dict[str, Any] = {
            "scene": self._normalize_optional_string(parsed_query.get("scene")),
            "style": self._normalize_optional_string(parsed_query.get("style")),
            "upper_garment": self._normalize_optional_string(parsed_query.get("upper_garment")),
            "upper_color": self._normalize_optional_string(parsed_query.get("upper_color")),
            "lower_garment": self._normalize_optional_string(parsed_query.get("lower_garment")),
            "lower_color": self._normalize_optional_string(parsed_query.get("lower_color")),
            "outerwear": self._normalize_optional_string(parsed_query.get("outerwear")),
            "outerwear_color": self._normalize_optional_string(parsed_query.get("outerwear_color")),
            "dress": self._normalize_optional_string(parsed_query.get("dress")),
            "tie": self._normalize_optional_string(parsed_query.get("tie")),
            "hat": self._normalize_optional_string(parsed_query.get("hat")),
            "bag": self._normalize_optional_string(parsed_query.get("bag")),
            "footwear": self._normalize_optional_string(parsed_query.get("footwear")),
            "keywords": self._normalize_keywords(parsed_query.get("keywords", [])),
        }
        return normalized_query

    def _normalize_candidate(
        self,
        candidate: "SearchResult" | Mapping[str, Any] | CandidateRecord,
    ) -> CandidateRecord:
        """Convert supported candidate shapes into a common SearchResult object."""
        if isinstance(candidate, CandidateRecord):
            return candidate

        if not isinstance(candidate, Mapping):
            metadata = getattr(candidate, "metadata", None)
            if not isinstance(metadata, ImageMetadata):
                raise ValueError("Candidate objects must expose an ImageMetadata 'metadata' attribute.")

            return CandidateRecord(
                vector_id=int(getattr(candidate, "vector_id", -1)),
                image_id=int(getattr(candidate, "image_id")),
                score=float(getattr(candidate, "score")),
                metadata=metadata,
            )

        metadata_payload = candidate.get("metadata")
        if metadata_payload is None:
            raise ValueError("Candidate dictionaries must include a 'metadata' field.")

        metadata = (
            metadata_payload
            if isinstance(metadata_payload, ImageMetadata)
            else ImageMetadata.from_dict(metadata_payload)
        )
        return CandidateRecord(
            vector_id=int(candidate["vector_id"]) if candidate.get("vector_id") is not None else -1,
            image_id=int(candidate.get("image_id", metadata.image_id)),
            score=float(
                candidate.get(
                    "score",
                    candidate.get("similarity_score", candidate.get("clip_similarity", 0.0)),
                ),
            ),
            metadata=metadata,
        )

    def _build_caption_proxy(self, parsed_query: Mapping[str, Any]) -> str:
        """Build a compact text description from the parsed structured query."""
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
        parts = [
            value
            for field_name in ordered_fields
            if (value := self._normalize_optional_string(parsed_query.get(field_name))) is not None
        ]
        parts.extend(self._normalize_keywords(parsed_query.get("keywords", [])))
        return " ".join(parts)

    def _average_field_match(
        self,
        parsed_query: Mapping[str, Any],
        attributes: FashionAttributes,
        field_names: Sequence[str],
    ) -> float:
        """Average exact-match score across requested fields in a group."""
        matches = 0.0
        requested_fields = 0
        for field_name in field_names:
            requested_value = self._normalize_optional_string(parsed_query.get(field_name))
            if requested_value is None:
                continue

            requested_fields += 1
            candidate_value = self._normalize_optional_string(getattr(attributes, field_name))
            if candidate_value == requested_value:
                matches += 1.0
            elif candidate_value is not None and (
                requested_value in candidate_value or candidate_value in requested_value
            ):
                matches += 0.5

        if requested_fields == 0:
            return 0.0

        return matches / requested_fields

    def _normalize_optional_string(self, value: Any) -> str | None:
        """Normalize nullable string-like values."""
        if value is None:
            return None

        normalized_value = str(value).strip().lower()
        if not normalized_value or normalized_value == "null":
            return None

        return normalized_value

    def _normalize_keywords(self, value: Any) -> list[str]:
        """Normalize keyword payloads into lowercase unique terms."""
        if value is None:
            return []

        if isinstance(value, str):
            raw_keywords = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_keywords = [str(item) for item in value]
        else:
            raw_keywords = [str(value)]

        normalized_keywords: list[str] = []
        seen_keywords: set[str] = set()
        for raw_keyword in raw_keywords:
            normalized_keyword = raw_keyword.strip().lower()
            if not normalized_keyword or normalized_keyword in seen_keywords:
                continue

            seen_keywords.add(normalized_keyword)
            normalized_keywords.append(normalized_keyword)

        return normalized_keywords

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize free-form text into simple lowercase tokens."""
        return TOKEN_PATTERN.findall(text.lower())

    def _clamp_01(self, value: float) -> float:
        """Clamp a floating-point value into [0, 1]."""
        if math.isnan(value):
            return 0.0

        return max(0.0, min(1.0, value))
