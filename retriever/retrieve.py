"""End-to-end retrieval CLI for multimodal fashion search."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metadata import FashionAttributes, ImageMetadata
from retriever.reranker import RerankedCandidate, Reranker
from retriever.search import SearchEngine

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
DEFAULT_FIRST_STAGE_TOP_K = 100


@dataclass(frozen=True, slots=True)
class RetrievedResult:
    """Represents a final retrieval result returned to the user."""

    rank: int
    image_id: int
    score: float
    image_path: Path
    caption: str
    matched_attributes: list[str]
    matched_scene: str | None
    metadata: ImageMetadata

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the result."""
        return {
            "rank": self.rank,
            "image_id": self.image_id,
            "score": self.score,
            "image_path": str(self.image_path),
            "caption": self.caption,
            "matched_attributes": list(self.matched_attributes),
            "matched_scene": self.matched_scene,
            "metadata": self.metadata.to_dict(),
        }


class RetrievalService:
    """Coordinates query parsing, search, and reranking."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        first_stage_top_k: int = DEFAULT_FIRST_STAGE_TOP_K,
        search_engine: SearchEngine | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if first_stage_top_k <= 0:
            raise ValueError("first_stage_top_k must be greater than 0.")

        self.config_path = config_path
        self.first_stage_top_k = first_stage_top_k
        self.search_engine = search_engine or SearchEngine(config_path=self.config_path)
        self.reranker = reranker or Reranker()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        """Run search plus hybrid reranking and return final ranked results."""
        return self.retrieve_detailed(query=query, top_k=top_k)

    def retrieve_detailed(self, query: str, top_k: int) -> list[RetrievedResult]:
        """Run the full retrieval pipeline and return rich result objects."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        search_context = self.search_engine.search_with_context(
            query=query,
            top_k=self.first_stage_top_k,
        )
        reranked_candidates = self.reranker.rerank(
            parsed_query=search_context.parsed_query,
            candidates=[candidate.to_dict() for candidate in search_context.candidates],
            top_k=top_k,
            query_embedding=search_context.query_embedding,
        )

        results: list[RetrievedResult] = []
        for rank, candidate in enumerate(reranked_candidates, start=1):
            results.append(
                RetrievedResult(
                    rank=rank,
                    image_id=candidate.image_id,
                    score=candidate.scores.final_score,
                    image_path=candidate.metadata.image_path,
                    caption=candidate.metadata.attributes.caption,
                    matched_attributes=self._extract_matched_attributes(
                        parsed_query=search_context.parsed_query,
                        attributes=candidate.metadata.attributes,
                    ),
                    matched_scene=self._extract_matched_scene(
                        parsed_query=search_context.parsed_query,
                        candidate=candidate,
                    ),
                    metadata=candidate.metadata,
                ),
            )

        LOGGER.info(
            "Completed retrieval for query '%s' with %d final results.",
            query,
            len(results),
        )
        return results

    def warmup(self) -> None:
        """Load the retrieval index and metadata cache dependencies once at startup."""
        self.search_engine.warmup()

    def _extract_matched_attributes(
        self,
        parsed_query: dict[str, Any],
        attributes: FashionAttributes,
    ) -> list[str]:
        """List the requested attributes that match the candidate metadata."""
        field_order = (
            "upper_garment",
            "upper_color",
            "lower_garment",
            "lower_color",
            "outerwear",
            "outerwear_color",
            "dress",
            "tie",
            "hat",
            "bag",
            "footwear",
            "style",
        )
        matched_attributes: list[str] = []
        for field_name in field_order:
            requested_value = self._normalize_optional_string(parsed_query.get(field_name))
            candidate_value = self._normalize_optional_string(getattr(attributes, field_name))
            if requested_value is None or candidate_value is None:
                continue

            if self._values_match(requested_value, candidate_value):
                matched_attributes.append(f"{field_name}={candidate_value}")

        return matched_attributes

    def _extract_matched_scene(
        self,
        parsed_query: dict[str, Any],
        candidate: RerankedCandidate,
    ) -> str | None:
        """Return the matched scene label if the candidate satisfies the scene request."""
        requested_scene = self._normalize_optional_string(parsed_query.get("scene"))
        candidate_scene = self._normalize_optional_string(candidate.metadata.attributes.scene)
        if requested_scene is None or candidate_scene is None:
            return None

        if self._values_match(requested_scene, candidate_scene):
            return candidate_scene

        return None

    def _normalize_optional_string(self, value: Any) -> str | None:
        """Normalize nullable string-like values."""
        if value is None:
            return None

        normalized_value = str(value).strip().lower()
        if not normalized_value or normalized_value == "null":
            return None

        return normalized_value

    def _values_match(self, requested_value: str, candidate_value: str) -> bool:
        """Return whether two structured values should count as a match."""
        return (
            requested_value == candidate_value
            or requested_value in candidate_value
            or candidate_value in requested_value
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for retrieval."""
    parser = argparse.ArgumentParser(description="Run multimodal fashion retrieval.")
    parser.add_argument("--query", required=True, help="Natural-language fashion query.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of final results to return.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--first_stage_k",
        type=int,
        default=DEFAULT_FIRST_STAGE_TOP_K,
        help="Number of FAISS candidates to retrieve before reranking.",
    )
    return parser


def _print_results(results: list[RetrievedResult], elapsed_seconds: float) -> None:
    """Print retrieval results in a readable CLI format."""
    if not results:
        print("No results found.")
        print(f"Execution time: {elapsed_seconds:.2f}s")
        return

    for result in results:
        matched_attributes = ", ".join(result.matched_attributes) if result.matched_attributes else "None"
        matched_scene = result.matched_scene or "None"
        caption = result.caption or "None"

        print(f"Rank: {result.rank}")
        print(f"Image ID: {result.image_id}")
        print(f"Score: {result.score:.4f}")
        print(f"Caption: {caption}")
        print(f"Matched attributes: {matched_attributes}")
        print(f"Matched scene: {matched_scene}")
        print()

    print(f"Execution time: {elapsed_seconds:.2f}s")


def main() -> None:
    """Run the retrieval CLI."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    start_time = time.perf_counter()
    retrieval_service = RetrievalService(
        config_path=args.config,
        first_stage_top_k=args.first_stage_k,
    )
    results = retrieval_service.retrieve(
        query=args.query,
        top_k=args.top_k,
    )
    elapsed_seconds = time.perf_counter() - start_time
    _print_results(results, elapsed_seconds=elapsed_seconds)


if __name__ == "__main__":
    main()
