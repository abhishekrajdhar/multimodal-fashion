"""Visualization utilities for retrieval results."""

from __future__ import annotations

import logging
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from metadata import ImageMetadata
from retriever.reranker import RerankedCandidate
from retriever.retrieve import RetrievedResult
from vector_db.faiss_manager import SearchResult

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH: Final[Path] = Path("outputs/results.png")
DEFAULT_MAX_COLUMNS: Final[int] = 3
DEFAULT_FIGURE_WIDTH_PER_COLUMN: Final[float] = 5.5
DEFAULT_FIGURE_HEIGHT_PER_ROW: Final[float] = 6.5
TEXT_WRAP_WIDTH: Final[int] = 38


@dataclass(frozen=True, slots=True)
class VisualizationItem:
    """Normalized payload used for rendering a retrieval result tile."""

    rank: int
    image_id: int
    image_path: Path
    similarity_score: float
    scene: str | None
    caption: str
    matched_attributes: list[str]


class ResultVisualizer:
    """Creates a single grid image for Top-K retrieval results."""

    def __init__(
        self,
        output_path: Path = DEFAULT_OUTPUT_PATH,
        max_columns: int = DEFAULT_MAX_COLUMNS,
    ) -> None:
        if max_columns <= 0:
            raise ValueError("max_columns must be greater than 0.")

        self.output_path = output_path
        self.max_columns = max_columns

    def save_results_grid(
        self,
        results: Sequence[VisualizationItem | RerankedCandidate | RetrievedResult | SearchResult | Mapping[str, Any]],
        output_path: Path | None = None,
    ) -> Path:
        """Render retrieval results into a grid and save a single PNG image."""
        normalized_results = self._normalize_results(results)
        if not normalized_results:
            raise ValueError("At least one retrieval result is required for visualization.")

        save_path = (output_path or self.output_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        column_count = min(self.max_columns, len(normalized_results))
        row_count = math.ceil(len(normalized_results) / column_count)

        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(
                column_count * DEFAULT_FIGURE_WIDTH_PER_COLUMN,
                row_count * DEFAULT_FIGURE_HEIGHT_PER_ROW,
            ),
        )
        figure.patch.set_facecolor("#f7f5f0")
        axes_array = np.atleast_1d(axes).ravel()

        for axis, result in zip(axes_array, normalized_results, strict=False):
            self._render_result_tile(axis=axis, result=result)

        for axis in axes_array[len(normalized_results) :]:
            axis.axis("off")

        figure.suptitle("Fashion Retrieval Results", fontsize=18, fontweight="bold", y=0.99)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
        figure.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)

        LOGGER.info("Saved retrieval visualization to '%s'.", save_path)
        return save_path

    def _normalize_results(
        self,
        results: Sequence[VisualizationItem | RerankedCandidate | RetrievedResult | SearchResult | Mapping[str, Any]],
    ) -> list[VisualizationItem]:
        """Normalize supported result payloads into renderable visualization items."""
        normalized_results: list[VisualizationItem] = []
        for index, result in enumerate(results, start=1):
            normalized_results.append(self._normalize_single_result(result=result, default_rank=index))

        return normalized_results

    def _normalize_single_result(
        self,
        result: VisualizationItem | RerankedCandidate | RetrievedResult | SearchResult | Mapping[str, Any],
        default_rank: int,
    ) -> VisualizationItem:
        """Normalize one result object into a visualization item."""
        if isinstance(result, VisualizationItem):
            return result

        if isinstance(result, RerankedCandidate):
            return VisualizationItem(
                rank=default_rank,
                image_id=result.image_id,
                image_path=result.metadata.image_path,
                similarity_score=result.scores.final_score,
                scene=self._normalize_optional_string(result.metadata.attributes.scene),
                caption=result.metadata.attributes.caption,
                matched_attributes=self._extract_non_empty_attributes(result.metadata),
            )

        if isinstance(result, SearchResult):
            return VisualizationItem(
                rank=default_rank,
                image_id=result.image_id,
                image_path=result.metadata.image_path,
                similarity_score=result.score,
                scene=self._normalize_optional_string(result.metadata.attributes.scene),
                caption=result.metadata.attributes.caption,
                matched_attributes=self._extract_non_empty_attributes(result.metadata),
            )

        if isinstance(result, RetrievedResult):
            return VisualizationItem(
                rank=result.rank,
                image_id=result.image_id,
                image_path=result.image_path,
                similarity_score=result.score,
                scene=self._normalize_optional_string(result.matched_scene or result.metadata.attributes.scene),
                caption=result.caption,
                matched_attributes=list(result.matched_attributes),
            )

        metadata_payload = result.get("metadata")
        if metadata_payload is None:
            raise ValueError("Visualization result mappings must include a 'metadata' field.")

        metadata = (
            metadata_payload
            if isinstance(metadata_payload, ImageMetadata)
            else ImageMetadata.from_dict(metadata_payload)
        )
        matched_attributes = result.get("matched_attributes")
        normalized_matched_attributes = (
            [str(item) for item in matched_attributes]
            if isinstance(matched_attributes, Sequence) and not isinstance(matched_attributes, (str, bytes, bytearray))
            else self._extract_non_empty_attributes(metadata)
        )

        return VisualizationItem(
            rank=int(result.get("rank", default_rank)),
            image_id=int(result.get("image_id", metadata.image_id)),
            image_path=metadata.image_path,
            similarity_score=float(
                result.get(
                    "score",
                    result.get("similarity_score", 0.0),
                ),
            ),
            scene=self._normalize_optional_string(
                result.get("matched_scene", metadata.attributes.scene),
            ),
            caption=str(result.get("caption", metadata.attributes.caption)).strip(),
            matched_attributes=normalized_matched_attributes,
        )

    def _render_result_tile(self, axis: Any, result: VisualizationItem) -> None:
        """Render one result image and its annotation block."""
        axis.set_facecolor("#ffffff")
        axis.set_xticks([])
        axis.set_yticks([])

        try:
            image_array = self._load_image(result.image_path)
            axis.imshow(image_array)
        except FileNotFoundError:
            axis.text(
                0.5,
                0.55,
                "Image not found",
                ha="center",
                va="center",
                fontsize=12,
                color="#7a2e2e",
                transform=axis.transAxes,
            )
            axis.set_facecolor("#ece7df")

        header_text = f"Rank #{result.rank} | ID {result.image_id} | Score {result.similarity_score:.4f}"
        caption_text = self._wrap_text(f"Caption: {result.caption or 'None'}")
        scene_text = self._wrap_text(f"Scene: {result.scene or 'None'}")
        matched_attributes = ", ".join(result.matched_attributes) if result.matched_attributes else "None"
        attributes_text = self._wrap_text(f"Matched attributes: {matched_attributes}")
        annotation_text = f"{scene_text}\n{caption_text}\n{attributes_text}"

        axis.text(
            0.02,
            0.98,
            header_text,
            ha="left",
            va="top",
            fontsize=10,
            color="#ffffff",
            fontweight="bold",
            transform=axis.transAxes,
            bbox={"facecolor": "#111827", "alpha": 0.85, "pad": 4, "edgecolor": "none"},
        )
        axis.text(
            0.02,
            0.02,
            annotation_text,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#111827",
            transform=axis.transAxes,
            bbox={"facecolor": "#f9fafb", "alpha": 0.92, "pad": 6, "edgecolor": "#d1d5db"},
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from disk as an RGB array."""
        resolved_image_path = image_path.expanduser().resolve()
        if not resolved_image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {resolved_image_path}")

        with Image.open(resolved_image_path) as image:
            rgb_image = image.convert("RGB")
            return np.asarray(rgb_image)

    def _extract_non_empty_attributes(self, metadata: ImageMetadata) -> list[str]:
        """Extract present fashion attributes from metadata for display."""
        attributes = metadata.attributes
        field_order = (
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
        extracted_attributes: list[str] = []
        for field_name in field_order:
            value = self._normalize_optional_string(getattr(attributes, field_name))
            if value is not None:
                extracted_attributes.append(f"{field_name}={value}")

        return extracted_attributes

    def _normalize_optional_string(self, value: Any) -> str | None:
        """Normalize nullable string-like values."""
        if value is None:
            return None

        normalized_value = str(value).strip()
        if not normalized_value:
            return None

        return normalized_value

    def _wrap_text(self, text: str) -> str:
        """Wrap longer text for compact display in each tile."""
        return "\n".join(textwrap.wrap(text, width=TEXT_WRAP_WIDTH, break_long_words=False))
