"""Evaluation runner for multimodal fashion retrieval."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from retriever.retrieve import DEFAULT_CONFIG_PATH, RetrievalService, RetrievedResult
from retriever.visualization import ResultVisualizer
from utils.config import load_config
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K: Final[int] = 5
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/evaluation")
DEFAULT_REPORT_PATH: Final[Path] = Path("outputs/evaluation_report.md")
EVALUATION_QUERIES: Final[tuple[str, ...]] = (
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Evaluation metrics for one query result set."""

    average_similarity_score: float
    precision_at_k: float | None = None
    recall_at_k: float | None = None


@dataclass(frozen=True, slots=True)
class QueryEvaluationResult:
    """Complete evaluation output for a single query."""

    query: str
    top_k: int
    execution_time_seconds: float
    results: list[RetrievedResult]
    metrics: RetrievalMetrics
    visualization_path: Path


class RetrievalEvaluator:
    """Runs a fixed retrieval evaluation suite and writes a report."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        top_k: int = DEFAULT_TOP_K,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.config_path = config_path
        self.top_k = top_k
        self.project_root = config_path.expanduser().resolve().parent.parent
        self.output_dir = self._resolve_path(output_dir)
        self.report_path = self._resolve_path(report_path)
        self.retrieval_service = RetrievalService(config_path=self.config_path)
        self.visualizer = ResultVisualizer()

    def evaluate(self) -> list[QueryEvaluationResult]:
        """Evaluate the retrieval pipeline on the fixed query set."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_service.warmup()
        LOGGER.info("Starting evaluation for %d queries.", len(EVALUATION_QUERIES))

        evaluation_results: list[QueryEvaluationResult] = []
        for query_index, query in enumerate(EVALUATION_QUERIES, start=1):
            LOGGER.info("Evaluating query %d/%d: %s", query_index, len(EVALUATION_QUERIES), query)
            start_time = time.perf_counter()
            results = self.retrieval_service.retrieve_detailed(query=query, top_k=self.top_k)
            execution_time_seconds = time.perf_counter() - start_time

            metrics = self._compute_metrics(results=results)
            visualization_path = self._save_visualization(
                results=results,
                query_index=query_index,
            )
            evaluation_results.append(
                QueryEvaluationResult(
                    query=query,
                    top_k=self.top_k,
                    execution_time_seconds=execution_time_seconds,
                    results=results,
                    metrics=metrics,
                    visualization_path=visualization_path,
                ),
            )

        self._write_report(evaluation_results=evaluation_results)
        LOGGER.info("Evaluation complete. Report written to '%s'.", self.report_path)
        return evaluation_results

    def _resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a project-relative or absolute path."""
        candidate_path = Path(path_value).expanduser()
        if candidate_path.is_absolute():
            return candidate_path

        return (self.project_root / candidate_path).resolve()

    def _compute_metrics(self, results: list[RetrievedResult]) -> RetrievalMetrics:
        """Compute metrics that do not require ground-truth labels."""
        if not results:
            return RetrievalMetrics(average_similarity_score=0.0)

        average_similarity_score = sum(result.score for result in results) / len(results)
        return RetrievalMetrics(
            average_similarity_score=average_similarity_score,
            precision_at_k=None,
            recall_at_k=None,
        )

    def _save_visualization(
        self,
        results: list[RetrievedResult],
        query_index: int,
    ) -> Path:
        """Save a retrieval visualization image for one query."""
        visualization_filename = f"query_{query_index:02d}_results.png"
        visualization_path = self.output_dir / visualization_filename
        if not results:
            return visualization_path

        return self.visualizer.save_results_grid(results=results, output_path=visualization_path)

    def _write_report(self, evaluation_results: list[QueryEvaluationResult]) -> None:
        """Write a markdown report summarizing the evaluation."""
        report_lines: list[str] = [
            "# Retrieval Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Queries evaluated: {len(evaluation_results)}",
            f"- Top-K per query: {self.top_k}",
            (
                "- Precision@K: Not computed because the Fashionpedia test split does not provide "
                "ground-truth relevance labels."
            ),
            (
                "- Recall@K: Not computed because the Fashionpedia test split does not provide "
                "ground-truth relevance labels."
            ),
            (
                "- Future metric hook: `RetrievalMetrics` already includes nullable "
                "`precision_at_k` and `recall_at_k` fields so labeled evaluation can be added later."
            ),
            "",
            "## Aggregate Results",
            "",
        ]

        if evaluation_results:
            overall_average_similarity = (
                sum(result.metrics.average_similarity_score for result in evaluation_results)
                / len(evaluation_results)
            )
            overall_average_time = (
                sum(result.execution_time_seconds for result in evaluation_results)
                / len(evaluation_results)
            )
            report_lines.extend(
                [
                    f"- Mean average similarity score: {overall_average_similarity:.4f}",
                    f"- Mean execution time: {overall_average_time:.2f}s",
                    "",
                ],
            )

        for query_index, evaluation_result in enumerate(evaluation_results, start=1):
            visualization_relative_path = self._relative_to_project_root(evaluation_result.visualization_path)
            report_lines.extend(
                [
                    f"## Query {query_index}",
                    "",
                    f"**Query:** `{evaluation_result.query}`",
                    "",
                    f"- Execution time: {evaluation_result.execution_time_seconds:.2f}s",
                    f"- Average similarity score: {evaluation_result.metrics.average_similarity_score:.4f}",
                    (
                        f"- Retrieval visualization: "
                        f"[{visualization_relative_path}]({visualization_relative_path})"
                    ),
                    "",
                    "### Top-5 Results",
                    "",
                    "| Rank | Image ID | Score | Caption | Matched attributes | Matched scene |",
                    "| --- | --- | --- | --- | --- | --- |",
                ],
            )

            for result in evaluation_result.results:
                matched_attributes = ", ".join(result.matched_attributes) if result.matched_attributes else "None"
                matched_scene = result.matched_scene or "None"
                caption = self._escape_markdown_table_cell(result.caption or "None")
                report_lines.append(
                    f"| {result.rank} | {result.image_id} | {result.score:.4f} | "
                    f"{caption} | {matched_attributes} | {matched_scene} |",
                )

            if not evaluation_result.results:
                report_lines.append("| - | - | - | No results | - | - |")

            report_lines.append("")

        self.report_path.write_text("\n".join(report_lines), encoding="utf-8")

    def _relative_to_project_root(self, path: Path) -> str:
        """Return a project-relative path string for markdown links."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _escape_markdown_table_cell(self, value: str) -> str:
        """Escape markdown table separators in cell content."""
        return value.replace("\n", " ").replace("|", "\\|")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the multimodal fashion retrieval system.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of final results to evaluate per query.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-query retrieval visualizations.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the markdown report file.",
    )
    return parser


def main() -> None:
    """Run the evaluation suite."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    setup_logging(config.get("logging", {}))

    evaluator = RetrievalEvaluator(
        config_path=args.config,
        top_k=args.top_k,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
