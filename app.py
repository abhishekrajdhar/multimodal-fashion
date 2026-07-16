"""Gradio application for multimodal fashion retrieval."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from retriever.retrieve import DEFAULT_CONFIG_PATH, RetrievalService, RetrievedResult
from utils.config import load_config
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)
APP_TITLE = "Fashion Multimodal Retrieval"
DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 20


def build_demo(config_path: Path = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    """Create the Gradio demo and warm up retrieval resources once."""
    config = load_config(config_path=config_path)
    setup_logging(config.get("logging", {}))

    retrieval_service = RetrievalService(config_path=config_path)
    retrieval_service.warmup()
    LOGGER.info("Retrieval service warmed up successfully.")

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")

        with gr.Row():
            query_input = gr.Textbox(
                label="Query",
                placeholder="Professional business attire inside a modern office.",
                lines=2,
                scale=4,
            )
            top_k_input = gr.Slider(
                label="Top-K",
                minimum=MIN_TOP_K,
                maximum=MAX_TOP_K,
                step=1,
                value=DEFAULT_TOP_K,
                scale=1,
            )

        search_button = gr.Button("Search", variant="primary")
        execution_time_output = gr.Textbox(label="Execution time", interactive=False)

        with gr.Row():
            gallery_output = gr.Gallery(
                label="Gallery",
                columns=3,
                rows=2,
                object_fit="cover",
                height="auto",
                preview=True,
                allow_preview=True,
                scale=3,
            )
            metadata_output = gr.JSON(label="Metadata panel", scale=2)

        results_state = gr.State(value=[])

        search_button.click(
            fn=lambda query, top_k: _run_search(
                retrieval_service=retrieval_service,
                query=query,
                top_k=int(top_k),
            ),
            inputs=[query_input, top_k_input],
            outputs=[gallery_output, metadata_output, execution_time_output, results_state],
        )
        gallery_output.select(
            fn=_show_selected_metadata,
            inputs=[results_state],
            outputs=[metadata_output],
        )

    return demo


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the multimodal fashion retrieval app.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    """Launch the Gradio application."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    app_config = config.get("app", {})
    host = str(app_config.get("host", "0.0.0.0"))
    port = int(app_config.get("port", 8000))

    demo = build_demo(config_path=args.config)
    demo.launch(server_name=host, server_port=port)


def _run_search(
    retrieval_service: RetrievalService,
    query: str,
    top_k: int,
) -> tuple[list[tuple[str, str]], dict[str, Any], str, list[dict[str, Any]]]:
    """Execute retrieval and format outputs for the Gradio interface."""
    query_text = query.strip()
    if not query_text:
        return ([], {"message": "Please enter a query."}, "Execution time: 0.00s", [])

    start_time = time.perf_counter()
    results = retrieval_service.retrieve_detailed(query=query_text, top_k=top_k)
    elapsed_seconds = time.perf_counter() - start_time

    gallery_items = [_to_gallery_item(result) for result in results]
    serialized_results = [result.to_dict() for result in results]
    metadata_panel = (
        _build_metadata_panel(results[0]) if results else {"message": "No results found."}
    )
    execution_time_text = f"Execution time: {elapsed_seconds:.2f}s"
    return (gallery_items, metadata_panel, execution_time_text, serialized_results)


def _to_gallery_item(result: RetrievedResult) -> tuple[str, str]:
    """Convert a retrieval result into a gallery tile."""
    label = f"Rank #{result.rank} | Score {result.score:.4f} | ID {result.image_id}"
    return (str(result.image_path), label)


def _show_selected_metadata(
    results: list[dict[str, Any]],
    evt: gr.SelectData,
) -> dict[str, Any]:
    """Update the metadata panel when a gallery item is selected."""
    if not results:
        return {"message": "No results found."}

    selected_index = int(evt.index) if evt.index is not None else 0
    if selected_index < 0 or selected_index >= len(results):
        selected_index = 0

    return _build_metadata_panel_from_payload(results[selected_index])


def _build_metadata_panel(result: RetrievedResult) -> dict[str, Any]:
    """Build a metadata view for one retrieval result."""
    return {
        "rank": result.rank,
        "image_id": result.image_id,
        "score": result.score,
        "caption": result.caption,
        "matched_attributes": result.matched_attributes,
        "matched_scene": result.matched_scene,
        "image_path": str(result.image_path),
        "metadata": result.metadata.to_dict(),
    }


def _build_metadata_panel_from_payload(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a metadata view from serialized result state."""
    return {
        "rank": result_payload.get("rank"),
        "image_id": result_payload.get("image_id"),
        "score": result_payload.get("score"),
        "caption": result_payload.get("caption"),
        "matched_attributes": result_payload.get("matched_attributes", []),
        "matched_scene": result_payload.get("matched_scene"),
        "image_path": result_payload.get("image_path"),
        "metadata": result_payload.get("metadata", {}),
    }


if __name__ == "__main__":
    main()
