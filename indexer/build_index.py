"""End-to-end index building pipeline for multimodal fashion retrieval."""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import Any, Final

from tqdm import tqdm

from indexer.dataset import FashionDataset, ImageRecord
from indexer.image_encoder import ImageEncoder
from indexer.metadata_generator import MetadataGenerator, MetadataRecord as GeneratedMetadataRecord
from metadata import FashionAttributes, ImageMetadata
from utils.config import load_config
from utils.logger import setup_logging
from vector_db.faiss_manager import FaissManager

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/config.yaml")


class IndexBuilder:
    """Coordinates dataset preparation and vector index construction."""

    def __init__(self, config_path: Path) -> None:
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
        self.dataset_dir = self._resolve_dataset_dir()
        self.batch_size = int(self.indexing_config.get("batch_size", 32))
        self.num_workers = int(
            self.indexing_config.get(
                "num_workers",
                max(1, multiprocessing.cpu_count() - 1),
            ),
        )
        self.embedding_dim = int(self.indexing_config.get("embedding_dim", 1024))

        self._validate_runtime_config()

    def build(self) -> None:
        """Build the multimodal retrieval index."""
        LOGGER.info("Starting index build for dataset '%s'.", self.dataset_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        dataset = FashionDataset(
            data_dir=self.dataset_dir,
            num_workers=self.num_workers,
        )
        all_records = dataset.load_records()

        faiss_manager = self._load_or_initialize_faiss_manager()
        processed_image_ids = faiss_manager.known_image_ids
        pending_records = [record for record in all_records if record.image_id not in processed_image_ids]

        if not pending_records:
            LOGGER.info(
                "All %d images are already indexed. Nothing to do.",
                len(all_records),
            )
            return

        LOGGER.info(
            "Loaded %d images. Skipping %d already processed images. Remaining: %d.",
            len(all_records),
            len(processed_image_ids),
            len(pending_records),
        )

        image_encoder = self._create_image_encoder()
        metadata_generator = self._create_metadata_generator()

        batch_starts = range(0, len(pending_records), self.batch_size)
        for batch_start in tqdm(
            batch_starts,
            total=(len(pending_records) + self.batch_size - 1) // self.batch_size,
            desc="Indexing batches",
            unit="batch",
        ):
            batch_records = pending_records[batch_start : batch_start + self.batch_size]
            self._process_batch(
                batch_records=batch_records,
                image_encoder=image_encoder,
                metadata_generator=metadata_generator,
                faiss_manager=faiss_manager,
            )

        LOGGER.info(
            "Index build complete. Indexed %d total images.",
            len(all_records),
        )

    def _resolve_path(self, path_value: str | Path) -> Path:
        """Resolve a project-relative or absolute path."""
        candidate_path = Path(path_value).expanduser()
        if candidate_path.is_absolute():
            return candidate_path

        return (self.project_root / candidate_path).resolve()

    def _resolve_dataset_dir(self) -> Path:
        """Resolve the dataset directory from configuration or project layout."""
        configured_dataset_dir = self.paths_config.get("dataset_dir")
        if configured_dataset_dir:
            return self._resolve_path(configured_dataset_dir)

        project_test_dir = (self.project_root / "test").resolve()
        if project_test_dir.exists():
            return project_test_dir

        raw_data_dir = self.paths_config.get("raw_data_dir", "data/raw")
        return self._resolve_path(raw_data_dir)

    def _validate_runtime_config(self) -> None:
        """Validate runtime configuration values."""
        if self.batch_size <= 0:
            raise ValueError("indexing.batch_size must be greater than 0.")

        if self.num_workers <= 0:
            raise ValueError("indexing.num_workers must be greater than 0.")

        if self.embedding_dim <= 0:
            raise ValueError("indexing.embedding_dim must be greater than 0.")

    def _load_or_initialize_faiss_manager(self) -> FaissManager:
        """Load an existing index if available, otherwise start a fresh one."""
        faiss_index_path = self.output_dir / "faiss.index"
        mapping_path = self.output_dir / "mapping.json"
        metadata_manifest_path = self.output_dir / "metadata.json"
        existing_artifact_paths = [
            artifact_path
            for artifact_path in (faiss_index_path, mapping_path, metadata_manifest_path)
            if artifact_path.exists()
        ]

        faiss_manager = FaissManager(
            index_dir=self.output_dir,
            embedding_dim=self.embedding_dim,
        )

        if len(existing_artifact_paths) == 3:
            LOGGER.info("Found existing index artifacts. Resuming from '%s'.", self.output_dir)
            faiss_manager.load()
            return faiss_manager

        if existing_artifact_paths:
            existing_names = ", ".join(path.name for path in existing_artifact_paths)
            raise RuntimeError(
                "Found a partial index state that cannot be resumed safely. "
                f"Existing artifacts: {existing_names}. "
                "Remove the incomplete outputs or restore the missing files before rerunning.",
            )

        LOGGER.info("No complete existing index found. Starting a fresh build.")
        return faiss_manager

    def _create_image_encoder(self) -> ImageEncoder:
        """Create the CLIP image encoder from configuration."""
        image_encoder_config = self.models_config.get("image_encoder", {})
        return ImageEncoder(
            model_name=str(image_encoder_config.get("name", "ViT-H-14")),
            pretrained=str(image_encoder_config.get("checkpoint", "laion2b_s32b_b79k")),
            batch_size=self.batch_size,
        )

    def _create_metadata_generator(self) -> MetadataGenerator:
        """Create the Florence-2 metadata generator."""
        metadata_generator_config = self.models_config.get("metadata_generator", {})
        return MetadataGenerator(
            model_name=str(
                metadata_generator_config.get("name", "microsoft/Florence-2-large"),
            ),
            batch_size=self.batch_size,
        )

    def _process_batch(
        self,
        batch_records: list[ImageRecord],
        image_encoder: ImageEncoder,
        metadata_generator: MetadataGenerator,
        faiss_manager: FaissManager,
    ) -> None:
        """Process a single batch end to end and persist progress."""
        batch_paths = [record.image_path for record in batch_records]
        LOGGER.info(
            "Processing batch of %d images starting at image_id=%d.",
            len(batch_records),
            batch_records[0].image_id,
        )

        embeddings = image_encoder.encode_batch(batch_paths)
        generated_metadata_records = metadata_generator.generate_batch(batch_paths)
        image_metadata_batch = self._merge_batch_outputs(
            image_records=batch_records,
            generated_metadata_records=generated_metadata_records,
            embeddings=embeddings,
        )

        faiss_manager.add(embeddings=embeddings, metadata_items=image_metadata_batch)
        faiss_manager.save()

    def _merge_batch_outputs(
        self,
        image_records: list[ImageRecord],
        generated_metadata_records: list[GeneratedMetadataRecord],
        embeddings: Any,
    ) -> list[ImageMetadata]:
        """Combine image records, Florence metadata, and CLIP embeddings."""
        if len(image_records) != len(generated_metadata_records):
            raise ValueError("Metadata batch size does not match image batch size.")

        if len(image_records) != len(embeddings):
            raise ValueError("Embedding batch size does not match image batch size.")

        metadata_records_by_path = {
            record.image_path.resolve(): record for record in generated_metadata_records
        }

        merged_metadata: list[ImageMetadata] = []
        for image_record, embedding in zip(image_records, embeddings, strict=True):
            generated_metadata_record = metadata_records_by_path.get(image_record.image_path.resolve())
            if generated_metadata_record is None:
                raise ValueError(f"Missing metadata record for image: {image_record.image_path}")

            merged_metadata.append(
                ImageMetadata(
                    image_id=image_record.image_id,
                    image_path=image_record.image_path,
                    attributes=self._build_fashion_attributes(
                        generated_metadata=generated_metadata_record,
                    ),
                    clip_embedding=[float(value) for value in embedding.tolist()],
                ),
            )

        return merged_metadata

    def _build_fashion_attributes(
        self,
        generated_metadata: GeneratedMetadataRecord,
    ) -> FashionAttributes:
        """Convert Florence metadata output into the shared metadata schema."""
        metadata = generated_metadata.metadata
        return FashionAttributes(
            caption=metadata.caption,
            scene=metadata.scene,
            style=metadata.style,
            upper_garment=metadata.upper_garment,
            upper_color=metadata.upper_color,
            lower_garment=metadata.lower_garment,
            lower_color=metadata.lower_color,
            outerwear=metadata.outerwear,
            outerwear_color=metadata.outerwear_color,
            dress=metadata.dress,
            tie=metadata.tie,
            hat=metadata.hat,
            bag=metadata.bag,
            footwear=metadata.footwear,
            dominant_colors=list(metadata.dominant_colors),
        )


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run the index build pipeline."""
    builder = IndexBuilder(config_path=config_path)
    builder.build()


if __name__ == "__main__":
    main()
