"""Metadata generation for fashion images using Florence-2."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Final, Protocol, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Florence2ForConditionalGeneration

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME: Final[str] = "microsoft/Florence-2-large"
DEFAULT_BATCH_SIZE: Final[int] = 4
DEFAULT_MAX_NEW_TOKENS: Final[int] = 256
PROMPT: Final[str] = (
    "Analyze this fashion image.\n\n"
    "Return ONLY valid JSON.\n\n"
    "{\n"
    ' "caption":"",\n'
    ' "scene":"",\n'
    ' "style":"",\n'
    ' "upper_garment":"",\n'
    ' "upper_color":"",\n'
    ' "lower_garment":"",\n'
    ' "lower_color":"",\n'
    ' "outerwear":"",\n'
    ' "outerwear_color":"",\n'
    ' "dress":"",\n'
    ' "tie":"",\n'
    ' "hat":"",\n'
    ' "bag":"",\n'
    ' "footwear":"",\n'
    ' "dominant_colors":[]\n'
    "}\n\n"
    "Do not explain.\n\n"
    "Do not use markdown.\n\n"
    "Return JSON only."
)
EXPECTED_STRING_FIELDS: Final[tuple[str, ...]] = (
    "caption",
    "scene",
    "style",
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
)


class FlorenceProcessorProtocol(Protocol):
    """Protocol for the Florence-2 processor interface used here."""

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> Any: ...

    def batch_decode(
        self,
        sequences: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]: ...


class FlorenceModelProtocol(Protocol):
    """Protocol for the Florence-2 generation model used here."""

    device: torch.device

    def eval(self) -> torch.nn.Module: ...

    def requires_grad_(self, requires_grad: bool) -> torch.nn.Module: ...

    def to(self, device: torch.device) -> torch.nn.Module: ...

    def generate(self, **kwargs: Any) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class FashionMetadata:
    """Structured fashion metadata generated from an image."""

    caption: str = ""
    scene: str = ""
    style: str = ""
    upper_garment: str = ""
    upper_color: str = ""
    lower_garment: str = ""
    lower_color: str = ""
    outerwear: str = ""
    outerwear_color: str = ""
    dress: str = ""
    tie: str = ""
    hat: str = ""
    bag: str = ""
    footwear: str = ""
    dominant_colors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """Generated metadata associated with an image."""

    image_path: Path
    metadata: FashionMetadata
    raw_response: str


class MetadataGenerator:
    """Generates caption, attributes, and scene metadata with Florence-2."""

    _MODEL_CACHE: ClassVar[
        dict[tuple[str, str, str], tuple[FlorenceModelProtocol, FlorenceProcessorProtocol]]
    ] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = self._select_device(device=device)
        self.dtype = self._select_dtype(device=self.device)
        self.model: FlorenceModelProtocol | None = None
        self.processor: FlorenceProcessorProtocol | None = None

    def load_model(self) -> None:
        """Load and cache the Florence-2 model and processor."""
        cache_key = (self.model_name, self.device.type, str(self.dtype))
        if cache_key in self._MODEL_CACHE:
            self.model, self.processor = self._MODEL_CACHE[cache_key]
            LOGGER.debug(
                "Reusing cached Florence-2 model '%s' on device '%s'.",
                self.model_name,
                self.device,
            )
            return

        LOGGER.info(
            "Loading Florence-2 model '%s' on device '%s' with dtype '%s'.",
            self.model_name,
            self.device,
            self.dtype,
        )
        model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = model
        self.processor = processor
        self._MODEL_CACHE[cache_key] = (model, processor)
        LOGGER.info("Florence-2 model loaded successfully.")

    def generate_metadata(self, image_path: Path) -> MetadataRecord:
        """Generate metadata for a single image."""
        records = self.generate_batch([image_path])
        return records[0]

    def generate_batch(self, image_paths: Sequence[Path]) -> list[MetadataRecord]:
        """Generate metadata for a batch of images."""
        self.load_model()

        if self.model is None or self.processor is None:
            raise RuntimeError("Metadata generator failed to initialize.")

        resolved_paths = [self._validate_image_path(image_path=image_path) for image_path in image_paths]
        if not resolved_paths:
            return []

        records: list[MetadataRecord] = []
        total_batches = (len(resolved_paths) + self.batch_size - 1) // self.batch_size
        LOGGER.info(
            "Generating metadata for %d images in %d batches.",
            len(resolved_paths),
            total_batches,
        )

        batch_starts = range(0, len(resolved_paths), self.batch_size)
        for batch_start in tqdm(batch_starts, desc="Generating metadata", unit="batch"):
            batch_paths = resolved_paths[batch_start : batch_start + self.batch_size]
            batch_images = [self._load_image(image_path=path) for path in batch_paths]
            prompts = [PROMPT] * len(batch_images)

            batch_inputs = self.processor(
                text=prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = self._move_inputs_to_device(batch_inputs=batch_inputs)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )

            responses = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            for image_path, response in zip(batch_paths, responses, strict=True):
                metadata = self._safe_parse_metadata(response=response)
                records.append(
                    MetadataRecord(
                        image_path=image_path,
                        metadata=metadata,
                        raw_response=response,
                    ),
                )

        LOGGER.info("Generated metadata records for %d images.", len(records))
        return records

    def _select_device(self, device: str | None) -> torch.device:
        """Resolve the inference device."""
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _select_dtype(self, device: torch.device) -> torch.dtype:
        """Choose an inference dtype based on the target device."""
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16

            return torch.float16

        return torch.float32

    def _validate_image_path(self, image_path: Path) -> Path:
        """Validate that the provided image path exists."""
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

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load an image for Florence-2 inference."""
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        return rgb_image

    def _move_inputs_to_device(self, batch_inputs: Any) -> dict[str, torch.Tensor]:
        """Move processor outputs to the configured device safely."""
        prepared_inputs: dict[str, torch.Tensor] = {}
        for key, value in batch_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue

            if torch.is_floating_point(value):
                prepared_inputs[key] = value.to(self.device, dtype=self.dtype)
            else:
                prepared_inputs[key] = value.to(self.device)

        return prepared_inputs

    def _safe_parse_metadata(self, response: str) -> FashionMetadata:
        """Parse the model response into structured metadata safely."""
        parsed_object = self._parse_json_object(response=response)
        if parsed_object is None:
            LOGGER.warning("Unable to parse Florence-2 output as JSON: %s", response)
            return FashionMetadata()

        return self._coerce_metadata(data=parsed_object)

    def _parse_json_object(self, response: str) -> dict[str, Any] | None:
        """Extract and parse the first JSON object found in the response."""
        cleaned_response = self._strip_code_fences(response=response)
        json_candidate = self._extract_json_candidate(response=cleaned_response)
        if json_candidate is None:
            return None

        parse_attempts: tuple[Callable[[str], Any], ...] = (
            json.loads,
            ast.literal_eval,
        )
        normalized_candidate = re.sub(r",(\s*[}\]])", r"\1", json_candidate)

        for parse_attempt in parse_attempts:
            try:
                parsed = parse_attempt(normalized_candidate)
            except (ValueError, SyntaxError):
                continue

            if isinstance(parsed, dict):
                return parsed

        return None

    def _strip_code_fences(self, response: str) -> str:
        """Remove markdown code fences if the model returns them."""
        cleaned_response = response.strip()
        cleaned_response = cleaned_response.removeprefix("```json").removeprefix("```")
        cleaned_response = cleaned_response.removesuffix("```").strip()
        return cleaned_response

    def _extract_json_candidate(self, response: str) -> str | None:
        """Extract the outermost JSON object candidate from a response."""
        start_index = response.find("{")
        end_index = response.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return None

        return response[start_index : end_index + 1]

    def _coerce_metadata(self, data: dict[str, Any]) -> FashionMetadata:
        """Normalize parsed JSON into the expected metadata schema."""
        normalized_data: dict[str, Any] = {}

        for field_name in EXPECTED_STRING_FIELDS:
            value = data.get(field_name, "")
            normalized_data[field_name] = self._coerce_string(value=value)

        dominant_colors_value = data.get("dominant_colors", [])
        normalized_data["dominant_colors"] = self._coerce_color_list(value=dominant_colors_value)

        return FashionMetadata(**normalized_data)

    def _coerce_string(self, value: Any) -> str:
        """Normalize an arbitrary value into a clean string."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        return str(value).strip()

    def _coerce_color_list(self, value: Any) -> list[str]:
        """Normalize dominant colors into a list of strings."""
        if value is None:
            return []

        if isinstance(value, str):
            cleaned_value = value.strip()
            return [cleaned_value] if cleaned_value else []

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            colors = [self._coerce_string(item) for item in value]
            return [color for color in colors if color]

        cleaned_value = self._coerce_string(value)
        return [cleaned_value] if cleaned_value else []
