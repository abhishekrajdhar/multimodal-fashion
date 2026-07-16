"""Serializable metadata dataclasses for fashion retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Self


@dataclass(frozen=True, slots=True)
class FashionAttributes:
    """Structured fashion attributes associated with an image."""

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
        """Return a JSON-serializable representation of the attributes."""
        return {
            "caption": self.caption,
            "scene": self.scene,
            "style": self.style,
            "upper_garment": self.upper_garment,
            "upper_color": self.upper_color,
            "lower_garment": self.lower_garment,
            "lower_color": self.lower_color,
            "outerwear": self.outerwear,
            "outerwear_color": self.outerwear_color,
            "dress": self.dress,
            "tie": self.tie,
            "hat": self.hat,
            "bag": self.bag,
            "footwear": self.footwear,
            "dominant_colors": list(self.dominant_colors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build attributes from a dictionary payload."""
        return cls(
            caption=_coerce_string(data.get("caption", "")),
            scene=_coerce_string(data.get("scene", "")),
            style=_coerce_string(data.get("style", "")),
            upper_garment=_coerce_string(data.get("upper_garment", "")),
            upper_color=_coerce_string(data.get("upper_color", "")),
            lower_garment=_coerce_string(data.get("lower_garment", "")),
            lower_color=_coerce_string(data.get("lower_color", "")),
            outerwear=_coerce_string(data.get("outerwear", "")),
            outerwear_color=_coerce_string(data.get("outerwear_color", "")),
            dress=_coerce_string(data.get("dress", "")),
            tie=_coerce_string(data.get("tie", "")),
            hat=_coerce_string(data.get("hat", "")),
            bag=_coerce_string(data.get("bag", "")),
            footwear=_coerce_string(data.get("footwear", "")),
            dominant_colors=_coerce_string_list(data.get("dominant_colors", [])),
        )

    def save_json(self, output_path: Path) -> None:
        """Save the attributes as JSON."""
        resolved_output_path = output_path.expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, input_path: Path) -> Self:
        """Load the attributes from a JSON file."""
        resolved_input_path = input_path.expanduser().resolve()
        with resolved_input_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, Mapping):
            raise ValueError("Fashion attributes JSON must contain an object.")

        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Complete metadata record for a fashion image."""

    image_id: int
    image_path: Path
    attributes: FashionAttributes = field(default_factory=FashionAttributes)
    clip_embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a flattened JSON-serializable metadata payload."""
        return {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            **self.attributes.to_dict(),
            "clip_embedding": list(self.clip_embedding),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build image metadata from a dictionary payload."""
        image_id_value = data.get("image_id")
        if image_id_value is None:
            raise ValueError("Image metadata requires 'image_id'.")

        image_path_value = data.get("image_path")
        if image_path_value is None:
            raise ValueError("Image metadata requires 'image_path'.")

        return cls(
            image_id=int(image_id_value),
            image_path=Path(str(image_path_value)),
            attributes=FashionAttributes.from_dict(data),
            clip_embedding=_coerce_float_list(data.get("clip_embedding", [])),
        )

    def save_json(self, output_path: Path) -> None:
        """Save the image metadata as JSON."""
        resolved_output_path = output_path.expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, input_path: Path) -> Self:
        """Load image metadata from a JSON file."""
        resolved_input_path = input_path.expanduser().resolve()
        with resolved_input_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        if not isinstance(payload, Mapping):
            raise ValueError("Image metadata JSON must contain an object.")

        return cls.from_dict(payload)


def _coerce_string(value: Any) -> str:
    """Normalize a value into a string."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _coerce_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []

    if isinstance(value, str):
        cleaned_value = value.strip()
        return [cleaned_value] if cleaned_value else []

    if isinstance(value, list | tuple):
        return [item for item in (_coerce_string(element) for element in value) if item]

    cleaned_value = _coerce_string(value)
    return [cleaned_value] if cleaned_value else []


def _coerce_float_list(value: Any) -> list[float]:
    """Normalize a value into a list of floats."""
    if value is None:
        return []

    if isinstance(value, list | tuple):
        return [float(element) for element in value]

    return [float(value)]
