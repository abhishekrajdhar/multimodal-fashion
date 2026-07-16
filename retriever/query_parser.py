"""Natural-language fashion query parsing."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

LOGGER = logging.getLogger(__name__)
DEFAULT_LLM_MODEL: Final[str] = "gpt-4.1-mini"
SCHEMA_KEYS: Final[tuple[str, ...]] = (
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
    "keywords",
)
DEFAULT_SCHEMA: Final[dict[str, Any]] = {
    "scene": None,
    "style": None,
    "upper_garment": None,
    "upper_color": None,
    "lower_garment": None,
    "lower_color": None,
    "outerwear": None,
    "outerwear_color": None,
    "dress": None,
    "tie": None,
    "hat": None,
    "bag": None,
    "footwear": None,
    "keywords": [],
}
COLORS: Final[set[str]] = {
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "brown",
    "gray",
    "grey",
    "beige",
    "tan",
    "gold",
    "silver",
    "navy",
    "maroon",
    "olive",
    "teal",
    "cyan",
    "cream",
}
SCENES: Final[dict[str, str]] = {
    "park": "park",
    "beach": "beach",
    "street": "street",
    "sidewalk": "street",
    "city": "city",
    "office": "office",
    "workplace": "office",
    "home": "home",
    "living room": "home",
    "bedroom": "home",
    "studio": "studio",
    "runway": "runway",
    "catwalk": "runway",
    "restaurant": "restaurant",
    "cafe": "cafe",
    "coffee shop": "cafe",
    "gym": "gym",
    "court": "sports court",
    "stadium": "stadium",
    "garden": "garden",
    "forest": "outdoor",
    "outdoors": "outdoor",
    "mountain": "outdoor",
    "snow": "snow",
    "party": "party",
}
STYLES: Final[dict[str, str]] = {
    "casual": "casual",
    "formal": "formal",
    "business": "business",
    "streetwear": "streetwear",
    "sporty": "sporty",
    "athleisure": "athleisure",
    "vintage": "vintage",
    "bohemian": "bohemian",
    "minimalist": "minimalist",
    "chic": "chic",
    "elegant": "elegant",
    "punk": "punk",
    "grunge": "grunge",
}
UPPER_GARMENTS: Final[dict[str, str]] = {
    "shirt": "shirt",
    "t-shirt": "t-shirt",
    "tee": "t-shirt",
    "blouse": "blouse",
    "top": "top",
    "sweater": "sweater",
    "hoodie": "hoodie",
    "tank top": "tank top",
    "polo": "polo",
    "jersey": "jersey",
}
LOWER_GARMENTS: Final[dict[str, str]] = {
    "jeans": "jeans",
    "pants": "pants",
    "trousers": "trousers",
    "shorts": "shorts",
    "skirt": "skirt",
    "leggings": "leggings",
}
OUTERWEAR_ITEMS: Final[dict[str, str]] = {
    "jacket": "jacket",
    "coat": "coat",
    "blazer": "blazer",
    "cardigan": "cardigan",
    "parka": "parka",
    "windbreaker": "windbreaker",
}
DRESSES: Final[dict[str, str]] = {
    "dress": "dress",
    "gown": "gown",
    "sundress": "sundress",
    "maxi dress": "maxi dress",
}
TIES: Final[dict[str, str]] = {
    "tie": "tie",
    "necktie": "necktie",
    "bow tie": "bow tie",
}
HATS: Final[dict[str, str]] = {
    "hat": "hat",
    "cap": "cap",
    "beanie": "beanie",
    "beret": "beret",
    "fedora": "fedora",
    "bucket hat": "bucket hat",
}
BAGS: Final[dict[str, str]] = {
    "bag": "bag",
    "handbag": "handbag",
    "backpack": "backpack",
    "purse": "purse",
    "tote": "tote",
    "satchel": "satchel",
    "briefcase": "briefcase",
}
FOOTWEAR: Final[dict[str, str]] = {
    "shoes": "shoes",
    "sneakers": "sneakers",
    "boots": "boots",
    "sandals": "sandals",
    "heels": "heels",
    "loafers": "loafers",
    "flats": "flats",
    "trainers": "trainers",
}
IGNORED_KEYWORDS: Final[set[str]] = {
    "someone",
    "somebody",
    "person",
    "wearing",
    "wears",
    "wear",
    "dressed",
    "outfit",
    "clothing",
    "fashion",
}
FIELD_TO_VOCAB: Final[dict[str, dict[str, str]]] = {
    "scene": SCENES,
    "style": STYLES,
    "upper_garment": UPPER_GARMENTS,
    "lower_garment": LOWER_GARMENTS,
    "outerwear": OUTERWEAR_ITEMS,
    "dress": DRESSES,
    "tie": TIES,
    "hat": HATS,
    "bag": BAGS,
    "footwear": FOOTWEAR,
}


class LlmParserProtocol(Protocol):
    """Protocol for optional LLM-backed query parsing."""

    def __call__(self, query_text: str) -> Mapping[str, Any]: ...


class QueryParser:
    """Parses multimodal user queries into structured components."""

    def __init__(
        self,
        config_path: Path | None = None,
        llm_parser: LlmParserProtocol | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.config_path = config_path
        self.llm_parser = llm_parser
        self.llm_model = llm_model
        self.nlp = self._create_nlp()
        self.matcher = self._create_matcher(nlp=self.nlp)

    def parse(self, query: str | dict[str, Any]) -> dict[str, Any]:
        """Convert a natural-language fashion query into a fixed schema."""
        query_text = self._extract_query_text(query=query)
        if not query_text:
            return self._empty_result()

        llm_result = self._parse_with_llm(query_text=query_text)
        if llm_result is not None:
            LOGGER.info("Parsed query with LLM-backed parser.")
            return llm_result

        LOGGER.info("Falling back to spaCy rule-based query parsing.")
        return self._parse_with_spacy(query_text=query_text)

    def _parse_with_llm(self, query_text: str) -> dict[str, Any] | None:
        """Parse with an LLM if a parser or supported client is available."""
        if self.llm_parser is not None:
            try:
                payload = self.llm_parser(query_text)
            except Exception as error:
                LOGGER.warning("Custom LLM query parser failed: %s", error)
                return None

            return self._normalize_result(payload)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        prompt = (
            "Convert the following natural-language fashion query into JSON with exactly this schema:\n"
            "{"
            '"scene": null,'
            '"style": null,'
            '"upper_garment": null,'
            '"upper_color": null,'
            '"lower_garment": null,'
            '"lower_color": null,'
            '"outerwear": null,'
            '"outerwear_color": null,'
            '"dress": null,'
            '"tie": null,'
            '"hat": null,'
            '"bag": null,'
            '"footwear": null,'
            '"keywords": []'
            "}\n"
            "Return valid JSON only. Use null for missing values.\n"
            f"Query: {query_text}"
        )

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=self.llm_model,
                input=prompt,
            )
            response_text = getattr(response, "output_text", "")
            if not response_text:
                return None

            payload = json.loads(response_text)
        except Exception as error:
            LOGGER.warning("Automatic OpenAI query parsing failed: %s", error)
            return None

        if not isinstance(payload, Mapping):
            return None

        return self._normalize_result(payload)

    def _parse_with_spacy(self, query_text: str) -> dict[str, Any]:
        """Parse a fashion query using spaCy tokenization and rule-based heuristics."""
        doc = self.nlp(query_text)
        result = self._empty_result()
        used_token_indexes: set[int] = set()

        phrase_matches = self._collect_phrase_matches(doc=doc)
        self._assign_single_value(
            result=result,
            field_name="scene",
            matches=phrase_matches["scene"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="style",
            matches=phrase_matches["style"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="dress",
            matches=phrase_matches["dress"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="tie",
            matches=phrase_matches["tie"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="hat",
            matches=phrase_matches["hat"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="bag",
            matches=phrase_matches["bag"],
            used_token_indexes=used_token_indexes,
        )
        self._assign_single_value(
            result=result,
            field_name="footwear",
            matches=phrase_matches["footwear"],
            used_token_indexes=used_token_indexes,
        )

        upper_match = self._select_best_match(phrase_matches["upper_garment"])
        lower_match = self._select_best_match(phrase_matches["lower_garment"])
        outerwear_match = self._select_best_match(phrase_matches["outerwear"])

        if upper_match is not None:
            result["upper_garment"] = self._canonicalize_match(
                field_name="upper_garment",
                match_text=upper_match.text,
            )
            used_token_indexes.update(range(upper_match.start, upper_match.end))
            color_value, color_indexes = self._find_color_near_span(doc=doc, span=upper_match)
            result["upper_color"] = color_value
            used_token_indexes.update(color_indexes)

        if lower_match is not None:
            result["lower_garment"] = self._canonicalize_match(
                field_name="lower_garment",
                match_text=lower_match.text,
            )
            used_token_indexes.update(range(lower_match.start, lower_match.end))
            color_value, color_indexes = self._find_color_near_span(doc=doc, span=lower_match)
            result["lower_color"] = color_value
            used_token_indexes.update(color_indexes)

        if outerwear_match is not None:
            result["outerwear"] = self._canonicalize_match(
                field_name="outerwear",
                match_text=outerwear_match.text,
            )
            used_token_indexes.update(range(outerwear_match.start, outerwear_match.end))
            color_value, color_indexes = self._find_color_near_span(doc=doc, span=outerwear_match)
            result["outerwear_color"] = color_value
            used_token_indexes.update(color_indexes)

        if result["scene"] is None:
            inferred_scene = self._infer_scene_from_keywords(doc=doc)
            if inferred_scene is not None:
                result["scene"] = inferred_scene

        result["keywords"] = self._extract_keywords(
            doc=doc,
            used_token_indexes=used_token_indexes,
        )
        return self._normalize_result(result)

    def _extract_query_text(self, query: str | dict[str, Any]) -> str:
        """Extract raw query text from supported query inputs."""
        if isinstance(query, str):
            return query.strip()

        if "text" in query:
            return str(query["text"]).strip()

        if "query" in query:
            return str(query["query"]).strip()

        raise ValueError("Query input must be a string or a dict containing 'text' or 'query'.")

    def _create_nlp(self) -> Language:
        """Create a lightweight spaCy English pipeline."""
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    def _create_matcher(self, nlp: Language) -> PhraseMatcher:
        """Create a phrase matcher for scenes, garments, and accessories."""
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        for field_name, vocabulary in FIELD_TO_VOCAB.items():
            patterns = [nlp.make_doc(term) for term in vocabulary]
            if patterns:
                matcher.add(field_name, patterns)

        return matcher

    def _collect_phrase_matches(self, doc: Doc) -> dict[str, list[Span]]:
        """Collect grouped phrase matches from the spaCy matcher."""
        grouped_matches: dict[str, list[Span]] = {field_name: [] for field_name in FIELD_TO_VOCAB}
        for match_id, start, end in self.matcher(doc):
            field_name = self.nlp.vocab.strings[match_id]
            grouped_matches[field_name].append(doc[start:end])

        for field_name, spans in grouped_matches.items():
            grouped_matches[field_name] = self._deduplicate_spans(spans)

        return grouped_matches

    def _deduplicate_spans(self, spans: Sequence[Span]) -> list[Span]:
        """Deduplicate spans while preserving order."""
        seen_keys: set[tuple[int, int]] = set()
        deduplicated_spans: list[Span] = []
        for span in spans:
            span_key = (span.start, span.end)
            if span_key in seen_keys:
                continue

            seen_keys.add(span_key)
            deduplicated_spans.append(span)

        return deduplicated_spans

    def _assign_single_value(
        self,
        result: dict[str, Any],
        field_name: str,
        matches: Sequence[Span],
        used_token_indexes: set[int],
    ) -> None:
        """Assign the first matched value for a single-valued field."""
        match = self._select_best_match(matches)
        if match is None:
            return

        result[field_name] = self._canonicalize_match(
            field_name=field_name,
            match_text=match.text,
        )
        used_token_indexes.update(range(match.start, match.end))

    def _select_best_match(self, matches: Sequence[Span]) -> Span | None:
        """Select the most specific match from a list of spans."""
        if not matches:
            return None

        return max(matches, key=lambda span: (span.end - span.start, -span.start))

    def _canonicalize_match(self, field_name: str, match_text: str) -> str:
        """Map a matched phrase to its canonical schema value."""
        normalized_match_text = match_text.strip().lower()
        vocabulary = FIELD_TO_VOCAB.get(field_name, {})
        return vocabulary.get(normalized_match_text, normalized_match_text)

    def _find_color_near_span(self, doc: Doc, span: Span, window: int = 3) -> tuple[str | None, set[int]]:
        """Find a nearby color token for a garment or accessory span."""
        candidate_indexes = list(range(max(0, span.start - window), span.start))
        candidate_indexes.extend(range(span.end, min(len(doc), span.end + 2)))

        for token_index in reversed(candidate_indexes[:window]):
            token = doc[token_index]
            token_text = token.text.lower()
            if token_text in COLORS:
                return (token_text, {token_index})

        for token_index in candidate_indexes:
            token = doc[token_index]
            token_text = token.text.lower()
            if token_text in COLORS:
                return (token_text, {token_index})

        return (None, set())

    def _infer_scene_from_keywords(self, doc: Doc) -> str | None:
        """Infer a scene from individual tokens when phrase matching misses it."""
        for token in doc:
            token_text = token.text.lower()
            if token_text in SCENES:
                return SCENES[token_text]

        return None

    def _extract_keywords(self, doc: Doc, used_token_indexes: set[int]) -> list[str]:
        """Extract remaining content words as free-form query keywords."""
        keywords: list[str] = []
        seen_keywords: set[str] = set()
        for token in doc:
            token_text = token.text.lower()
            if token.i in used_token_indexes:
                continue

            if not token.is_alpha:
                continue

            if token.is_stop or token_text in IGNORED_KEYWORDS:
                continue

            if token_text in COLORS or token_text in SCENES or token_text in STYLES:
                continue

            if token_text in seen_keywords:
                continue

            seen_keywords.add(token_text)
            keywords.append(token_text)

        return keywords

    def _normalize_result(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize any parser output into the fixed query schema."""
        normalized_result = self._empty_result()
        for field_name in SCHEMA_KEYS:
            if field_name == "keywords":
                normalized_result[field_name] = self._normalize_keywords(payload.get(field_name, []))
                continue

            value = payload.get(field_name)
            normalized_result[field_name] = self._normalize_nullable_string(value)

        return normalized_result

    def _normalize_keywords(self, value: Any) -> list[str]:
        """Normalize keywords into a unique list of non-empty strings."""
        if value is None:
            return []

        if isinstance(value, str):
            keyword_candidates = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            keyword_candidates = [str(item) for item in value]
        else:
            keyword_candidates = [str(value)]

        normalized_keywords: list[str] = []
        seen_keywords: set[str] = set()
        for candidate in keyword_candidates:
            normalized_candidate = candidate.strip().lower()
            if not normalized_candidate or normalized_candidate in seen_keywords:
                continue

            seen_keywords.add(normalized_candidate)
            normalized_keywords.append(normalized_candidate)

        return normalized_keywords

    def _normalize_nullable_string(self, value: Any) -> str | None:
        """Normalize nullable string fields."""
        if value is None:
            return None

        normalized_value = str(value).strip().lower()
        if not normalized_value or normalized_value == "null":
            return None

        return normalized_value

    def _empty_result(self) -> dict[str, Any]:
        """Return a fresh empty query schema."""
        return {
            "scene": None,
            "style": None,
            "upper_garment": None,
            "upper_color": None,
            "lower_garment": None,
            "lower_color": None,
            "outerwear": None,
            "outerwear_color": None,
            "dress": None,
            "tie": None,
            "hat": None,
            "bag": None,
            "footwear": None,
            "keywords": [],
        }
