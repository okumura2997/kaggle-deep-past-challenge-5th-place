"""JSON parsing utilities shared across extraction modules."""

from __future__ import annotations

import json
import re


def parse_first_json_object(text: str) -> dict:
    """Extract the first JSON object from text with fallback decoder."""
    content = (text or "").strip()
    if not content:
        return {}

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def extract_final_answer(raw_text: str) -> str:
    """Extract content after </think> tag and remove markdown code blocks."""
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_ws(text: str) -> str:
    """Normalize whitespace to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()
