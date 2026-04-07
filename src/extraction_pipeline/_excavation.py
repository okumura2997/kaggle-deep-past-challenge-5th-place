"""Excavation number canonicalization and matching utilities.

Extracted from find_excavation_number_pages.py for reuse in the unified
extraction pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

from extraction_pipeline._json_utils import normalize_ws

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

VOLUME_KEY_RE = re.compile(r"(?i)\bAKT[\s_]*([0-9]+[a-z]?)\b")
TOKEN_PATTERN = (
    r"[0-9]+[a-z]?(?:/[a-z])?(?:\s*\+\s*(?:[0-9]+[a-z]?(?:/[a-z])?|[a-z]))*"
)
PLUS_SPLIT_RE = re.compile(r"\s*\+\s*")
LEADING_NUMBERING_RE = re.compile(r"(?i)^\s*(?:no\.?\s*)?\d+\s*[:.)-]?\s*")
KT_EXCAVATION_RE = re.compile(
    rf"(?ix)\bkt\.?\s*(?P<prefix>[0-9]{{1,3}}|[a-z]{{1,3}})\s*/\s*(?P<series>[a-z])\s*(?P<token>{TOKEN_PATTERN})"
)
BARE_EXCAVATION_RE = re.compile(
    rf"(?ix)\b(?P<prefix>[0-9]{{1,3}}|[a-z]{{1,3}})\s*/\s*(?P<series>[a-z])\s*(?P<token>{TOKEN_PATTERN})"
)
E_EXCAVATION_RE = re.compile(
    rf"(?ix)\be\s*/\s*(?P<series>[a-z])\s*(?P<token>{TOKEN_PATTERN})"
)
KT_SEGMENT_RE = re.compile(
    r"(?ix)^kt\.?\s*(?P<prefix>[0-9]{1,3}|[a-z]{1,3})\s*(?:/\s*|(?=[a-z]\s*[0-9]))(?P<series>[a-z])\s*(?P<token>[0-9]+[a-z]?(?:/[a-z])?)$"
)
SIG_SERIES_SEGMENT_RE = re.compile(
    r"(?ix)^(?P<prefix>[0-9]{1,3}|[a-z]{1,3})\s*(?:/\s*|(?=[a-z]\s*[0-9]))(?P<series>[a-z])\s*(?P<token>[0-9]+[a-z]?(?:/[a-z])?)$"
)
BARE_TOKEN_RE = re.compile(r"(?ix)^(?P<token>[0-9]+[a-z]?(?:/[a-z])?)$")
COMPACT_SIGLUM_RE = re.compile(
    r"(?i)\b(?P<prefix>[0-9]{1,3})(?P<series>[a-z])(?=\s*[0-9])"
)
COMMA_SUFFIX_SHORTHAND_RE = re.compile(
    r"(?ix)^kt\.?\s*(?P<prefix>[0-9]{1,3}|[a-z]{1,3})\s*/\s*(?P<series>[a-z])\s*(?P<number>[0-9]+)(?P<first>[a-z])\s*,\s*(?P<second>[a-z])$"
)
SLASH_SUFFIX_SHORTHAND_RE = re.compile(
    r"(?ix)^kt\.?\s*(?P<prefix>[0-9]{1,3}|[a-z]{1,3})\s*/\s*(?P<series>[a-z])\s*(?P<number>[0-9]+)(?P<first>[a-z])\s*/\s*(?P<second>[a-z])$"
)
AKT1_HEADING_NUMBER_RE = re.compile(
    r"(?ix)^\s*(?:no\.?\s*)?(?P<number>\d{1,4}[a-z]?)"
    r"(?:\s+(?P<suffix>env(?:elope)?|tab(?:let)?|zarf))?\s*[:.)-]?\s*$"
)
AKT1_HEADING_LINE_RE = re.compile(
    r"(?ix)^\s*no\s*[\.:]?\s*(?P<number>(?:\d\s*){1,4}[a-z]?)\s*$"
)
LARSEN_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,4})\s*(?:[.)-]\s*|\s+)(?P<label>.+?)\s*$"
)
ICK4_HEADING_RE = re.compile(r"(?i)^\s*I\s+(\d+(?:\s*\([a-z]\))?)\s*$")

# ---------------------------------------------------------------------------
# Exception tables
# ---------------------------------------------------------------------------

EXCEPTIONAL_KEY_VARIANTS: dict[str, tuple[str, ...]] = {
    "Kt. 88/k 534b": ("Kt. 88/k 534",),
    "Kt. 88/k 707b": ("Kt. 88/k 707",),
    "Kt. 88/k 435+210": ("Kt. 88/k 435",),
    "Kt. 94/k 862a+1635": ("Kt. 94/k 862a",),
    "Kt. 94/k 1056a": ("Kt. 94/k 1056",),
    "Kt. c/k 1015+e": ("Kt. c/k 1015",),
    "Kt. c/k 350": ("Kt. c/k 1350",),
    "Kt. c/k 504": ("Kt. c/k 1504",),
    "Kt. 91/k 515": ("Kt. 91/k 515a", "Kt. 91/k 515b"),
}

MATCH_KEY_VARIANTS: dict[str, tuple[str, ...]] = {}
for _canonical_key, _variant_keys in EXCEPTIONAL_KEY_VARIANTS.items():
    MATCH_KEY_VARIANTS.setdefault(_canonical_key, tuple())
    _related_keys = {_canonical_key, *_variant_keys}
    for _key in _related_keys:
        _existing = set(MATCH_KEY_VARIANTS.get(_key, ()))
        MATCH_KEY_VARIANTS[_key] = tuple(
            sorted(_existing.union(_related_keys.difference({_key})))
        )

EXCEPTIONAL_RAW_CANONICAL_FORMS: dict[str, str] = {
    "kt 91/k k 514": "Kt. 91/k 514",
    "kt. 91/k k 514": "Kt. 91/k 514",
}

KNOWN_NONEXISTENT_TARGETS: set[str] = {
    "Kt. 91/k 502",
    "Kt. 94/k 648",
    "Kt. n/k 593",
}

FORCED_PAGE_LOCATIONS: dict[str, dict[str, object]] = {
    "Kt. n/k 596": {
        "volume_key": "AKT 2",
        "page": 95,
        "expected_pdf_name": "AKT_2_1995_fixed.pdf",
        "detected_text": "Kt. n/k 596",
    },
}

DEFAULT_PAGE_RANGES: dict[str, tuple[int, int]] = {
    "AKT 1": (34, 124),
    "AKT 2": (21, 121),
    "AKT 3": (15, 197),
    "AKT 4": (37, 159),
    "AKT 5": (43, 168),
    "AKT 6A": (52, 436),
    "AKT 6B": (59, 353),
    "AKT 6C": (31, 297),
    "AKT 6D": (11, 147),
    "AKT 6E": (85, 281),
    "AKT 7A": (101, 579),
    "AKT 7B": (102, 420),
    "AKT 8": (49, 522),
    "AKT 9A": (47, 237),
    "AKT 10": (28, 154),
    "AKT 11A": (82, 261),
    "AKT 11B": (37, 180),
    "AKT 12": (43, 174),
    "LARSEN 2002 - THE ASSUR-NADA ARCHIVE. PIHANS 96 2002": (51, 281),
}

PREFERRED_PDF_FILENAMES: dict[str, str] = {
    "AKT 1": "AKT_1_1990_fixed.pdf",
    "AKT 2": "AKT_2_1995_fixed.pdf",
    "AKT 3": "AKT_3_1995_fixed.pdf",
}

# ---------------------------------------------------------------------------
# Token / siglum normalization
# ---------------------------------------------------------------------------


def normalize_token(text: str) -> str:
    token = normalize_ws(text).lower()
    token = re.sub(r"\s+", "", token)
    return re.sub(r"(?<=\d)/(?=[a-z]\b)", "", token)


def normalize_siglum(text: str) -> str:
    siglum = normalize_ws(text).lower()
    if siglum.isdigit():
        return str(int(siglum))
    return siglum


def normalize_compact_siglum(text: str) -> str:
    return COMPACT_SIGLUM_RE.sub(r"\g<prefix>/\g<series>", text)


# ---------------------------------------------------------------------------
# Volume key normalization
# ---------------------------------------------------------------------------


def normalize_volume_key(text: str) -> str:
    normalized = normalize_ws(text)
    if "/" in normalized or "\\" in normalized or normalized.lower().endswith(".pdf"):
        normalized = Path(normalized).stem

    raw = normalized.replace("_", " ").upper()
    match = VOLUME_KEY_RE.search(raw)
    if match is None:
        return raw
    return f"AKT {match.group(1).upper()}"


def is_akt1_volume(volume_key: str) -> bool:
    return normalize_volume_key(volume_key) == "AKT 1"


def is_larsen_volume(volume_key: str) -> bool:
    return normalize_volume_key(volume_key) == "LARSEN 2002 - THE ASSUR-NADA ARCHIVE. PIHANS 96 2002"


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonicalize_compound_excavation(text: str) -> str:
    cleaned = normalize_ws(text)
    if not cleaned:
        return ""

    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = LEADING_NUMBERING_RE.sub("", cleaned)
    cleaned = normalize_compact_siglum(cleaned)
    cleaned = normalize_ws(cleaned)
    if not cleaned:
        return ""

    segments = [
        segment.strip() for segment in PLUS_SPLIT_RE.split(cleaned) if segment.strip()
    ]
    if not segments:
        return ""

    base_prefix = ""
    base_series = ""
    tokens: list[str] = []

    for idx, segment in enumerate(segments):
        match = KT_SEGMENT_RE.match(segment)
        if match is not None:
            prefix = normalize_siglum(match.group("prefix"))
            series = match.group("series").lower()
            token = normalize_token(match.group("token"))
        else:
            match = SIG_SERIES_SEGMENT_RE.match(segment)
            if match is not None:
                prefix = normalize_siglum(match.group("prefix"))
                series = match.group("series").lower()
                token = normalize_token(match.group("token"))
            else:
                bare_match = BARE_TOKEN_RE.match(segment)
                if bare_match is None or idx == 0 or not base_prefix:
                    return ""
                prefix = base_prefix
                series = base_series
                token = normalize_token(bare_match.group("token"))

        if not base_prefix:
            base_prefix = prefix
            base_series = series
        elif prefix != base_prefix or series != base_series:
            return ""
        tokens.append(token)

    if not base_prefix or not base_series or not tokens:
        return ""
    return f"Kt. {base_prefix}/{base_series} {'+'.join(tokens)}"


def canonicalize_excavation(text: str) -> str:
    raw = normalize_ws(text)
    if not raw:
        return ""
    raw = normalize_ws(raw.replace("*", " "))
    if not raw:
        return ""
    exceptional_canonical = EXCEPTIONAL_RAW_CANONICAL_FORMS.get(raw.lower())
    if exceptional_canonical:
        return exceptional_canonical

    sanitized = raw.replace("(", " ").replace(")", " ")
    compound_candidates = [LEADING_NUMBERING_RE.sub("", sanitized)]
    kt_match = re.search(r"(?i)\bkt\.?", sanitized)
    if kt_match is not None:
        compound_candidates.append(sanitized[kt_match.start() :])
    for candidate in compound_candidates:
        canonical = canonicalize_compound_excavation(candidate)
        if canonical:
            return canonical

    sanitized = normalize_compact_siglum(sanitized)
    match = KT_EXCAVATION_RE.search(sanitized)
    if match is not None:
        prefix = match.group("prefix").lower()
        series = match.group("series").lower()
        token = normalize_token(match.group("token"))
        return f"Kt. {prefix}/{series} {token}"

    match = BARE_EXCAVATION_RE.search(sanitized)
    if match is not None:
        prefix = match.group("prefix").lower()
        series = match.group("series").lower()
        token = normalize_token(match.group("token"))
        return f"Kt. {prefix}/{series} {token}"

    match = E_EXCAVATION_RE.search(sanitized)
    if match is not None:
        series = match.group("series").lower()
        token = normalize_token(match.group("token"))
        return f"e/{series} {token}"

    return ""


def canonicalize_akt1_heading_number(text: object) -> str:
    raw = normalize_ws("" if text is None else str(text))
    if not raw:
        return ""

    match = AKT1_HEADING_NUMBER_RE.match(raw)
    if match is None:
        return ""

    number = normalize_ws(match.group("number")).lower()
    if not number:
        return ""
    return number


def canonicalize_larsen_heading(text: object) -> str:
    raw = normalize_ws("" if text is None else str(text))
    if not raw:
        return ""

    match = LARSEN_HEADING_RE.match(raw)
    if match is None:
        return raw

    heading_number = str(int(match.group("number")))
    label = normalize_ws(match.group("label"))
    label = re.sub(r"\s*,\s*", ", ", label)
    label = re.sub(r"\b([A-Za-z])\s+(?=\d)", r"\1", label)
    label = normalize_ws(label)
    if not label:
        return ""
    return f"{heading_number}. {label}"


def canonicalize_ick4_heading(text: object) -> str:
    """Canonicalize ICK 4 heading like 'I 427' → 'I 427'."""
    raw = normalize_ws("" if text is None else str(text))
    if not raw:
        return ""
    match = ICK4_HEADING_RE.match(raw)
    if match is None:
        # Try loose match
        cleaned = normalize_ws(raw)
        if cleaned.upper().startswith("I "):
            return cleaned
        return ""
    return f"I {normalize_ws(match.group(1))}"


def canonicalize_volume_identifier(volume_key: str, text: object) -> str:
    if is_akt1_volume(volume_key):
        akt1_heading_number = canonicalize_akt1_heading_number(text)
        if akt1_heading_number:
            return akt1_heading_number
    if is_larsen_volume(volume_key):
        larsen_heading = canonicalize_larsen_heading(text)
        if larsen_heading:
            return larsen_heading

    raw = "" if text is None else str(text)
    return canonicalize_excavation(raw)


# ---------------------------------------------------------------------------
# Match key expansion
# ---------------------------------------------------------------------------


def expand_match_keys(canonical: str) -> list[str]:
    """Generate match-key variants for a canonical excavation number.

    Handles Kt./Kt period normalization and static exception variants.
    Volume-specific expansions (AKT 1 prefix, Larsen label extraction)
    are handled by VolumeProfile.expand_link_keys() overrides.
    """
    normalized = normalize_ws(canonical)
    if not normalized:
        return []

    keys = [normalized]

    # Generate "Kt." ↔ "Kt " variants for flexible matching
    for k in list(keys):
        if k.startswith("Kt. "):
            alt = "Kt " + k[4:]
            if alt not in keys:
                keys.append(alt)
        elif k.startswith("Kt ") and not k.startswith("Kt. "):
            alt = "Kt. " + k[3:]
            if alt not in keys:
                keys.append(alt)

    for variant in MATCH_KEY_VARIANTS.get(normalized, ()):
        if variant not in keys:
            keys.append(variant)
    return keys


# ---------------------------------------------------------------------------
# Detected text expansion (shorthand splitting)
# ---------------------------------------------------------------------------


def expand_detected_excavation_texts(
    text: str, volume_key: str = ""
) -> list[str]:
    detected_text = normalize_ws(text)
    if not detected_text:
        return []
    if is_larsen_volume(volume_key):
        return [detected_text]

    stripped = LEADING_NUMBERING_RE.sub("", detected_text)
    shorthand_match = COMMA_SUFFIX_SHORTHAND_RE.match(stripped)
    if shorthand_match is None:
        shorthand_match = SLASH_SUFFIX_SHORTHAND_RE.match(stripped)
    if shorthand_match is None:
        return [detected_text]

    prefix = normalize_siglum(shorthand_match.group("prefix"))
    series = shorthand_match.group("series").lower()
    number = shorthand_match.group("number")
    first = shorthand_match.group("first").lower()
    second = shorthand_match.group("second").lower()
    return [
        f"Kt. {prefix}/{series} {number}{first}",
        f"Kt. {prefix}/{series} {number}{second}",
    ]


# ---------------------------------------------------------------------------
# Text layer extraction (AKT 1 fast path)
# ---------------------------------------------------------------------------


def build_textpage_lines(
    text_page,
    y_tolerance: float = 4.0,
    gap_threshold: float = 2.0,
) -> list[str]:
    items: list[dict[str, float | str]] = []
    for rect_index in range(text_page.count_rects()):
        rect = text_page.get_rect(rect_index)
        text = normalize_ws(
            text_page.get_text_bounded(*rect).replace("\r", " ").replace("\n", " ")
        )
        if not text:
            continue

        x0, y0, x1, y1 = rect
        items.append(
            {
                "text": text,
                "x0": x0,
                "x1": x1,
                "y_center": (y0 + y1) / 2.0,
            }
        )

    items.sort(key=lambda item: (-float(item["y_center"]), float(item["x0"])))
    lines: list[dict[str, object]] = []
    for item in items:
        y_center = float(item["y_center"])
        matched_line: dict[str, object] | None = None
        for existing_line in lines:
            if abs(float(existing_line["y_center"]) - y_center) <= y_tolerance:
                matched_line = existing_line
                break

        if matched_line is None:
            lines.append({"y_center": y_center, "items": [item]})
            continue

        line_items = matched_line["items"]
        assert isinstance(line_items, list)
        line_items.append(item)

    output_lines: list[str] = []
    for line in lines:
        line_items = line["items"]
        assert isinstance(line_items, list)
        line_items.sort(key=lambda item: float(item["x0"]))

        parts: list[str] = []
        last_x1: float | None = None
        for item in line_items:
            x0 = float(item["x0"])
            if last_x1 is not None and x0 - last_x1 > gap_threshold:
                parts.append(" ")
            parts.append(str(item["text"]))
            last_x1 = float(item["x1"])

        line_text = normalize_ws("".join(parts))
        if line_text:
            output_lines.append(line_text)
    return output_lines


def extract_akt1_heading_numbers_from_text_layer(
    pdf_path: Path,
    page_num: int,
) -> list[dict] | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None

    from extraction_pipeline._pdf_renderer import _PDFIUM_LOCK

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            text_page = doc[page_num - 1].get_textpage()
            rows: list[dict] = []
            seen: set[str] = set()
            for line in build_textpage_lines(text_page):
                match = AKT1_HEADING_LINE_RE.match(line)
                if match is None:
                    continue

                number = re.sub(r"\s+", "", match.group("number")).lower()
                detected_text = f"No. {number}"
                canonical_excavation_number = canonicalize_akt1_heading_number(
                    detected_text
                )
                if not canonical_excavation_number:
                    continue

                dedupe_key = canonical_excavation_number.lower()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(
                    {
                        "detected_text": detected_text,
                        "canonical_excavation_number": canonical_excavation_number,
                    }
                )
            return rows
        finally:
            doc.close()


def normalize_detected_excavation_numbers(
    parsed: dict, volume_key: str
) -> list[dict]:
    raw_values = parsed.get("excavation_numbers", [])
    if not isinstance(raw_values, list):
        return []

    rows: list[dict] = []
    seen: set[str] = set()
    for value in raw_values:
        for detected_text in expand_detected_excavation_texts(
            str(value), volume_key=volume_key
        ):
            if not detected_text:
                continue
            canonical_excavation_number = canonicalize_volume_identifier(
                volume_key, detected_text
            )
            if not canonical_excavation_number:
                continue
            dedupe_key = canonical_excavation_number.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "detected_text": detected_text,
                    "canonical_excavation_number": canonical_excavation_number,
                }
            )
    return rows
