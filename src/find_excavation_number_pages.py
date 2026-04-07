import gc
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
import tyro
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


STANDARD_DETECT_SYSTEM_PROMPT = """You extract excavation numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY excavation numbers used as the main text heading/section heading, or on a metadata line such as "Fundnummer".
- Return only the excavation number itself, not running index, chapter number, parentheses, page title, or surrounding words.
- Valid examples: "Kt. v/k 8", "Kt. 88/k 110", "Kt. g/k 1b", "Kt. c/k 929+1118+1450", "Kt. y/t 4".
- If the page shows "Fundnummer: Kt. v/k 8", output only "Kt. v/k 8".
- If the page shows "1. Kt. 88/k 110" or "No. 1: 88/k 110", output only "Kt. 88/k 110".
- Ignore inline/body-text mentions, commentary, notes, bibliography, running headers, footnotes, and references.
- Ignore tables of contents, indexes, and list pages that are not the main edition text of a document.
- Keep the output in visual reading order.
- If no readable excavation number exists, return an empty list.
"""

STANDARD_DETECT_USER_PROMPT_TEMPLATE = """Extract excavation numbers from this page image.
Target page number: {page_number}

Important:
- Return only excavation numbers like "Kt. v/k 8", "Kt. 88/k 110", or "Kt. y/t 4".
- Do not include running indices such as "1." or "No. 1:".
- Do not include the word "Fundnummer:".
- Do not extract inline/body mentions that are not the main heading or metadata label for the document on this page.

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""

AKT1_DETECT_SYSTEM_PROMPT = """You extract AKT 1 heading numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY main heading numbers written like "No. 1", "No. 2", or "No. 27".
- Return only the heading number itself, normalized as "No. <number>".
- Ignore body-text references such as "a/k 533", "BIN IV 12", commentary, notes, bibliography, running headers, and footnotes.
- Ignore subsection labels such as "Tablet", "Zarf", "St. 2", and page numbers.
- Keep the output in visual reading order.
- If no readable heading number exists, return an empty list.
"""

AKT1_DETECT_USER_PROMPT_TEMPLATE = """Extract AKT 1 main heading numbers from this page image.
Target page number: {page_number}

Important:
- Extract only main headings like "No. 1", "No. 2", or "No. 27".
- Do not extract excavation references such as "a/k 533" from body text.
- Do not extract page numbers or subsection labels such as "Tablet", "Zarf", or "St. 2".

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""

LARSEN_DETECT_SYSTEM_PROMPT = """You extract main section headings from Larsen's scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY the main section/document heading shown for the edited text on the page.
- Return the heading exactly as printed, including the running section number and reference label.
- Valid examples: "4. C33", "5. CCT 4, lb", "6. TC 3, 95".
- Keep the heading label itself; do not rewrite it into Kt.-style notation.
- Ignore the transliteration text, translation text, body lines, running headers, footnotes, commentary, bibliography, and page numbers.
- Ignore table-of-contents or index pages that list many headings without presenting the edited text itself.
- Keep output in visual reading order.
- If no readable main heading exists, return an empty list.
"""

LARSEN_DETECT_USER_PROMPT_TEMPLATE = """Extract the main section headings from this Larsen edition page image.
Target page number: {page_number}

Important:
- Return the heading exactly as shown on the page.
- Include the leading running number when present, such as "4. C33".
- Do not convert the label into Kt.-style excavation notation.
- Do not extract body-text lines from the transliteration or translation columns.

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""


ICK_PRAG_DETECT_SYSTEM_PROMPT = """You extract ICK 4 main heading numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY the main heading/document number written like "I 441" or "I 704".
- Return only the heading identifier itself, normalized as "I <number>".
- Ignore transliteration text, translation text, commentary, notes, bibliography, running headers, footnotes, and page numbers.
- Ignore inline references to other texts and ignore table-of-contents or index pages.
- Keep output in visual reading order.
- If no readable main heading exists, return an empty list.
"""

ICK_PRAG_DETECT_USER_PROMPT_TEMPLATE = """Extract the main heading numbers from this ICK 4 edition page image.
Target page number: {page_number}

Important:
- Extract only main headings like "I 441" or "I 704".
- Return the heading as "I <number>" with a single space.
- Do not extract page numbers or body-text references.

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""
DETECT_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_numbers": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["excavation_numbers"],
    "additionalProperties": False,
}


ICK_PRAG_PDF_FILENAME = (
    "Hecker Kryszat Matous - Kappadokische Keilschrifttafeln aus den "
    "Sammlungen der Karlsuniversitat Prag. ICK 4 1998.pdf"
)
ICK_PRAG_PDF_PATH = f"input/pdfs/{ICK_PRAG_PDF_FILENAME}"
ICK_PRAG_VOLUME_KEY = (
    "HECKER KRYSZAT MATOUS - KAPPADOKISCHE KEILSCHRIFTTAFELN AUS DEN "
    "SAMMLUNGEN DER KARLSUNIVERSITAT PRAG. ICK 4 1998"
)

PAGE_IMAGE_RE = re.compile(r"^page_(\d{3,})\.(?:png|jpg|jpeg|webp)$", flags=re.IGNORECASE)
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
COMPACT_SIGLUM_RE = re.compile(r"(?i)\b(?P<prefix>[0-9]{1,3})(?P<series>[a-z])(?=\s*[0-9])")
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
ICK_PRAG_HEADING_RE = re.compile(
    r"(?ix)^\s*(?:i|l)\s*[\.:-]?\s*(?P<number>\d{1,4}[a-z]?)\s*$"
)

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
for canonical_key, variant_keys in EXCEPTIONAL_KEY_VARIANTS.items():
    MATCH_KEY_VARIANTS.setdefault(canonical_key, tuple())
    related_keys = {canonical_key, *variant_keys}
    for key in related_keys:
        existing = set(MATCH_KEY_VARIANTS.get(key, ()))
        MATCH_KEY_VARIANTS[key] = tuple(
            sorted(existing.union(related_keys.difference({key})))
        )

EXCEPTIONAL_RAW_CANONICAL_FORMS: dict[str, str] = {
    "kt 91/k k 514": "Kt. 91/k 514",
    "kt. 91/k k 514": "Kt. 91/k 514",
}

# Catalog rows that should not produce "missing excavation number" targets.
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

DETECT_CACHE_VERSION = "heading_or_fundnummer_v1"
PREFERRED_PDF_FILENAMES: dict[str, str] = {
    "AKT 1": "AKT_1_1990_fixed.pdf",
    "AKT 2": "AKT_2_1995_fixed.pdf",
    "AKT 3": "AKT_3_1995_fixed.pdf",
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
    ICK_PRAG_VOLUME_KEY: (26, 406),
}
DEFAULT_PDF_ONLY_VOLUMES: tuple[str, ...] = (
    "AKT 7B",
    "AKT 11B",
    "AKT 12",
    "input/pdfs/Larsen 2002 - The Assur-nada Archive. PIHANS 96 2002.pdf",
    ICK_PRAG_PDF_PATH,
)


@dataclass
class Args:
    csv_path: str = "./data/published_texts_akt_subset.csv"
    pdf_root: str = "./input/pdfs"
    output_root: str = "./data/find_excavation_number_pages"
    dpi: int = 140
    render_workers: int = 32
    use_known_page_ranges: bool = True
    fallback_start_page: int = 1
    fallback_end_page: int = -1
    target_volumes: list[str] = field(default_factory=list)
    exclude_volumes: list[str] = field(default_factory=list)
    pdf_only_volumes: list[str] = field(
        default_factory=lambda: list(DEFAULT_PDF_ONLY_VOLUMES)
    )
    dryrun_volumes: int = -1
    dryrun_pages_per_volume: int = -1
    retry_missing_pdf_iterations: int = 5
    pdf_only_detection_passes: int = 2

    model: str = "Qwen/Qwen3.5-9B"
    detect_batch_size: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0

    repetition_penalty: float = 1.0
    detect_max_tokens: int = 512
    use_structured_outputs: bool = True

    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    seed: int = 42
    trust_remote_code: bool = False
    enforce_eager: bool = False


@dataclass
class VolumeJob:
    akt_volume: str
    volume_key: str
    pdf_path: Path
    run_name: str
    run_dir: Path
    images_dir: Path
    numbers_dir: Path
    raw_output_path: Path
    output_numbers_path: Path
    start_page: int
    end_page: int
    rendered: list[tuple[int, Path]]


def parse_first_json_object(text: str) -> dict:
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
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def is_ick_prag_volume(volume_key: str) -> bool:
    return normalize_volume_key(volume_key) == ICK_PRAG_VOLUME_KEY


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

    segments = [segment.strip() for segment in PLUS_SPLIT_RE.split(cleaned) if segment.strip()]
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


def expand_match_keys(canonical: str) -> list[str]:
    normalized = normalize_ws(canonical)
    if not normalized:
        return []

    keys = [normalized]
    for variant in MATCH_KEY_VARIANTS.get(normalized, ()):
        if variant not in keys:
            keys.append(variant)
    return keys


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
        compound_candidates.append(sanitized[kt_match.start():])
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


def canonicalize_ick_prag_heading(text: object) -> str:
    raw = normalize_ws("" if text is None else str(text))
    if not raw:
        return ""

    raw = LEADING_NUMBERING_RE.sub("", raw)
    match = ICK_PRAG_HEADING_RE.match(raw)
    if match is None:
        return ""

    number = normalize_ws(match.group("number")).lower()
    number = re.sub(r"\s+", "", number)
    if not number:
        return ""
    return f"I {number}"


def canonicalize_volume_identifier(volume_key: str, text: object) -> str:
    if is_akt1_volume(volume_key):
        akt1_heading_number = canonicalize_akt1_heading_number(text)
        if akt1_heading_number:
            return akt1_heading_number
    if is_larsen_volume(volume_key):
        larsen_heading = canonicalize_larsen_heading(text)
        if larsen_heading:
            return larsen_heading
    if is_ick_prag_volume(volume_key):
        ick_prag_heading = canonicalize_ick_prag_heading(text)
        if ick_prag_heading:
            return ick_prag_heading

    raw = "" if text is None else str(text)
    return canonicalize_excavation(raw)


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
            canonical_excavation_number = canonicalize_akt1_heading_number(detected_text)
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


def empty_page_numbers_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "akt_volume": pl.String,
            "volume_key": pl.String,
            "pdf_name": pl.String,
            "pdf_path": pl.String,
            "page": pl.Int64,
            "image_path": pl.String,
            "number_index": pl.Int64,
            "detected_text": pl.String,
            "canonical_excavation_number": pl.String,
        }
    )


def empty_unique_locations_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "akt_volume": pl.String,
            "volume_key": pl.String,
            "excavation_no": pl.String,
            "canonical_excavation_number": pl.String,
            "pdf_name": pl.String,
            "pdf_path": pl.String,
            "page": pl.Int64,
            "image_path": pl.String,
            "number_index": pl.Int64,
            "detected_text": pl.String,
        }
    )


def build_pdf_index(pdf_root: Path) -> dict[str, Path]:
    preferred_paths = {
        volume_key: pdf_root / filename
        for volume_key, filename in PREFERRED_PDF_FILENAMES.items()
        if (pdf_root / filename).exists()
    }
    pdf_index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for pdf_path in sorted(pdf_root.glob("*.pdf")):
        volume_key = normalize_volume_key(pdf_path.stem)
        preferred_path = preferred_paths.get(volume_key)
        if preferred_path is not None:
            if pdf_path == preferred_path:
                pdf_index[volume_key] = pdf_path
            continue
        if volume_key in pdf_index:
            duplicates.setdefault(volume_key, [pdf_index[volume_key]]).append(pdf_path)
            continue
        pdf_index[volume_key] = pdf_path

    if duplicates:
        lines = []
        for volume_key, paths in sorted(duplicates.items()):
            rendered_paths = ", ".join(str(path) for path in paths)
            lines.append(f"{volume_key}: {rendered_paths}")
        raise RuntimeError("Multiple PDFs resolved to the same AKT volume:\n" + "\n".join(lines))
    return pdf_index


def resolve_page_range(args: Args, volume_key: str) -> tuple[int, int]:
    if args.use_known_page_ranges and volume_key in DEFAULT_PAGE_RANGES:
        return DEFAULT_PAGE_RANGES[volume_key]
    return args.fallback_start_page, args.fallback_end_page


def chunk_page_tasks(
    page_tasks: list[tuple[int, Path]],
    chunk_count: int,
) -> list[list[tuple[int, Path]]]:
    if not page_tasks:
        return []
    chunk_size = max(1, (len(page_tasks) + chunk_count - 1) // chunk_count)
    return [page_tasks[i : i + chunk_size] for i in range(0, len(page_tasks), chunk_size)]


def render_pdf_page_chunk(
    pdf_path_str: str,
    page_tasks: list[tuple[int, str]],
    scale: float,
) -> tuple[int, list[tuple[int, str]]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path_str)
    try:
        rendered_count = 0
        rendered_paths: list[tuple[int, str]] = []
        for page_num, image_path_str in page_tasks:
            image_path = Path(image_path_str)
            if not image_path.exists():
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                rendered_count += 1
            rendered_paths.append((page_num, image_path_str))
        return rendered_count, rendered_paths
    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    start_page: int,
    end_page: int,
    dpi: int,
    render_workers: int,
) -> list[tuple[int, Path]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pypdfium2 as pdfium
    except ImportError:
        existing: list[tuple[int, Path]] = []
        for path in sorted(images_dir.glob("page_*")):
            match = PAGE_IMAGE_RE.match(path.name)
            if match is None:
                continue
            existing.append((int(match.group(1)), path))

        if not existing:
            raise ImportError("pypdfium2 is required. Install with: pip install pypdfium2")

        if start_page < 1:
            raise ValueError(f"start_page must be >= 1, got {start_page}")
        max_existing_page = max(page for page, _ in existing)
        effective_end_page = max_existing_page if end_page < 1 else end_page
        rendered = [
            (page, path)
            for page, path in existing
            if start_page <= page <= effective_end_page
        ]
        if not rendered:
            raise RuntimeError(
                "No cached page images found in requested range. "
                "Install pypdfium2 to render PDF pages."
            )
        print(
            "[warn] pypdfium2 is not installed. "
            f"Using existing cached images only: pdf={pdf_path.name}, pages={len(rendered)}"
        )
        return rendered

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        if start_page < 1:
            raise ValueError(f"start_page must be >= 1, got {start_page}")

        effective_end_page = total_pages if end_page < 1 else end_page
        if effective_end_page < start_page:
            raise ValueError(
                f"Invalid page range: start_page={start_page}, end_page={effective_end_page}"
            )
        if effective_end_page > total_pages:
            raise ValueError(
                f"end_page={effective_end_page} exceeds PDF pages={total_pages}"
            )

        rendered: list[tuple[int, Path]] = []
        num_skipped = 0
        pending_page_tasks: list[tuple[int, Path]] = []
        for page_num in range(start_page, effective_end_page + 1):
            image_path = images_dir / f"page_{page_num:03d}.png"
            if image_path.exists():
                num_skipped += 1
                rendered.append((page_num, image_path))
                continue
            pending_page_tasks.append((page_num, image_path))

        num_rendered = 0
        scale = dpi / 72.0
        worker_count = max(1, min(render_workers, len(pending_page_tasks))) if pending_page_tasks else 0
        if worker_count <= 1:
            for page_num, image_path in tqdm(
                pending_page_tasks,
                desc=f"render_pdf[{pdf_path.stem}]",
            ):
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                num_rendered += 1
                rendered.append((page_num, image_path))
        else:
            page_task_chunks = chunk_page_tasks(pending_page_tasks, worker_count)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        render_pdf_page_chunk,
                        str(pdf_path),
                        [(page_num, str(image_path)) for page_num, image_path in chunk],
                        scale,
                    )
                    for chunk in page_task_chunks
                ]
                with tqdm(total=len(pending_page_tasks), desc=f"render_pdf[{pdf_path.stem}]") as pbar:
                    for future in as_completed(futures):
                        rendered_count, rendered_paths = future.result()
                        num_rendered += rendered_count
                        rendered.extend(
                            (page_num, Path(image_path_str))
                            for page_num, image_path_str in rendered_paths
                        )
                        pbar.update(len(rendered_paths))

        rendered.sort(key=lambda item: item[0])

        print(
            f"render_pdf done. pdf={pdf_path.name}, pages={len(rendered)}, "
            f"rendered={num_rendered}, skipped_existing={num_skipped}"
        )
        return rendered
    finally:
        doc.close()


def make_detect_messages(volume_key: str, page_number: int, image_path: Path) -> list[dict]:
    if is_akt1_volume(volume_key):
        system_prompt = AKT1_DETECT_SYSTEM_PROMPT
        user_prompt = AKT1_DETECT_USER_PROMPT_TEMPLATE.format(page_number=page_number)
    elif is_larsen_volume(volume_key):
        system_prompt = LARSEN_DETECT_SYSTEM_PROMPT
        user_prompt = LARSEN_DETECT_USER_PROMPT_TEMPLATE.format(page_number=page_number)
    elif is_ick_prag_volume(volume_key):
        system_prompt = ICK_PRAG_DETECT_SYSTEM_PROMPT
        user_prompt = ICK_PRAG_DETECT_USER_PROMPT_TEMPLATE.format(page_number=page_number)
    else:
        system_prompt = STANDARD_DETECT_SYSTEM_PROMPT
        user_prompt = STANDARD_DETECT_USER_PROMPT_TEMPLATE.format(page_number=page_number)

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"file://{image_path.resolve()}"}},
            ],
        },
    ]


def build_sampling_params(args: Args, schema: dict, max_tokens: int) -> SamplingParams:
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": max_tokens,
    }
    if args.use_structured_outputs:
        if StructuredOutputsParams is None:
            print("[warn] StructuredOutputsParams is not available. Continue without constrained decoding.")
        else:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=schema,
                disable_fallback=True,
            )
    return SamplingParams(**sampling_kwargs)


def normalize_detected_excavation_numbers(parsed: dict, volume_key: str) -> list[dict]:
    raw_values = parsed.get("excavation_numbers", [])
    if not isinstance(raw_values, list):
        return []

    rows: list[dict] = []
    seen: set[str] = set()
    for value in raw_values:
        for detected_text in expand_detected_excavation_texts(str(value), volume_key=volume_key):
            if not detected_text:
                continue
            canonical_excavation_number = canonicalize_volume_identifier(volume_key, detected_text)
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


def expand_detected_excavation_texts(text: str, volume_key: str = "") -> list[str]:
    detected_text = normalize_ws(text)
    if not detected_text:
        return []
    if is_larsen_volume(volume_key) or is_ick_prag_volume(volume_key):
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


def merge_detected_page_numbers(
    existing_rows: list[dict],
    new_rows: list[dict],
) -> list[dict]:
    merged_by_canonical: dict[str, dict] = {}

    for row in existing_rows + new_rows:
        canonical = normalize_ws(str(row.get("canonical_excavation_number", "")))
        if not canonical:
            continue
        if canonical not in merged_by_canonical:
            merged_by_canonical[canonical] = dict(row)
            continue

        current = merged_by_canonical[canonical]
        current_text = normalize_ws(str(current.get("detected_text", "")))
        candidate_text = normalize_ws(str(row.get("detected_text", "")))
        if candidate_text and (not current_text or len(candidate_text) < len(current_text)):
            current["detected_text"] = candidate_text

    return list(merged_by_canonical.values())


def extract_page_numbers_from_payload(raw_payload: dict) -> list[dict]:
    candidate_payloads: list[dict] = []
    volume_key = str(raw_payload.get("volume_key", ""))

    for field_name in ("parsed", "attempt_parsed"):
        payload = raw_payload.get(field_name)
        if isinstance(payload, dict):
            candidate_payloads.append(payload)

    detection_attempts = raw_payload.get("detection_attempts")
    if isinstance(detection_attempts, list):
        for attempt in detection_attempts:
            if not isinstance(attempt, dict):
                continue
            for field_name in ("parsed", "attempt_parsed"):
                payload = attempt.get(field_name)
                if isinstance(payload, dict):
                    candidate_payloads.append(payload)

    merged_rows: list[dict] = []
    for payload in candidate_payloads:
        normalized_rows = normalize_detected_excavation_numbers(payload, volume_key=volume_key)
        merged_rows = merge_detected_page_numbers(merged_rows, normalized_rows)
    return merged_rows


def build_page_number_records(
    job: VolumeJob,
    page_num: int,
    image_path: Path,
    page_numbers: list[dict],
) -> list[dict]:
    return [
        {
            "akt_volume": job.akt_volume,
            "volume_key": job.volume_key,
            "pdf_name": job.pdf_path.name,
            "pdf_path": str(job.pdf_path),
            "page": page_num,
            "image_path": str(image_path),
            "number_index": idx,
            "detected_text": row["detected_text"],
            "canonical_excavation_number": row["canonical_excavation_number"],
        }
        for idx, row in enumerate(page_numbers)
    ]


def merge_page_number_records(
    existing_records: list[dict],
    new_records: list[dict],
) -> list[dict]:
    merged_by_canonical: dict[str, dict] = {}

    for row in existing_records + new_records:
        canonical = normalize_ws(str(row.get("canonical_excavation_number", ""))).lower()
        if not canonical:
            continue
        if canonical not in merged_by_canonical:
            merged_by_canonical[canonical] = dict(row)
            continue

        current = merged_by_canonical[canonical]
        current_text = normalize_ws(str(current.get("detected_text", "")))
        candidate_text = normalize_ws(str(row.get("detected_text", "")))
        if candidate_text and (not current_text or len(candidate_text) < len(current_text)):
            current["detected_text"] = candidate_text

    merged_records: list[dict] = []
    for number_index, row in enumerate(merged_by_canonical.values()):
        merged_row = dict(row)
        merged_row["number_index"] = number_index
        merged_records.append(merged_row)
    return merged_records


def build_detection_attempt(raw_payload: dict) -> dict:
    return {
        "retry_iteration": int(raw_payload.get("retry_iteration", 0)),
        "raw_text": str(raw_payload.get("raw_text", "")),
        "parsed": raw_payload.get("attempt_parsed", raw_payload.get("parsed", {})),
        "num_excavation_numbers": int(raw_payload.get("attempt_num_excavation_numbers", raw_payload.get("num_excavation_numbers", 0))),
    }


def merge_page_raw_payload(
    existing_payload: dict | None,
    attempt_payload: dict,
    merged_records: list[dict],
) -> dict:
    detection_attempts: list[dict] = []
    if existing_payload:
        existing_attempts = existing_payload.get("detection_attempts")
        if isinstance(existing_attempts, list) and existing_attempts:
            detection_attempts.extend(
                attempt for attempt in existing_attempts if isinstance(attempt, dict)
            )
        else:
            detection_attempts.append(build_detection_attempt(existing_payload))

    detection_attempts.append(build_detection_attempt(attempt_payload))

    aggregated_payload = dict(attempt_payload)
    aggregated_payload["parsed"] = {
        "excavation_numbers": [row["detected_text"] for row in merged_records]
    }
    aggregated_payload["num_excavation_numbers"] = len(merged_records)
    aggregated_payload["detection_attempts"] = detection_attempts
    return aggregated_payload


def build_numbers_df_from_page_map(
    page_records_by_key: dict[tuple[str, int], list[dict]],
) -> pl.DataFrame:
    all_records = [
        row
        for rows in page_records_by_key.values()
        for row in rows
    ]
    if not all_records:
        return empty_page_numbers_df()
    return pl.DataFrame(all_records).sort(["volume_key", "page", "number_index"])


def empty_match_summary_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "volume_key": pl.String,
            "canonical_excavation_number": pl.String,
            "match_count": pl.Int64,
            "pdf_names": pl.String,
            "pages": pl.String,
            "found": pl.Boolean,
        }
    )


def build_match_summary_df(unique_locations_df: pl.DataFrame) -> pl.DataFrame:
    if unique_locations_df.height == 0:
        return empty_match_summary_df()

    return (
        unique_locations_df
        .group_by(["volume_key", "canonical_excavation_number"])
        .agg(
            pl.col("pdf_name").drop_nulls().unique().sort().alias("pdf_name_list"),
            pl.col("page").drop_nulls().unique().sort().alias("page_list"),
            pl.col("page").drop_nulls().n_unique().alias("match_count"),
        )
        .with_columns(
            pl.col("pdf_name_list")
            .map_elements(stringify_unique_list, return_dtype=pl.String)
            .alias("pdf_names"),
            pl.col("page_list")
            .map_elements(stringify_int_list, return_dtype=pl.String)
            .alias("pages"),
            (pl.col("match_count") > 0).alias("found"),
        )
        .drop(["pdf_name_list", "page_list"])
    )


def build_detection_only_unique_locations_df(
    numbers_df: pl.DataFrame,
    volume_keys: set[str],
) -> pl.DataFrame:
    if not volume_keys or numbers_df.height == 0:
        return empty_unique_locations_df()

    return (
        numbers_df
        .filter(pl.col("volume_key").is_in(sorted(volume_keys)))
        .with_columns(
            pl.col("canonical_excavation_number").alias("excavation_no"),
        )
        .select(
            [
                "akt_volume",
                "volume_key",
                "excavation_no",
                "canonical_excavation_number",
                "pdf_name",
                "pdf_path",
                "page",
                "image_path",
                "number_index",
                "detected_text",
            ]
        )
        .unique(
            subset=[
                "volume_key",
                "canonical_excavation_number",
                "pdf_name",
                "pdf_path",
                "page",
                "image_path",
                "number_index",
            ],
            maintain_order=True,
        )
        .sort(["akt_volume", "canonical_excavation_number", "page", "number_index"])
    )


def build_detection_only_subset_summary_df(
    targets_df: pl.DataFrame,
    numbers_df: pl.DataFrame,
    volume_keys: set[str],
) -> pl.DataFrame:
    if not volume_keys or numbers_df.height == 0:
        return pl.DataFrame(
            schema={
                "input_row_index": pl.UInt32,
                "oare_id": pl.String,
                "transliteration": pl.String,
                "akt_volume": pl.String,
                "excavation_no": pl.String,
                "volume_key": pl.String,
                "canonical_excavation_number": pl.String,
                "match_count": pl.Int64,
                "pdf_names": pl.String,
                "pages": pl.String,
                "found": pl.Boolean,
            }
        )

    max_input_row_index = -1
    if targets_df.height and "input_row_index" in targets_df.columns:
        max_input_row_index = int(targets_df.get_column("input_row_index").max())

    return (
        numbers_df
        .filter(pl.col("volume_key").is_in(sorted(volume_keys)))
        .group_by(["akt_volume", "volume_key", "canonical_excavation_number"])
        .agg(
            pl.col("pdf_name").drop_nulls().unique().sort().alias("pdf_name_list"),
            pl.col("page").drop_nulls().unique().sort().alias("page_list"),
            pl.col("page").drop_nulls().n_unique().alias("match_count"),
        )
        .with_columns(
            pl.col("canonical_excavation_number").alias("excavation_no"),
            pl.col("pdf_name_list")
            .map_elements(stringify_unique_list, return_dtype=pl.String)
            .alias("pdf_names"),
            pl.col("page_list")
            .map_elements(stringify_int_list, return_dtype=pl.String)
            .alias("pages"),
            (pl.col("match_count") > 0).alias("found"),
            pl.lit("").alias("oare_id"),
            pl.lit("").alias("transliteration"),
        )
        .drop(["pdf_name_list", "page_list"])
        .sort(["akt_volume", "canonical_excavation_number"])
        .with_row_index("input_row_index", offset=max_input_row_index + 1)
        .select(
            [
                "input_row_index",
                "oare_id",
                "transliteration",
                "akt_volume",
                "excavation_no",
                "volume_key",
                "canonical_excavation_number",
                "match_count",
                "pdf_names",
                "pages",
                "found",
            ]
        )
    )


def compute_output_tables(
    targets_df: pl.DataFrame,
    unique_targets_df: pl.DataFrame,
    numbers_df: pl.DataFrame,
    scan_volume_keys: set[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    target_volume_keys = (
        set(unique_targets_df.get_column("volume_key").to_list())
        if unique_targets_df.height
        else set()
    )
    pdf_only_volume_keys = set(scan_volume_keys).difference(target_volume_keys)

    if unique_targets_df.height:
        target_match_keys_df = (
            unique_targets_df
            .with_columns(
                pl.col("canonical_excavation_number")
                .map_elements(expand_match_keys, return_dtype=pl.List(pl.String))
                .alias("match_keys"),
            )
            .explode("match_keys")
            .rename({"match_keys": "match_key"})
        )
        detection_match_keys_df = (
            numbers_df
            .with_columns(
                pl.col("canonical_excavation_number")
                .map_elements(expand_match_keys, return_dtype=pl.List(pl.String))
                .alias("match_keys"),
            )
            .explode("match_keys")
            .rename({"match_keys": "match_key"})
            .select(
                [
                    "volume_key",
                    "match_key",
                    "pdf_name",
                    "pdf_path",
                    "page",
                    "image_path",
                    "number_index",
                    "detected_text",
                    "canonical_excavation_number",
                ]
            )
            .unique(subset=["volume_key", "match_key", "page", "number_index"])
        )

        target_unique_locations_df = (
            target_match_keys_df
            .join(
                detection_match_keys_df,
                on=["volume_key", "match_key"],
                how="left",
                suffix="_detected",
            )
            .drop("match_key")
            .unique(
                subset=[
                    "volume_key",
                    "canonical_excavation_number",
                    "pdf_name",
                    "pdf_path",
                    "page",
                    "image_path",
                    "number_index",
                    "detected_text",
                    "canonical_excavation_number_detected",
                ],
                maintain_order=True,
            )
            .drop("canonical_excavation_number_detected")
            .sort(["akt_volume", "canonical_excavation_number", "page", "number_index"])
        )
        match_summary_df = build_match_summary_df(target_unique_locations_df)
        subset_summary_df = (
            targets_df.join(
                match_summary_df,
                on=["volume_key", "canonical_excavation_number"],
                how="left",
            )
            .with_columns(
                pl.col("match_count").fill_null(0),
                pl.col("pdf_names").fill_null(""),
                pl.col("pages").fill_null(""),
                pl.col("found").fill_null(False),
            )
            .sort("input_row_index")
        )
        missing_targets_df = (
            unique_targets_df.join(
                match_summary_df,
                on=["volume_key", "canonical_excavation_number"],
                how="left",
            )
            .with_columns(
                pl.col("match_count").fill_null(0),
                pl.col("pdf_names").fill_null(""),
                pl.col("pages").fill_null(""),
                pl.col("found").fill_null(False),
            )
            .filter(~pl.col("found"))
            .sort(["akt_volume", "canonical_excavation_number"])
        )
    else:
        target_unique_locations_df = empty_unique_locations_df()
        subset_summary_df = (
            targets_df
            .with_columns(
                pl.lit(0, dtype=pl.Int64).alias("match_count"),
                pl.lit("").alias("pdf_names"),
                pl.lit("").alias("pages"),
                pl.lit(False).alias("found"),
            )
            .sort("input_row_index")
        )
        missing_targets_df = (
            unique_targets_df
            .with_columns(
                pl.lit(0, dtype=pl.Int64).alias("match_count"),
                pl.lit("").alias("pdf_names"),
                pl.lit("").alias("pages"),
                pl.lit(False).alias("found"),
            )
            .sort(["akt_volume", "canonical_excavation_number"])
        )

    pdf_only_unique_locations_df = build_detection_only_unique_locations_df(
        numbers_df=numbers_df,
        volume_keys=pdf_only_volume_keys,
    )
    pdf_only_summary_df = build_detection_only_subset_summary_df(
        targets_df=targets_df,
        numbers_df=numbers_df,
        volume_keys=pdf_only_volume_keys,
    )

    return (
        target_unique_locations_df,
        subset_summary_df,
        missing_targets_df,
        pdf_only_unique_locations_df,
        pdf_only_summary_df,
    )


def persist_current_outputs(
    output_root: Path,
    jobs: list[VolumeJob],
    page_raw_payload_by_key: dict[tuple[str, int], dict],
    targets_df: pl.DataFrame,
    numbers_df: pl.DataFrame,
    unique_locations_df: pl.DataFrame,
    subset_summary_df: pl.DataFrame,
    missing_targets_df: pl.DataFrame,
    pdf_only_unique_locations_df: pl.DataFrame,
    pdf_only_summary_df: pl.DataFrame,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    page_detections_path = output_root / "page_detections.csv"
    unique_locations_path = output_root / "excavation_locations_unique_long.csv"
    subset_summary_path = output_root / "published_texts_akt_subset_with_locations.csv"
    generic_summary_path = output_root / "excavation_targets_with_locations.csv"
    pdf_only_unique_locations_path = output_root / "pdf_only_excavation_locations_unique_long.csv"
    pdf_only_summary_path = output_root / "pdf_only_excavation_numbers_with_locations.csv"
    missing_targets_path = output_root / "missing_excavation_numbers.csv"
    unmatched_targets_path = output_root / "unparsed_excavation_numbers.csv"

    numbers_df.write_csv(page_detections_path)

    for job in jobs:
        per_volume_numbers_df = numbers_df.filter(pl.col("volume_key") == job.volume_key)
        per_volume_numbers_df.write_csv(job.output_numbers_path)

        raw_records = [
            raw_payload
            for (volume_key, _page_num), raw_payload in page_raw_payload_by_key.items()
            if volume_key == job.volume_key
        ]
        raw_records.sort(key=lambda row: int(row.get("page", 0)))
        with job.raw_output_path.open("w", encoding="utf-8") as f:
            for record in raw_records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    unique_locations_df.write_csv(unique_locations_path)
    subset_summary_df.write_csv(subset_summary_path)
    subset_summary_df.write_csv(generic_summary_path)
    pdf_only_unique_locations_df.write_csv(pdf_only_unique_locations_path)
    pdf_only_summary_df.write_csv(pdf_only_summary_path)
    missing_targets_df.write_csv(missing_targets_path)

    unparsed_targets_df = targets_df.filter(pl.col("canonical_excavation_number") == "")
    unparsed_targets_df.write_csv(unmatched_targets_path)

    return {
        "page_detections_path": page_detections_path,
        "unique_locations_path": unique_locations_path,
        "subset_summary_path": subset_summary_path,
        "generic_summary_path": generic_summary_path,
        "pdf_only_unique_locations_path": pdf_only_unique_locations_path,
        "pdf_only_summary_path": pdf_only_summary_path,
        "missing_targets_path": missing_targets_path,
        "unmatched_targets_path": unmatched_targets_path,
    }


def stringify_unique_list(values: object) -> str:
    if isinstance(values, pl.Series):
        raw_values = values.to_list()
    elif isinstance(values, (list, tuple)):
        raw_values = list(values)
    else:
        return ""

    items = [normalize_ws(str(value)) for value in raw_values if normalize_ws(str(value))]
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return ", ".join(unique_items)


def stringify_int_list(values: object) -> str:
    if isinstance(values, pl.Series):
        raw_values = values.to_list()
    elif isinstance(values, (list, tuple)):
        raw_values = list(values)
    else:
        return ""

    normalized = sorted(
        {
            int(normalize_ws(str(value)))
            for value in raw_values
            if value is not None and normalize_ws(str(value))
        }
    )
    return ", ".join(str(value) for value in normalized)


def load_targets(args: Args) -> tuple[pl.DataFrame, pl.DataFrame]:
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    targets_df = (
        pl.read_csv(csv_path)
        .with_row_index(name="input_row_index")
        .with_columns(
            pl.col("akt_volume")
            .map_elements(normalize_volume_key, return_dtype=pl.String)
            .alias("volume_key"),
        )
        .with_columns(
            pl.struct(["volume_key", "excavation_no"])
            .map_elements(
                lambda row: canonicalize_volume_identifier(
                    str(row["volume_key"]),
                    row["excavation_no"],
                ),
                return_dtype=pl.String,
            )
            .alias("canonical_excavation_number"),
        )
    )

    if args.target_volumes:
        requested_volumes = {normalize_volume_key(volume) for volume in args.target_volumes}
        targets_df = targets_df.filter(pl.col("volume_key").is_in(sorted(requested_volumes)))

    excluded_volumes = {normalize_volume_key(volume) for volume in args.exclude_volumes}
    if excluded_volumes:
        targets_df = targets_df.filter(~pl.col("volume_key").is_in(sorted(excluded_volumes)))

    unparsed_targets_df = targets_df.filter(pl.col("canonical_excavation_number") == "")
    if unparsed_targets_df.height:
        print(
            f"[warn] Failed to canonicalize {unparsed_targets_df.height} target rows. "
            "They will be excluded from matching."
        )

    matchable_targets_df = targets_df.filter(pl.col("canonical_excavation_number") != "")
    unique_targets_df = (
        matchable_targets_df
        .group_by(["volume_key", "canonical_excavation_number"])
        .agg(
            pl.col("akt_volume").first().alias("akt_volume"),
            pl.col("excavation_no").first().alias("excavation_no"),
        )
        .select(["akt_volume", "volume_key", "excavation_no", "canonical_excavation_number"])
        .sort(["akt_volume", "canonical_excavation_number"])
    )
    if KNOWN_NONEXISTENT_TARGETS:
        unique_targets_df = unique_targets_df.filter(
            ~pl.col("canonical_excavation_number").is_in(sorted(KNOWN_NONEXISTENT_TARGETS))
        )
    return targets_df, unique_targets_df


def resolve_scan_volume_rows(args: Args, targets_df: pl.DataFrame) -> list[dict[str, str]]:
    target_volume_rows = (
        targets_df
        .select(["akt_volume", "volume_key"])
        .unique()
        .sort("volume_key")
        .to_dicts()
    )
    requested_volumes = {normalize_volume_key(volume) for volume in args.target_volumes}
    excluded_volumes = {normalize_volume_key(volume) for volume in args.exclude_volumes}
    default_pdf_only_volumes = {
        normalize_volume_key(volume)
        for volume in args.pdf_only_volumes
        if normalize_ws(volume)
    }

    display_by_volume = {
        str(row["volume_key"]): str(row["akt_volume"])
        for row in target_volume_rows
    }
    if requested_volumes:
        scan_volume_keys = set(requested_volumes)
    else:
        scan_volume_keys = set(display_by_volume).union(default_pdf_only_volumes)
    if excluded_volumes:
        scan_volume_keys.difference_update(excluded_volumes)

    return [
        {
            "akt_volume": display_by_volume.get(volume_key, volume_key),
            "volume_key": volume_key,
        }
        for volume_key in sorted(scan_volume_keys)
    ]


def prepare_volume_jobs(args: Args, scan_volume_rows: list[dict[str, str]]) -> list[VolumeJob]:
    pdf_root = Path(args.pdf_root)
    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {pdf_root}")

    output_root = Path(args.output_root)
    scans_root = output_root / "scans"
    scans_root.mkdir(parents=True, exist_ok=True)

    pdf_index = build_pdf_index(pdf_root)
    volume_rows = list(scan_volume_rows)
    if args.dryrun_volumes > 0:
        volume_rows = volume_rows[: args.dryrun_volumes]
        print(f"[dryrun] using first {len(volume_rows)} volumes")

    jobs: list[VolumeJob] = []
    for row in volume_rows:
        akt_volume = str(row["akt_volume"])
        volume_key = str(row["volume_key"])
        pdf_path = pdf_index.get(volume_key)
        if pdf_path is None:
            raise FileNotFoundError(f"No PDF found for volume {akt_volume} ({volume_key}) in {pdf_root}")

        start_page, end_page = resolve_page_range(args, volume_key)
        run_name = pdf_path.stem
        run_dir = scans_root / run_name
        images_dir = run_dir / "images"
        numbers_dir = run_dir / "numbers_by_page"
        output_numbers_path = run_dir / "excavation_numbers.csv"
        raw_output_path = run_dir / "raw_detect_responses.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        numbers_dir.mkdir(parents=True, exist_ok=True)

        rendered = render_pdf_pages(
            pdf_path=pdf_path,
            images_dir=images_dir,
            start_page=start_page,
            end_page=end_page,
            dpi=args.dpi,
            render_workers=args.render_workers,
        )
        if args.dryrun_pages_per_volume > 0:
            rendered = rendered[: args.dryrun_pages_per_volume]
            print(f"[dryrun] volume={akt_volume} pages={len(rendered)}")

        print(
            f"[info] prepared volume={akt_volume}, pdf={pdf_path.name}, "
            f"page_range={start_page}-{end_page if end_page > 0 else 'end'}, pages={len(rendered)}"
        )
        jobs.append(
            VolumeJob(
                akt_volume=akt_volume,
                volume_key=volume_key,
                pdf_path=pdf_path,
                run_name=run_name,
                run_dir=run_dir,
                images_dir=images_dir,
                numbers_dir=numbers_dir,
                raw_output_path=raw_output_path,
                output_numbers_path=output_numbers_path,
                start_page=start_page,
                end_page=end_page,
                rendered=rendered,
            )
        )

    return jobs


def shutdown_runtime(llm: LLM | None) -> None:
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", None)
            core_client = getattr(engine, "engine_core", None)
            shutdown = getattr(core_client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as e:
            print(f"[warn] LLM shutdown failed: {e}")

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main(args: Args) -> None:
    if args.detect_batch_size < 1:
        raise ValueError("detect_batch_size must be >= 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if args.fallback_start_page < 1:
        raise ValueError("fallback_start_page must be >= 1")

    targets_df, unique_targets_df = load_targets(args)
    scan_volume_rows = resolve_scan_volume_rows(args, targets_df)
    if not scan_volume_rows:
        raise RuntimeError("No AKT volumes selected for scanning.")

    if unique_targets_df.height == 0:
        print("[info] No published_texts-backed targets remain after filtering. Continuing with PDF-only detection.")

    jobs = prepare_volume_jobs(args, scan_volume_rows)
    if not jobs:
        raise RuntimeError("No AKT volumes selected for scanning.")

    output_root = Path(args.output_root)
    scan_volume_keys = {str(row["volume_key"]) for row in scan_volume_rows}
    llm: LLM | None = None
    detect_sampling: SamplingParams | None = None
    page_records_by_key: dict[tuple[str, int], list[dict]] = {}
    page_raw_payload_by_key: dict[tuple[str, int], dict] = {}
    output_paths: dict[str, Path] = {}
    selected_target_keys = {
        (str(row["volume_key"]), str(row["canonical_excavation_number"]))
        for row in unique_targets_df
        .select(["volume_key", "canonical_excavation_number"])
        .iter_rows(named=True)
    }
    try:
        if args.retry_missing_pdf_iterations < 0:
            raise ValueError("retry_missing_pdf_iterations must be >= 0")
        if args.pdf_only_detection_passes < 1:
            raise ValueError("pdf_only_detection_passes must be >= 1")

        target_volume_keys = (
            set(unique_targets_df.get_column("volume_key").to_list())
            if unique_targets_df.height
            else set()
        )
        pdf_only_volume_keys = scan_volume_keys.difference(target_volume_keys)
        pdf_only_jobs = [job for job in jobs if job.volume_key in pdf_only_volume_keys]

        def ensure_llm() -> tuple[LLM, SamplingParams]:
            nonlocal llm, detect_sampling
            if llm is not None and detect_sampling is not None:
                return llm, detect_sampling

            if torch.cuda.is_available() and args.tensor_parallel_size > torch.cuda.device_count():
                raise ValueError(
                    f"tensor_parallel_size={args.tensor_parallel_size} exceeds CUDA devices={torch.cuda.device_count()}"
                )

            quantization = (args.quantization or "").strip() or None
            if quantization is None and "fp8" in args.model.lower():
                quantization = "fp8"

            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                quantization=quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                enforce_eager=args.enforce_eager,
                compilation_config={"mode": "none", "cudagraph_mode": "none"} if args.enforce_eager else None,
                allowed_local_media_path=str(Path(args.output_root).resolve()),
                limit_mm_per_prompt={"image": 1},
            )

            detect_sampling = build_sampling_params(
                args=args,
                schema=DETECT_OUTPUT_JSON_SCHEMA,
                max_tokens=args.detect_max_tokens,
            )
            return llm, detect_sampling

        def upsert_page_detection(
            job: VolumeJob,
            page_num: int,
            image_path: Path,
            page_numbers_path: Path,
            page_raw_path: Path,
            page_numbers: list[dict],
            attempt_payload: dict,
        ) -> None:
            page_key = (job.volume_key, page_num)
            existing_records = page_records_by_key.get(page_key, [])
            new_records = build_page_number_records(job, page_num, image_path, page_numbers)
            merged_records = merge_page_number_records(existing_records, new_records)
            page_records_by_key[page_key] = merged_records

            existing_payload = page_raw_payload_by_key.get(page_key)
            merged_payload = merge_page_raw_payload(existing_payload, attempt_payload, merged_records)
            page_raw_payload_by_key[page_key] = merged_payload

            page_df = pl.DataFrame(merged_records) if merged_records else empty_page_numbers_df()
            page_df.write_csv(page_numbers_path)
            page_raw_path.write_text(
                json.dumps(merged_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        def apply_forced_page_locations() -> None:
            if not FORCED_PAGE_LOCATIONS:
                return

            jobs_by_volume = {job.volume_key: job for job in jobs}
            for canonical_excavation_number, forced_location in FORCED_PAGE_LOCATIONS.items():
                volume_key = normalize_volume_key(str(forced_location["volume_key"]))
                if (volume_key, canonical_excavation_number) not in selected_target_keys:
                    continue

                job = jobs_by_volume.get(volume_key)
                if job is None:
                    continue

                expected_pdf_name = normalize_ws(str(forced_location.get("expected_pdf_name", "")))
                if expected_pdf_name and job.pdf_path.name != expected_pdf_name:
                    print(
                        f"[warn] Skip forced location for {canonical_excavation_number}: "
                        f"expected pdf={expected_pdf_name}, resolved pdf={job.pdf_path.name}"
                    )
                    continue

                page_num = int(forced_location["page"])
                image_path = next(
                    (path for rendered_page, path in job.rendered if rendered_page == page_num),
                    None,
                )
                if image_path is None:
                    print(
                        f"[warn] Skip forced location for {canonical_excavation_number}: "
                        f"page {page_num} is not available in rendered pages for {job.pdf_path.name}"
                    )
                    continue

                detected_text = normalize_ws(
                    str(forced_location.get("detected_text", canonical_excavation_number))
                ) or canonical_excavation_number
                parsed = {"excavation_numbers": [detected_text]}
                raw_text = json.dumps(parsed, ensure_ascii=False)
                attempt_payload = {
                    "akt_volume": job.akt_volume,
                    "volume_key": job.volume_key,
                    "pdf_name": job.pdf_path.name,
                    "pdf_path": str(job.pdf_path),
                    "page": page_num,
                    "image_path": str(image_path),
                    "raw_text": raw_text,
                    "parsed": parsed,
                    "attempt_parsed": parsed,
                    "num_excavation_numbers": 1,
                    "attempt_num_excavation_numbers": 1,
                    "retry_iteration": -1,
                    "detect_cache_version": DETECT_CACHE_VERSION,
                    "detection_backend": "forced_location_exception",
                }
                upsert_page_detection(
                    job=job,
                    page_num=page_num,
                    image_path=image_path,
                    page_numbers_path=job.numbers_dir / f"page_{page_num:03d}_numbers.csv",
                    page_raw_path=job.numbers_dir / f"page_{page_num:03d}_raw.json",
                    page_numbers=[
                        {
                            "detected_text": detected_text,
                            "canonical_excavation_number": canonical_excavation_number,
                        }
                    ],
                    attempt_payload=attempt_payload,
                )

        def run_detection_pass(
            pending_detect: list[tuple[VolumeJob, int, Path, Path, Path]],
            retry_iteration: int,
        ) -> None:
            if not pending_detect:
                return

            akt1_pending: list[tuple[VolumeJob, int, Path, Path, Path]] = []
            llm_pending: list[tuple[VolumeJob, int, Path, Path, Path]] = []
            for item in pending_detect:
                if is_akt1_volume(item[0].volume_key):
                    akt1_pending.append(item)
                else:
                    llm_pending.append(item)

            fallback_akt1_pending: list[tuple[VolumeJob, int, Path, Path, Path]] = []
            for job, page_num, image_path, page_numbers_path, page_raw_path in akt1_pending:
                page_numbers = extract_akt1_heading_numbers_from_text_layer(
                    pdf_path=job.pdf_path,
                    page_num=page_num,
                )
                if page_numbers is None:
                    fallback_akt1_pending.append(
                        (job, page_num, image_path, page_numbers_path, page_raw_path)
                    )
                    continue

                parsed = {
                    "excavation_numbers": [row["detected_text"] for row in page_numbers]
                }
                raw_text = json.dumps(parsed, ensure_ascii=False)
                attempt_payload = {
                    "akt_volume": job.akt_volume,
                    "volume_key": job.volume_key,
                    "pdf_name": job.pdf_path.name,
                    "pdf_path": str(job.pdf_path),
                    "page": page_num,
                    "image_path": str(image_path),
                    "raw_text": raw_text,
                    "parsed": parsed,
                    "attempt_parsed": parsed,
                    "num_excavation_numbers": len(page_numbers),
                    "attempt_num_excavation_numbers": len(page_numbers),
                    "retry_iteration": retry_iteration,
                    "detect_cache_version": DETECT_CACHE_VERSION,
                    "detection_backend": "pdf_text_layer",
                }
                upsert_page_detection(
                    job=job,
                    page_num=page_num,
                    image_path=image_path,
                    page_numbers_path=page_numbers_path,
                    page_raw_path=page_raw_path,
                    page_numbers=page_numbers,
                    attempt_payload=attempt_payload,
                )

            llm_pending = fallback_akt1_pending + llm_pending
            if not llm_pending:
                return

            llm_instance, sampling = ensure_llm()
            desc = "detect_excavation_numbers"
            if retry_iteration > 0:
                desc = f"detect_excavation_numbers_retry_{retry_iteration}"

            for start in tqdm(
                range(0, len(llm_pending), args.detect_batch_size),
                desc=desc,
            ):
                end = min(start + args.detect_batch_size, len(llm_pending))
                batch = llm_pending[start:end]
                batch_messages = [
                    make_detect_messages(
                        volume_key=job.volume_key,
                        page_number=page_num,
                        image_path=image_path,
                    )
                    for job, page_num, image_path, _page_numbers_path, _page_raw_path in batch
                ]
                outputs = llm_instance.chat(batch_messages, sampling_params=sampling, use_tqdm=False)

                for (job, page_num, image_path, page_numbers_path, page_raw_path), out in zip(batch, outputs):
                    raw_text = out.outputs[0].text if out.outputs else ""
                    parsed = parse_first_json_object(extract_final_answer(raw_text))
                    page_numbers = normalize_detected_excavation_numbers(
                        parsed,
                        volume_key=job.volume_key,
                    )

                    attempt_payload = {
                        "akt_volume": job.akt_volume,
                        "volume_key": job.volume_key,
                        "pdf_name": job.pdf_path.name,
                        "pdf_path": str(job.pdf_path),
                        "page": page_num,
                        "image_path": str(image_path),
                        "raw_text": raw_text,
                        "parsed": parsed,
                        "attempt_parsed": parsed,
                        "num_excavation_numbers": len(page_numbers),
                        "attempt_num_excavation_numbers": len(page_numbers),
                        "retry_iteration": retry_iteration,
                        "detect_cache_version": DETECT_CACHE_VERSION,
                    }
                    upsert_page_detection(
                        job=job,
                        page_num=page_num,
                        image_path=image_path,
                        page_numbers_path=page_numbers_path,
                        page_raw_path=page_raw_path,
                        page_numbers=page_numbers,
                        attempt_payload=attempt_payload,
                    )

        pending_detect: list[tuple[VolumeJob, int, Path, Path, Path]] = []
        for job in jobs:
            for page_num, image_path in job.rendered:
                page_numbers_path = job.numbers_dir / f"page_{page_num:03d}_numbers.csv"
                page_raw_path = job.numbers_dir / f"page_{page_num:03d}_raw.json"

                if page_raw_path.exists():
                    try:
                        raw_payload = json.loads(page_raw_path.read_text(encoding="utf-8"))
                        if raw_payload.get("detect_cache_version") == DETECT_CACHE_VERSION:
                            page_numbers = extract_page_numbers_from_payload(raw_payload)
                            raw_payload["parsed"] = {
                                "excavation_numbers": [row["detected_text"] for row in page_numbers]
                            }
                            raw_payload["num_excavation_numbers"] = len(page_numbers)
                            page_records = build_page_number_records(job, page_num, image_path, page_numbers)
                            page_key = (job.volume_key, page_num)
                            page_records_by_key[page_key] = page_records
                            page_raw_payload_by_key[page_key] = raw_payload
                            page_df = pl.DataFrame(page_records) if page_records else empty_page_numbers_df()
                            page_df.write_csv(page_numbers_path)
                            continue
                        print(
                            f"[info] Detection cache is outdated for {job.pdf_path.name} page {page_num}. "
                            "Re-running inference."
                        )
                    except Exception as e:
                        print(
                            f"[warn] Failed to load cached detection for {job.pdf_path.name} "
                            f"page {page_num}: {e}. Re-running inference."
                        )

                pending_detect.append((job, page_num, image_path, page_numbers_path, page_raw_path))

        def recompute_and_persist_outputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            nonlocal output_paths

            numbers_df = build_numbers_df_from_page_map(page_records_by_key)
            (
                unique_locations_df,
                subset_summary_df,
                missing_targets_df,
                pdf_only_unique_locations_df,
                pdf_only_summary_df,
            ) = compute_output_tables(
                targets_df=targets_df,
                unique_targets_df=unique_targets_df,
                numbers_df=numbers_df,
                scan_volume_keys=scan_volume_keys,
            )
            output_paths = persist_current_outputs(
                output_root=output_root,
                jobs=jobs,
                page_raw_payload_by_key=page_raw_payload_by_key,
                targets_df=targets_df,
                numbers_df=numbers_df,
                unique_locations_df=unique_locations_df,
                subset_summary_df=subset_summary_df,
                missing_targets_df=missing_targets_df,
                pdf_only_unique_locations_df=pdf_only_unique_locations_df,
                pdf_only_summary_df=pdf_only_summary_df,
            )
            return (
                numbers_df,
                unique_locations_df,
                subset_summary_df,
                missing_targets_df,
                pdf_only_unique_locations_df,
                pdf_only_summary_df,
            )

        run_detection_pass(pending_detect, retry_iteration=0)
        apply_forced_page_locations()

        (
            numbers_df,
            unique_locations_df,
            subset_summary_df,
            missing_targets_df,
            pdf_only_unique_locations_df,
            pdf_only_summary_df,
        ) = recompute_and_persist_outputs()
        max_retry_iterations = max(
            args.retry_missing_pdf_iterations,
            args.pdf_only_detection_passes - 1,
        )
        print(
            f"[retry] iteration=0/{max_retry_iterations}, "
            f"missing_targets={missing_targets_df.height}, "
            f"pdf_only_unique_excavations={pdf_only_summary_df.height}"
        )

        for retry_iteration in range(1, max_retry_iterations + 1):
            missing_volume_keys = set(missing_targets_df.get_column("volume_key").to_list())
            retry_jobs_by_volume: dict[str, VolumeJob] = {
                job.volume_key: job
                for job in jobs
                if job.volume_key in missing_volume_keys
            }
            if retry_iteration < args.pdf_only_detection_passes:
                for job in pdf_only_jobs:
                    retry_jobs_by_volume[job.volume_key] = job

            retry_jobs = [
                retry_jobs_by_volume[volume_key]
                for volume_key in sorted(retry_jobs_by_volume)
            ]
            if not retry_jobs:
                break

            retry_pending_detect = [
                (
                    job,
                    page_num,
                    image_path,
                    job.numbers_dir / f"page_{page_num:03d}_numbers.csv",
                    job.numbers_dir / f"page_{page_num:03d}_raw.json",
                )
                for job in retry_jobs
                for page_num, image_path in job.rendered
            ]
            pdf_only_retry_jobs = [
                job for job in retry_jobs
                if job.volume_key in pdf_only_volume_keys
            ]
            print(
                f"[retry] iteration={retry_iteration}/{max_retry_iterations}, "
                f"missing_targets={missing_targets_df.height}, "
                f"retry_volumes={len(retry_jobs)}, "
                f"retry_pages={len(retry_pending_detect)}, "
                f"pdf_only_retry_volumes={len(pdf_only_retry_jobs)}"
            )
            run_detection_pass(retry_pending_detect, retry_iteration=retry_iteration)
            (
                numbers_df,
                unique_locations_df,
                subset_summary_df,
                missing_targets_df,
                pdf_only_unique_locations_df,
                pdf_only_summary_df,
            ) = recompute_and_persist_outputs()
            print(
                f"[retry] completed iteration={retry_iteration}/{max_retry_iterations}, "
                f"updated_missing_targets={missing_targets_df.height}, "
                f"pdf_only_unique_excavations={pdf_only_summary_df.height}"
            )

        matched_unique = unique_targets_df.height - missing_targets_df.height
        print(
            "done. "
            f"target_rows={targets_df.height}, "
            f"unique_targets={unique_targets_df.height}, "
            f"matched_unique_targets={matched_unique}, "
            f"missing_unique_targets={missing_targets_df.height}, "
            f"pdf_only_unique_excavations={pdf_only_summary_df.height}"
        )
        print(f"page_detections_csv={output_paths['page_detections_path']}")
        print(f"unique_locations_csv={output_paths['unique_locations_path']}")
        print(f"subset_summary_csv={output_paths['subset_summary_path']}")
        print(f"generic_summary_csv={output_paths['generic_summary_path']}")
        print(f"pdf_only_unique_locations_csv={output_paths['pdf_only_unique_locations_path']}")
        print(f"pdf_only_summary_csv={output_paths['pdf_only_summary_path']}")
        print(f"missing_targets_csv={output_paths['missing_targets_path']}")
    finally:
        shutdown_runtime(llm)


if __name__ == "__main__":
    main(tyro.cli(Args))
