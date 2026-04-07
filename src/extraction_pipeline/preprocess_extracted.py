from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass, field
from pathlib import Path
import sys
import unicodedata

import pandas as pd
import tyro

from extraction_pipeline.utils.preprocess import normalize_translation, normalize_transliteration

# ---------------------------------------------------------------------------
# Allowed-character lists (originally from constants.py)
# ---------------------------------------------------------------------------
transliteration_allowed = [
    "!",
    "+",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "<",
    ">",
    "A",
    "B",
    "D",
    "E",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "W",
    "Z",
    "_",
    "a",
    "b",
    "d",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "w",
    "z",
    "{",
    "}",
    "\u00bc",
    "\u00bd",
    "\u00c0",
    "\u00c1",
    "\u00c8",
    "\u00c9",
    "\u00cc",
    "\u00cd",
    "\u00d9",
    "\u00da",
    "\u00e0",
    "\u00e1",
    "\u00e8",
    "\u00e9",
    "\u00ec",
    "\u00ed",
    "\u00f9",
    "\u00fa",
    "\u0130",
    "\u0131",
    "\u015f",
    "\u0160",
    "\u0161",
    "\u1e62",
    "\u1e63",
    "\u1e6c",
    "\u1e6d",
    "\u2026",
    "\u2153",
    "\u2154",
    "\u2159",
    "\u215a",
]

translation_allowed = [
    "'",
    "?",
    "e",
    "E",
    "a",
    "A",
    "i",
    "I",
    "t",
    "T",
    "n",
    "N",
    "s",
    "S",
    "o",
    "O",
    "r",
    "R",
    "l",
    "L",
    "h",
    "H",
    "u",
    "U",
    "m",
    "M",
    "d",
    "D",
    "F",
    "f",
    "\u0161",
    "\u0160",
    "-",
    "p",
    "P",
    "w",
    "W",
    "b",
    "B",
    "g",
    "G",
    "y",
    "Y",
    ".",
    "K",
    "k",
    ",",
    "C",
    "c",
    "v",
    "\u0101",
    "1",
    ")",
    "(",
    "<",
    ">",
    "z",
    "Z",
    "_",
    "Q",
    "q",
    "2",
    "\u012b",
    "\u1e6d",
    "\u1e6c",
    "0",
    ":",
    "3",
    "\u00bd",
    "5",
    ";",
    "x",
    "\u0113",
    "4",
    "\u016b",
    "6",
    "\u1e63",
    "\u1e62",
    "\u2153",
    "8",
    "\u2019",
    "!",
    "7",
    "j",
    "J",
    "\u2154",
    "\u201c",
    "\u201d",
    "9",
    "\u2013",
    "\u215a",
    "\u00bc",
    "\u2159",
    '"',
    "\u2018",
    "\u0131",
    "\u2014",
    "[",
    "]",
    "\u011f",
    "\u00e2",
    "+",
    "\u00e0",
    "\u015f",
]


DEFAULT_INPUT_CSVS = (
    Path(
        "/kaggle/data/extract_excavation_translation_pairs_from_locations/"
        "pdf_only_excavation_numbers_with_locations_fixed/translations_by_record.csv"
    ),
    Path(
        "/kaggle/data/extract_excavation_translation_pairs_from_locations/"
        "published_texts_akt_subset_with_locations_merged_fixed/translations_by_record.csv"
    ),
)
DEFAULT_OUTPUT_CSV = Path(
    "/kaggle/data/extract_excavation_translation_pairs_from_locations/"
    "translations_by_record_merged_processed.csv"
)
DEFAULT_TRAIN_PROCESSED_CSV = Path("/kaggle/data/train_processed.csv")
EXCLUDED_TRANSLATION_START_PREFIXES = (
    "Fragmentary envelope",
    "Error",
    "Envelope with seal",
    "Fragment of envelope",
)
EXCLUDED_EXCAVATION_NUMBERS = {
    "Kt. 86/k 226",
    "Kt. 86/k 194",
    "Kt. 86/k 53",
    "Kt. 86/k 46",
    "Kt. 86/k 42",
    "Kt. 86/k 41",
    "Kt. 86/k 36",
    "Kt. 86/k 35",
    "Kt. 94/k 1650",
    "Kt. 94/k 1280",
    "Kt. 94/k 608",
    "Kt. n/k 608",
    "Kt. 92/k 141",
    "Kt. 91/k 378",
    "Kt. 91/k 380",
    "Kt. 91/k 405",
}
OUTPUT_COLUMNS = [
    "input_row_index",
    "oare_id",
    "oare_id_in_train_processed",
    "volume_key",
    "canonical_excavation_number",
    "transliteration",
    "translation",
]


@dataclass
class Args:
    input_csvs: list[Path] = field(default_factory=lambda: list(DEFAULT_INPUT_CSVS))
    output_csv: Path = DEFAULT_OUTPUT_CSV
    train_processed_csv: Path | None = None
    translation_disallowed_counts_csv: Path | None = None
    transliteration_disallowed_counts_csv: Path | None = None
    column: str = "translation"
    transliteration_column: str = "transliteration"


@dataclass
class PreprocessStats:
    input_rows: int
    output_rows: int
    line_range_removed_rows: int
    changed_rows: int
    explicit_excavation_excluded_rows: int
    empty_translation_excluded_rows: int
    excluded_translation_prefix_rows: int
    unavailable_transliteration_excluded_rows: int
    no_text_transliteration_excluded_rows: int
    gap_only_transliteration_excluded_rows: int
    transliteration_changed_rows: int

# Superscript digits: 0-9
SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
# Subscript digits: 0-9
SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
# Minus-like symbols: -, hyphen/dash variants, and superscript minus.
MINUS_VARIANTS = "-\u2010\u2011\u2012\u2013\u2014\u207b"
# Prime-like symbols used in line markers: 1'), 3\u2032), etc.
PRIME_MARKS = "'\u2019\u2032"
DIGIT_CLASS = f"0-9{SUPERSCRIPT_DIGITS}"
NUM_TOKEN = rf"[{DIGIT_CLASS}]+(?:\s*[{PRIME_MARKS}]+)?"

LINE_MARKER_RE = re.compile(
    rf"""
    (?<![{DIGIT_CLASS}])  # Avoid starting in the middle of a number token
    \(?\s*               # Optional opening parenthesis
    (?:
        {NUM_TOKEN}\s*
        [{MINUS_VARIANTS}]\s*
        {NUM_TOKEN}        # Range marker: 1-3), 1'-2'), \u00b9\u207b\u00b2)
      |
        {NUM_TOKEN}        # Single marker: 1), 2), \u00b9), 1')
    )
    \s*\)               # Closing parenthesis is required
    """,
    flags=re.VERBOSE,
)
# Comma-separated range markers without per-item ')' e.g. "1-2\u300113-17" / "1-2, 13-17".
RANGE_LIST_NO_PAREN_RE = re.compile(
    rf"""
    (?<![{DIGIT_CLASS}])  # Avoid starting in the middle of a number token
    \(?\s*               # Optional opening parenthesis
    {NUM_TOKEN}\s*
    [{MINUS_VARIANTS}]\s*
    {NUM_TOKEN}
    (?:
        \s*[、,，]\s*
        {NUM_TOKEN}\s*
        [{MINUS_VARIANTS}]\s*
        {NUM_TOKEN}
    )+                   # At least one additional comma-separated range
    \s*\)?              # Optional closing parenthesis
    (?=$|[^{DIGIT_CLASS}])  # Marker boundary, including glued text like "1-3Foo"
    """,
    flags=re.VERBOSE,
)
# Single range marker without required trailing ')' e.g. "13'-16'" / "27\u207b29".
RANGE_NO_PAREN_RE = re.compile(
    rf"""
    (?<![{DIGIT_CLASS}])  # Avoid starting in the middle of a number token
    \(?\s*               # Optional opening parenthesis
    {NUM_TOKEN}\s*
    [{MINUS_VARIANTS}]\s*
    {NUM_TOKEN}
    \s*\)?              # Optional closing parenthesis
    (?=$|[^{DIGIT_CLASS}])  # Marker boundary, including glued text like "1-3Foo"
    """,
    flags=re.VERBOSE,
)
# Some editions use superscript range markers without a trailing ')', e.g. "\u00b9\u207b\u2074".
SUPERSCRIPT_RANGE_NO_PAREN_RE = re.compile(
    rf"""
    (?<![{DIGIT_CLASS}])   # Avoid starting in the middle of a number token
    [{SUPERSCRIPT_DIGITS}]+(?:\s*[{PRIME_MARKS}]+)?\s*
    [{MINUS_VARIANTS}]\s*
    [{SUPERSCRIPT_DIGITS}]+(?:\s*[{PRIME_MARKS}]+)?
    (?=$|[^{DIGIT_CLASS}]) # Marker boundary, including glued text like "\u00b9\u207b\u2074Foo"
    """,
    flags=re.VERBOSE,
)

ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)")
GAP_CLUSTER_RE = re.compile(r"(?:<gap>\s*){2,}")
WS_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
TRANSLATION_X_TOKEN_RE = re.compile(r"(?<![A-Za-z])x+(?![A-Za-z])")
TRANSLATION_NOISE_MARKER_RE = re.compile(
    r"""
    (?:
        (?:Ö|A)\s*\.?\s*y\.           # Ö y., Ö. y., A. y., A.y.
      |
        (?<!\()\bAy\b(?!\))\s*:?
      |
        \bK\.\s*10\.
    )
    """,
    flags=re.VERBOSE,
)
TRANSLITERATION_NOISE_MARKER_RE = re.compile(
    r"""
    (?ix)
    (?:
        a\s+envelope
      |
        upside\s+down(?:\s+blank\s+space)?
      |
        (?:rest\s+)?invisible
      |
        silindir\s+m[üu]h[üu]r\s+bask[ıi]s[ıi](?:\s+[A-Za-zÇĞİÖŞÜçğıöşü?]+)?
      |
        silindir\s+m[üu]h[üu]r(?:\s+tamam[ıi]\s+kirik)?
      |
        seal\s+[ABC]
      |
        seal\s+impression(?:\s+[A-Z?])?
      |
        (?<![A-Za-z0-9\]])
        (?:
            [oᵒ][bᵇ][vᵛ]
          |
            [uᵘ]\s*\.\s*[eᵉ]
          |
            [lˡ]\s*\.\s*[eᵉ]
          |
            [lˡ][eᵉ]\s*\.\s*[eᵉ]
          |
            [rʳ][iⁱ]\s*\.\s*[eᵉ]
          |
            [rʳ][eᵉ][vᵛ]
        )
        \s*\.?
        (?![A-Za-z0-9-])
    )
    """,
    flags=re.VERBOSE,
)
SUP_SUB_TAG_RE = re.compile(r"<\s*(sup|sub)\s*>(.*?)<\s*/\s*\1\s*>", flags=re.IGNORECASE)
LONE_SUP_SUB_TAG_RE = re.compile(r"<\s*/?\s*(?:sup|sub)\s*>", flags=re.IGNORECASE)
MIXED_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s+(\d+)\s*[/⁄]\s*(\d+)(?!\d)")
COMPACT_MIXED_FRACTION_RE = re.compile(r"(?<!\d)(\d+)([1-9])\s*[/⁄]\s*(\d+)(?!\d)")
SIMPLE_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s*[/⁄]\s*(\d+)(?!\d)")
TRANSLITERATION_GAP_X_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[''′]?[xX][''′]?(?![A-Za-z0-9])")
TRANSLITERATION_GAP_DOT_RUN_RE = re.compile(r"[.…]{3,}|…+")
TRANSLITERATION_FRACTION_DIGITS = f"0-9{SUPERSCRIPT_DIGITS}{SUBSCRIPT_DIGITS}"
TRANSLITERATION_MIXED_FRACTION_RE = re.compile(
    rf"""
    (?<![{TRANSLITERATION_FRACTION_DIGITS}/⁄])
    ([0-9]+)\s+
    ([{TRANSLITERATION_FRACTION_DIGITS}]+)\s*[/⁄]\s*
    ([{TRANSLITERATION_FRACTION_DIGITS}]+)
    (?![{TRANSLITERATION_FRACTION_DIGITS}/⁄])
    """,
    flags=re.VERBOSE,
)
TRANSLITERATION_SIMPLE_FRACTION_RE = re.compile(
    rf"""
    (?<![{TRANSLITERATION_FRACTION_DIGITS}/⁄])
    ([{TRANSLITERATION_FRACTION_DIGITS}]+)\s*[/⁄]\s*
    ([{TRANSLITERATION_FRACTION_DIGITS}]+)
    (?![{TRANSLITERATION_FRACTION_DIGITS}/⁄])
    """,
    flags=re.VERBOSE,
)
FRACTION_GLYPH_RE = re.compile(r"([¼½⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞¾⅒])")

TRANSLITERATION_PREPROCESS_AKT_VOLUME_KEYS = {"7B", "11B", "12"}
TRANSLITERATION_BRACKET_STRIP_TABLE = str.maketrans("", "", "[]<>?/⁄*'")

EXCEPTIONAL_TRANSLATION_START_MARKERS: dict[str, tuple[str, ...]] = {
    "Kt. n/k 610": ("Uşur-ša-Īstar, Šū-Laban ve",),
    "Kt. 88/k 223": ("Abba'nın mühürü,",),
    "Kt. 88/k 733": ("Nishatum-vergisi ilâve",),
    "Kt. 88/k 26": ("Aššur-danni'nin bana",),
    "Kt. 88/k 66": ("Sermayedar, Aššur-bni",),
    "Kt. 88/k 227": ("Puzur-Aššur'un mührü",),
    "Kt. c/k 1063": ("<gap> mina gümüşü",),
}

SUP_SUB_DIGIT_TO_ASCII = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)
FRACTION_TO_GLYPH: dict[tuple[int, int], str] = {
    (1, 2): "½",
    (1, 3): "⅓",
    (2, 3): "⅔",
    (1, 4): "¼",
    (3, 4): "¾",
    (1, 5): "⅕",
    (2, 5): "⅖",
    (3, 5): "⅗",
    (4, 5): "⅘",
    (1, 6): "⅙",
    (5, 6): "⅚",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (5, 8): "⅝",
    (7, 8): "⅞",
    (1, 10): "⅒",
}

SUPERSCRIPT_CHAR_MAP: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SUBSCRIPT_CHAR_MAP: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}


def remove_line_range_markers(text: str) -> str:
    cleaned = RANGE_LIST_NO_PAREN_RE.sub(" ", text or "")
    cleaned = RANGE_NO_PAREN_RE.sub(" ", cleaned)
    cleaned = LINE_MARKER_RE.sub(" ", cleaned)
    cleaned = SUPERSCRIPT_RANGE_NO_PAREN_RE.sub(" ", cleaned)
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return WS_RE.sub(" ", cleaned).strip()


def normalize_translation_x_to_gap(text: str) -> str:
    return TRANSLATION_X_TOKEN_RE.sub("<gap>", text or "")


def remove_translation_noise_markers(text: str) -> str:
    cleaned = TRANSLATION_NOISE_MARKER_RE.sub(" ", text or "")
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return WS_RE.sub(" ", cleaned).strip()


def remove_transliteration_noise_markers(text: str) -> str:
    cleaned = TRANSLITERATION_NOISE_MARKER_RE.sub(" ", text or "")
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return WS_RE.sub(" ", cleaned).strip()


def collapse_gap_tokens(text: str) -> str:
    text = re.sub(r"\s*<gap>\s*", " <gap> ", text or "")
    text = re.sub(r"<gap>\s*-\s*", "<gap>-", text)
    text = re.sub(r"\s*-\s*<gap>", "-<gap>", text)

    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"<gap>(?:\s*-\s*<gap>)+", "<gap>", text)
        text = GAP_CLUSTER_RE.sub("<gap> ", text)
        text = re.sub(r"<gap>\s*-\s*", "<gap>-", text)
        text = re.sub(r"\s*-\s*<gap>", "-<gap>", text)

    return WS_RE.sub(" ", text).strip()


def normalize_gap_tokens(text: str) -> str:
    text = ELLIPSIS_RE.sub(" <gap> ", text or "")
    return collapse_gap_tokens(text)


def remove_square_brackets(text: str) -> str:
    return WS_RE.sub(" ", (text or "").replace("[", "").replace("]", "")).strip()


def remove_translation_superscript_like_chars(text: str) -> str:
    text = text or ""
    for ch in ("ᵖ", "ˡ", "ʳ", "ᵘ", "ʼ"):
        text = text.replace(ch, "")
    return text


def convert_sup_sub_html_to_unicode(text: str) -> str:
    text = text or ""

    def to_super(content: str) -> str:
        return "".join(
            SUPERSCRIPT_CHAR_MAP.get(ch, SUPERSCRIPT_CHAR_MAP.get(ch.lower(), ch))
            for ch in content
        )

    def to_sub(content: str) -> str:
        return "".join(
            SUBSCRIPT_CHAR_MAP.get(ch, SUBSCRIPT_CHAR_MAP.get(ch.lower(), ch))
            for ch in content
        )

    def repl(match: re.Match[str]) -> str:
        tag = match.group(1).lower()
        content = match.group(2)
        if tag == "sup":
            return to_super(content)
        return to_sub(content)

    prev = None
    while prev != text:
        prev = text
        text = SUP_SUB_TAG_RE.sub(repl, text)

    return LONE_SUP_SUB_TAG_RE.sub("", text)


def normalize_common_fractions(text: str) -> str:
    text = (text or "").translate(SUP_SUB_DIGIT_TO_ASCII)

    def replace_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole} {glyph}"

    def replace_compact_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole} {glyph}"

    def replace_simple(match: re.Match[str]) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return glyph

    text = MIXED_FRACTION_RE.sub(replace_mixed, text)
    text = COMPACT_MIXED_FRACTION_RE.sub(replace_compact_mixed, text)
    text = SIMPLE_FRACTION_RE.sub(replace_simple, text)
    return text


def parse_fraction_number(value: str) -> int | None:
    normalized = (value or "").translate(SUP_SUB_DIGIT_TO_ASCII)
    if not normalized.isdigit():
        return None
    return int(normalized)


def normalize_transliteration_fraction_glyphs(text: str) -> str:
    text = text or ""

    def replace_flexible_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = parse_fraction_number(match.group(2))
        denominator = parse_fraction_number(match.group(3))
        if numerator is None or denominator is None:
            return match.group(0)
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole} {glyph}"

    def replace_flexible_simple(match: re.Match[str]) -> str:
        numerator = parse_fraction_number(match.group(1))
        denominator = parse_fraction_number(match.group(2))
        if numerator is None or denominator is None:
            return match.group(0)
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return glyph

    def replace_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole} {glyph}"

    def replace_compact_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole} {glyph}"

    def replace_simple(match: re.Match[str]) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return glyph

    text = TRANSLITERATION_MIXED_FRACTION_RE.sub(replace_flexible_mixed, text)
    text = TRANSLITERATION_SIMPLE_FRACTION_RE.sub(replace_flexible_simple, text)
    text = MIXED_FRACTION_RE.sub(replace_mixed, text)
    text = COMPACT_MIXED_FRACTION_RE.sub(replace_compact_mixed, text)
    text = SIMPLE_FRACTION_RE.sub(replace_simple, text)
    return text


def normalize_transliteration_gap_candidates(text: str) -> str:
    text = TRANSLITERATION_GAP_X_TOKEN_RE.sub(" <gap> ", text or "")
    text = TRANSLITERATION_GAP_DOT_RUN_RE.sub(" <gap> ", text)
    return collapse_gap_tokens(text)


def is_superscript_char(char: str) -> bool:
    name = unicodedata.name(char, "")
    return "SUPERSCRIPT" in name or name.startswith(
        ("MODIFIER LETTER SMALL", "MODIFIER LETTER CAPITAL")
    )


def normalize_akt_transliteration_superscripts(text: str) -> str:
    text = (text or "").replace("ᵈ", "{d}")
    text = re.sub(r"ᵏ[ⁱᶦ]", "{ki}", text)
    return "".join(char for char in text if not is_superscript_char(char))


def ensure_spaces_around_fraction_glyphs(text: str) -> str:
    return WS_RE.sub(" ", FRACTION_GLYPH_RE.sub(r" \1 ", text or "")).strip()


def normalize_translation_slash_and_punctuation(text: str) -> str:
    text = (text or "").replace("*", "")
    text = re.sub(r"\s*[/⁄]\s*", " / ", text)
    text = WS_RE.sub(" ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def normalize_ascii_apostrophes(text: str) -> str:
    return (text or "").replace("\u2018", "'").replace("\u2019", "'")


def normalize_ascii_double_quotes(text: str) -> str:
    return (text or "").replace("\u201c", '"').replace("\u201d", '"')


def is_unavailable_text(text: str) -> bool:
    return (text or "").strip().lower() == "unavailable"


def is_empty_text(text: str) -> bool:
    return not (text or "").strip()


def contains_no_text_marker(text: str) -> bool:
    return "no text" in (text or "").strip().lower()


def starts_with_excluded_translation_prefix(text: str) -> str | None:
    stripped = (text or "").lstrip()
    lowered = stripped.casefold()
    for prefix in EXCLUDED_TRANSLATION_START_PREFIXES:
        if lowered.startswith(prefix.casefold()):
            return prefix
    return None


def is_gap_only_text(text: str) -> bool:
    return bool(re.fullmatch(r"(?:\s*<gap>\s*)+", text or ""))


def should_preprocess_transliteration(volume_key: str) -> bool:
    normalized_volume_key = re.sub(r"\s+", " ", (volume_key or "").strip()).upper()
    if not normalized_volume_key:
        return False
    if normalized_volume_key.startswith("LARSEN"):
        return True
    akt_volume_key = normalized_volume_key.removeprefix("AKT ").strip()
    return akt_volume_key in TRANSLITERATION_PREPROCESS_AKT_VOLUME_KEYS


def preprocess_transliteration_before_normalization(volume_key: str, text: str) -> str:
    current = (
        normalize_ascii_apostrophes(text)
        .replace("`", "")
        .replace("\u00ab", "")
        .replace("\u00bb", "")
        .replace("\u02da", "")
        .replace("\u02c8", "")
        .replace("\u02f3", "")
        .replace("\u02d2", "")
        .replace("\u02bb", "")
        .replace("\u02c0", "")
        .replace("\u02c1", "")
        .replace("\u02fa", "")
        .replace("\u02f9", "")
    )
    current = remove_transliteration_noise_markers(current)
    if should_preprocess_transliteration(volume_key):
        current = convert_sup_sub_html_to_unicode(current)
        current = normalize_transliteration_fraction_glyphs(current)
        current = normalize_akt_transliteration_superscripts(current)
        current = current.translate(TRANSLITERATION_BRACKET_STRIP_TABLE)
        return normalize_transliteration_gap_candidates(current)
    return current


def marker_variants(marker: str) -> tuple[str, ...]:
    variants = [marker]
    straight = marker.replace("\u2019", "'")
    curly = marker.replace("'", "\u2019")
    for variant in (straight, curly):
        if variant not in variants:
            variants.append(variant)
    return tuple(variants)


def apply_exceptional_translation_overrides(
    excavation_number: str,
    text: str,
) -> str:
    normalized_text = text or ""
    markers = EXCEPTIONAL_TRANSLATION_START_MARKERS.get(excavation_number or "", ())
    if not markers:
        return normalized_text

    for marker in markers:
        for variant in marker_variants(marker):
            idx = normalized_text.find(variant)
            if idx >= 0:
                return normalized_text[idx:].strip()
    return normalized_text


def preprocess_translation_text(text: str) -> str:
    # Extraction-specific pre-processing (HTML, line markers, noise)
    text = remove_line_range_markers(text or "")
    text = remove_translation_noise_markers(text)
    text = convert_sup_sub_html_to_unicode(text)
    text = remove_square_brackets(text)
    text = normalize_common_fractions(text)
    # Universal normalization (handles all remaining steps)
    text = normalize_translation(text)
    return remove_translation_noise_markers(text)


def resolve_excavation_numbers(df: pd.DataFrame, column: str) -> list[str]:
    if column != "translation":
        return [""] * len(df)

    candidate_columns = (
        "excavation_number",
        "canonical_excavation_number",
        "excavation_no",
        "response_excavation_number",
        "translation_request_excavation_number",
    )
    for excavation_column in candidate_columns:
        if excavation_column in df.columns:
            return df[excavation_column].tolist()
    return [""] * len(df)


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    input_csv: Path,
    column: str,
    transliteration_column: str,
) -> tuple[pd.DataFrame, PreprocessStats, str, str]:
    input_columns = df.columns.tolist()
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {input_csv}. Available columns: {list(df.columns)}"
        )
    if transliteration_column not in df.columns:
        raise ValueError(
            "Column "
            f"'{transliteration_column}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    excavation_numbers = resolve_excavation_numbers(df, column)
    source_values = df[column].tolist()
    translation_volume_keys = df["volume_key"].tolist() if "volume_key" in df.columns else [""] * len(df)
    normalized_values: list[str] = []
    keep_mask: list[bool] = []
    explicit_excavation_excluded_rows = 0
    empty_translation_excluded_rows = 0
    excluded_translation_prefix_rows = 0
    line_range_removed_rows = 0
    changed_rows = 0

    for excavation_number, volume_key, value in zip(excavation_numbers, translation_volume_keys, source_values):
        original = value or ""
        if excavation_number in EXCLUDED_EXCAVATION_NUMBERS:
            keep_mask.append(False)
            normalized_values.append("")
            explicit_excavation_excluded_rows += 1
            print(
                "Excluded row due to explicit excavation-number exclusion: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"translation={original!r}",
                file=sys.stderr,
            )
            continue
        no_line_ranges = remove_line_range_markers(original)
        if no_line_ranges != original:
            line_range_removed_rows += 1
        normalized = preprocess_translation_text(original)
        if is_empty_text(original) or is_empty_text(normalized):
            keep_mask.append(False)
            normalized_values.append("")
            empty_translation_excluded_rows += 1
            print(
                "Excluded row due to empty translation: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"translation={original!r}",
                file=sys.stderr,
            )
            continue
        excluded_prefix = starts_with_excluded_translation_prefix(normalized)
        if excluded_prefix is not None:
            keep_mask.append(False)
            normalized_values.append("")
            excluded_translation_prefix_rows += 1
            print(
                "Excluded row due to translation start prefix: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"matched_prefix={excluded_prefix!r}, "
                f"translation={original!r}",
                file=sys.stderr,
            )
            continue
        normalized = apply_exceptional_translation_overrides(excavation_number, normalized)

        keep_mask.append(True)
        if normalized != original:
            changed_rows += 1
        normalized_values.append(normalized)

    if explicit_excavation_excluded_rows or empty_translation_excluded_rows or excluded_translation_prefix_rows:
        df = df.loc[keep_mask].copy()
        source_values = [value for value, keep in zip(source_values, keep_mask) if keep]
        normalized_values = [value for value, keep in zip(normalized_values, keep_mask) if keep]

    orig_column = "translation_orig" if column == "translation" else f"{column}_orig"
    df[orig_column] = source_values
    df[column] = normalized_values

    transliteration_volume_keys = df["volume_key"].tolist() if "volume_key" in df.columns else [""] * len(df)
    transliteration_source_values = df[transliteration_column].tolist()
    transliteration_excavation_numbers = resolve_excavation_numbers(df, column)
    transliteration_normalized_values: list[str] = []
    transliteration_keep_mask: list[bool] = []
    unavailable_transliteration_excluded_rows = 0
    no_text_transliteration_excluded_rows = 0
    gap_only_transliteration_excluded_rows = 0
    transliteration_changed_rows = 0
    for excavation_number, volume_key, value in zip(
        transliteration_excavation_numbers,
        transliteration_volume_keys,
        transliteration_source_values,
    ):
        original = value or ""
        preprocessed = preprocess_transliteration_before_normalization(volume_key, original)
        normalized = remove_transliteration_noise_markers(normalize_transliteration(preprocessed))
        if should_preprocess_transliteration(volume_key):
            normalized = collapse_gap_tokens(normalized)
        normalized = ensure_spaces_around_fraction_glyphs(normalized)
        if is_unavailable_text(original) or is_unavailable_text(normalized):
            transliteration_keep_mask.append(False)
            unavailable_transliteration_excluded_rows += 1
            print(
                "Excluded row due to unavailable transliteration: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"transliteration={original!r}",
                file=sys.stderr,
            )
            continue
        if contains_no_text_marker(original) or contains_no_text_marker(normalized):
            transliteration_keep_mask.append(False)
            no_text_transliteration_excluded_rows += 1
            print(
                "Excluded row due to 'no text' transliteration marker: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"transliteration={original!r}",
                file=sys.stderr,
            )
            continue
        if is_gap_only_text(normalized):
            transliteration_keep_mask.append(False)
            gap_only_transliteration_excluded_rows += 1
            print(
                "Excluded row due to gap-only transliteration: "
                f"excavation_number={excavation_number}, "
                f"volume_key={volume_key}, "
                f"transliteration={original!r}, "
                f"normalized_transliteration={normalized!r}",
                file=sys.stderr,
            )
            continue
        transliteration_keep_mask.append(True)
        if normalized != original:
            transliteration_changed_rows += 1
        transliteration_normalized_values.append(normalized)

    if (
        unavailable_transliteration_excluded_rows
        or no_text_transliteration_excluded_rows
        or gap_only_transliteration_excluded_rows
    ):
        df = df.loc[transliteration_keep_mask].copy()
        transliteration_source_values = [
            value for value, keep in zip(transliteration_source_values, transliteration_keep_mask) if keep
        ]

    transliteration_orig_column = (
        "transliteration_orig"
        if transliteration_column == "transliteration"
        else f"{transliteration_column}_orig"
    )
    df[transliteration_orig_column] = transliteration_source_values
    df[transliteration_column] = transliteration_normalized_values

    extra_columns = [col for col in df.columns if col not in input_columns]
    df = df[input_columns + extra_columns]

    stats = PreprocessStats(
        input_rows=len(excavation_numbers),
        output_rows=len(df),
        line_range_removed_rows=line_range_removed_rows,
        changed_rows=changed_rows,
        explicit_excavation_excluded_rows=explicit_excavation_excluded_rows,
        empty_translation_excluded_rows=empty_translation_excluded_rows,
        excluded_translation_prefix_rows=excluded_translation_prefix_rows,
        unavailable_transliteration_excluded_rows=unavailable_transliteration_excluded_rows,
        no_text_transliteration_excluded_rows=no_text_transliteration_excluded_rows,
        gap_only_transliteration_excluded_rows=gap_only_transliteration_excluded_rows,
        transliteration_changed_rows=transliteration_changed_rows,
    )
    return df, stats, orig_column, transliteration_orig_column


def concat_processed_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_columns: list[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)
    normalized_frames = [frame.reindex(columns=all_columns, fill_value="") for frame in frames]
    return pd.concat(normalized_frames, ignore_index=True)


def load_train_processed_oare_ids(path: Path) -> set[str]:
    train_df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["oare_id"])
    return {value for value in train_df["oare_id"].tolist() if value}


def default_disallowed_counts_csv(output_csv: Path, column_name: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_{column_name}_disallowed_char_counts.csv")


def build_disallowed_char_counts_df(values: pd.Series, allowed_chars: list[str]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    allowed_set = set(allowed_chars)
    for value in values.fillna("").astype(str):
        for char in value:
            if char == " ":
                continue
            if char not in allowed_set:
                counter[char] += 1

    rows = [
        {
            "char": char,
            "count": count,
            "codepoint": f"U+{ord(char):04X}",
            "unicode_name": unicodedata.name(char, "<unknown>"),
        }
        for char, count in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=["char", "count", "codepoint", "unicode_name"])


def main(args: Args) -> None:
    if not args.input_csvs:
        raise ValueError("Specify at least one input CSV")

    processed_frames: list[pd.DataFrame] = []
    all_stats: list[tuple[Path, PreprocessStats]] = []
    orig_column_name = ""
    transliteration_orig_column_name = ""

    for input_csv in args.input_csvs:
        df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
        processed_df, stats, orig_column_name, transliteration_orig_column_name = preprocess_dataframe(
            df,
            input_csv=input_csv,
            column=args.column,
            transliteration_column=args.transliteration_column,
        )
        processed_frames.append(processed_df)
        all_stats.append((input_csv, stats))

    merged_df = concat_processed_frames(processed_frames)
    has_train_processed = (
        args.train_processed_csv is not None and args.train_processed_csv.exists()
    )
    if has_train_processed:
        train_processed_oare_ids = load_train_processed_oare_ids(args.train_processed_csv)
        merged_df["oare_id_in_train_processed"] = merged_df["oare_id"].fillna("").isin(
            train_processed_oare_ids
        )
    output_columns = [
        col for col in OUTPUT_COLUMNS
        if col != "oare_id_in_train_processed" or has_train_processed
    ]
    merged_df = merged_df.reindex(columns=output_columns, fill_value="")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.output_csv, index=False)
    matched_train_processed_rows = (
        int(merged_df["oare_id_in_train_processed"].sum()) if has_train_processed else 0
    )
    translation_disallowed_counts_csv = (
        args.translation_disallowed_counts_csv
        or default_disallowed_counts_csv(args.output_csv, "translation")
    )
    transliteration_disallowed_counts_csv = (
        args.transliteration_disallowed_counts_csv
        or default_disallowed_counts_csv(args.output_csv, "transliteration")
    )
    translation_disallowed_df = build_disallowed_char_counts_df(
        merged_df["translation"], translation_allowed
    )
    transliteration_disallowed_df = build_disallowed_char_counts_df(
        merged_df["transliteration"], transliteration_allowed
    )
    translation_disallowed_df.to_csv(translation_disallowed_counts_csv, index=False)
    transliteration_disallowed_df.to_csv(transliteration_disallowed_counts_csv, index=False)
    translation_disallowed_total = int(translation_disallowed_df["count"].sum())
    transliteration_disallowed_total = int(transliteration_disallowed_df["count"].sum())

    total_input_rows = sum(stats.input_rows for _, stats in all_stats)
    total_output_rows = sum(stats.output_rows for _, stats in all_stats)
    total_line_range_removed_rows = sum(stats.line_range_removed_rows for _, stats in all_stats)
    total_changed_rows = sum(stats.changed_rows for _, stats in all_stats)
    total_explicit_excavation_excluded_rows = sum(
        stats.explicit_excavation_excluded_rows for _, stats in all_stats
    )
    total_empty_translation_excluded_rows = sum(
        stats.empty_translation_excluded_rows for _, stats in all_stats
    )
    total_excluded_translation_prefix_rows = sum(
        stats.excluded_translation_prefix_rows for _, stats in all_stats
    )
    total_unavailable_transliteration_excluded_rows = sum(
        stats.unavailable_transliteration_excluded_rows for _, stats in all_stats
    )
    total_no_text_transliteration_excluded_rows = sum(
        stats.no_text_transliteration_excluded_rows for _, stats in all_stats
    )
    total_gap_only_transliteration_excluded_rows = sum(
        stats.gap_only_transliteration_excluded_rows for _, stats in all_stats
    )
    total_transliteration_changed_rows = sum(
        stats.transliteration_changed_rows for _, stats in all_stats
    )

    for input_csv, stats in all_stats:
        print(f"Processed: {input_csv}")
        print(f"  Input rows: {stats.input_rows}")
        print(f"  Output rows: {stats.output_rows}")
        print(f"  Rows with line range markers removed: {stats.line_range_removed_rows}")
        print(f"  Rows changed by preprocessing: {stats.changed_rows}")
        print(
            "  Rows excluded due to explicit excavation-number exclusion: "
            f"{stats.explicit_excavation_excluded_rows}"
        )
        print(f"  Rows excluded due to empty translation: {stats.empty_translation_excluded_rows}")
        print(
            "  Rows excluded due to translation start prefix: "
            f"{stats.excluded_translation_prefix_rows}"
        )
        print(
            "  Rows excluded due to unavailable transliteration: "
            f"{stats.unavailable_transliteration_excluded_rows}"
        )
        print(
            "  Rows excluded due to 'no text' transliteration marker: "
            f"{stats.no_text_transliteration_excluded_rows}"
        )
        print(
            "  Rows excluded due to gap-only transliteration: "
            f"{stats.gap_only_transliteration_excluded_rows}"
        )
        print(f"  Rows changed by transliteration preprocessing: {stats.transliteration_changed_rows}")

    print(f"Saved merged CSV: {args.output_csv}")
    print(f"Input files: {len(args.input_csvs)}")
    print(f"Total input rows: {total_input_rows}")
    print(f"Total output rows: {total_output_rows}")
    print(f"Updated from column: {args.column}")
    print(f"Wrote original text to column: {orig_column_name}")
    print(f"Wrote normalized text to column: {args.column}")
    print(f"Rows with line range markers removed: {total_line_range_removed_rows}")
    print(f"Rows changed by preprocessing: {total_changed_rows}")
    print(
        "Rows excluded due to explicit excavation-number exclusion: "
        f"{total_explicit_excavation_excluded_rows}"
    )
    print(f"Rows excluded due to empty translation: {total_empty_translation_excluded_rows}")
    print(
        "Rows excluded due to translation start prefix: "
        f"{total_excluded_translation_prefix_rows}"
    )
    print(
        "Rows excluded due to unavailable transliteration: "
        f"{total_unavailable_transliteration_excluded_rows}"
    )
    print(
        "Rows excluded due to 'no text' transliteration marker: "
        f"{total_no_text_transliteration_excluded_rows}"
    )
    print(
        "Rows excluded due to gap-only transliteration: "
        f"{total_gap_only_transliteration_excluded_rows}"
    )
    print(f"Updated from transliteration column: {args.transliteration_column}")
    print(f"Wrote original transliteration to column: {transliteration_orig_column_name}")
    print(f"Wrote normalized transliteration to column: {args.transliteration_column}")
    print(f"Rows changed by transliteration preprocessing: {total_transliteration_changed_rows}")
    if has_train_processed:
        print(f"Loaded train_processed CSV: {args.train_processed_csv}")
        print(f"Rows with oare_id found in train_processed: {matched_train_processed_rows}")
    else:
        print("Skipped train_processed CSV (not provided or not found)")
    print(f"Saved translation disallowed-char counts CSV: {translation_disallowed_counts_csv}")
    print(f"Translation disallowed-char total count (space excluded): {translation_disallowed_total}")
    print(
        "Saved transliteration disallowed-char counts CSV: "
        f"{transliteration_disallowed_counts_csv}"
    )
    print(
        "Transliteration disallowed-char total count (space excluded): "
        f"{transliteration_disallowed_total}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
