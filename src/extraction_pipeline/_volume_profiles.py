"""Volume profile configuration for the unified extraction pipeline.

Each volume (PDF) has a profile that controls detection, extraction, and
translation behaviour.  Profiles are defined as Python dataclasses (with
subclasses for volumes needing custom link-key expansion) and resolved
automatically from the PDF filename / volume key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from extraction_pipeline._excavation import (
    LARSEN_HEADING_RE,
    expand_match_keys,
)
from extraction_pipeline._json_utils import normalize_ws
from extraction_pipeline._prompts import (
    AKT1_DETECT_PROFILE,
    AKT2_TRANSLATION_PROFILE,
    AKT3_TRANSLATION_PROFILE,
    COMBINED_AKT3_EXTRACTION_PROFILE,
    COMBINED_EXTRACTION_PROFILE,
    COMBINED_SIDE_BY_SIDE_EXTRACTION_PROFILE,
    DEFAULT_TRANSLATION_PROFILE,
    DEFAULT_TRANSLITERATION_PROFILE,
    ICK4_DETECT_PROFILE,
    ICK4_EXTRACT_PROFILE,
    LARSEN_DETECT_PROFILE,
    SIDE_BY_SIDE_TRANSLATION_PROFILE,
    SIDE_BY_SIDE_TRANSLITERATION_PROFILE,
    STANDARD_DETECT_PROFILE,
    PromptProfile,
)


# ---------------------------------------------------------------------------
# Volume profile dataclass (flat — no nested configs)
# ---------------------------------------------------------------------------


@dataclass
class VolumeProfile:
    """Full configuration for processing a single PDF volume."""

    name: str = ""
    volume_key: str = ""
    pdf_filename: str = ""
    page_start: int = 1
    page_end: int = -1
    dpi: int = 200
    context_next_pages: int = 2

    # Detection (Phase 1)
    detect_profile_name: str = "standard_detect_v1"
    detect_json_key: str = "excavation_numbers"
    canonicalize_fn: str = (
        "standard"  # standard|akt1_heading|larsen_heading|ick4_heading
    )
    use_text_layer: bool = False
    heading_examples: str = ""
    detect_num_samples: int = 1  # >1 enables self-consistency (majority vote)

    # Extraction (Phase 3)
    extract_profile_name: str = "combined_extraction_v1"
    num_samples: int = 1

    # Translation (Phase 4)
    source_language: str = "Turkish"  # Turkish|German|English
    few_shot_count: int = 5
    skip_translation: bool = False

    def expand_link_keys(self, canonical: str) -> list[str]:
        """Generate match keys for linking canonical detection to published_texts.

        Override in subclasses for volumes with non-standard matching.
        """
        return expand_match_keys(canonical)


# ---------------------------------------------------------------------------
# Subclasses with custom link-key logic
# ---------------------------------------------------------------------------


@dataclass
class AKT1Profile(VolumeProfile):
    """AKT 1: heading numbers like '1' need prefix 'AKT 1' → 'AKT 1 1'."""

    def expand_link_keys(self, canonical: str) -> list[str]:
        keys = super().expand_link_keys(canonical)
        prefixed = f"AKT 1 {canonical}"
        if prefixed not in keys:
            keys.append(prefixed)
        return keys


_PAREN_RE = re.compile(r"\(([^)]+)\)")
_TRAILING_LETTER_RE = re.compile(r"^(.+?)\s+([A-Za-z])$")


@dataclass
class LarsenProfile(VolumeProfile):
    """Larsen 2002: headings like '1. CCT 5, 6a' → extract label 'CCT 5 6a'."""

    def expand_link_keys(self, canonical: str) -> list[str]:
        keys = super().expand_link_keys(canonical)
        m = LARSEN_HEADING_RE.match(canonical)
        if m:
            label = normalize_ws(m.group("label"))
            if label and label not in keys:
                keys.append(label)
            # Strip commas: "CCT 5, 6a" → "CCT 5 6a"
            label_no_comma = normalize_ws(label.replace(",", " "))
            if label_no_comma not in keys:
                keys.append(label_no_comma)
            # Collapse trailing letter: "RC 1749 B" → "RC 1749B" / "RC 1749b"
            for lbl in (label, label_no_comma):
                tm = _TRAILING_LETTER_RE.match(lbl)
                if tm:
                    collapsed = f"{tm.group(1)}{tm.group(2).lower()}"
                    if collapsed not in keys:
                        keys.append(collapsed)
            # "C33" → "Nesr. C33" (Larsen uses bare C-numbers for Nesr. collection)
            for lbl in (label, label_no_comma):
                if re.match(r"^C\d", lbl):
                    nesr = f"Nesr. {lbl}"
                    if nesr not in keys:
                        keys.append(nesr)
            # Extract parenthetical museum numbers: "Pa. 5 (L29-558)" → "L29-558"
            paren_m = _PAREN_RE.search(label)
            if paren_m:
                paren_val = normalize_ws(paren_m.group(1))
                if paren_val and paren_val not in keys:
                    keys.append(paren_val)
                # Also try label without parens: "Pa. 5"
                label_no_paren = normalize_ws(_PAREN_RE.sub("", label))
                if label_no_paren and label_no_paren not in keys:
                    keys.append(label_no_paren)
        return keys


# ---------------------------------------------------------------------------
# Prompt profile registry
# ---------------------------------------------------------------------------

_DETECT_PROFILES: dict[str, PromptProfile] = {
    STANDARD_DETECT_PROFILE.name: STANDARD_DETECT_PROFILE,
    AKT1_DETECT_PROFILE.name: AKT1_DETECT_PROFILE,
    LARSEN_DETECT_PROFILE.name: LARSEN_DETECT_PROFILE,
    ICK4_DETECT_PROFILE.name: ICK4_DETECT_PROFILE,
}

_EXTRACTION_PROFILES: dict[str, PromptProfile] = {
    COMBINED_EXTRACTION_PROFILE.name: COMBINED_EXTRACTION_PROFILE,
    COMBINED_SIDE_BY_SIDE_EXTRACTION_PROFILE.name: COMBINED_SIDE_BY_SIDE_EXTRACTION_PROFILE,
    COMBINED_AKT3_EXTRACTION_PROFILE.name: COMBINED_AKT3_EXTRACTION_PROFILE,
    ICK4_EXTRACT_PROFILE.name: ICK4_EXTRACT_PROFILE,
    DEFAULT_TRANSLITERATION_PROFILE.name: DEFAULT_TRANSLITERATION_PROFILE,
    SIDE_BY_SIDE_TRANSLITERATION_PROFILE.name: SIDE_BY_SIDE_TRANSLITERATION_PROFILE,
    DEFAULT_TRANSLATION_PROFILE.name: DEFAULT_TRANSLATION_PROFILE,
    AKT2_TRANSLATION_PROFILE.name: AKT2_TRANSLATION_PROFILE,
    AKT3_TRANSLATION_PROFILE.name: AKT3_TRANSLATION_PROFILE,
    SIDE_BY_SIDE_TRANSLATION_PROFILE.name: SIDE_BY_SIDE_TRANSLATION_PROFILE,
}


def get_detect_prompt_profile(name: str) -> PromptProfile:
    """Resolve a detection prompt profile by name."""
    if name not in _DETECT_PROFILES:
        raise ValueError(
            f"Unknown detect profile {name!r}. Available: {sorted(_DETECT_PROFILES)}"
        )
    return _DETECT_PROFILES[name]


def get_extraction_prompt_profile(name: str) -> PromptProfile:
    """Resolve an extraction prompt profile by name."""
    if name not in _EXTRACTION_PROFILES:
        raise ValueError(
            f"Unknown extraction profile {name!r}. "
            f"Available: {sorted(_EXTRACTION_PROFILES)}"
        )
    return _EXTRACTION_PROFILES[name]


# ---------------------------------------------------------------------------
# Built-in profile registry
# ---------------------------------------------------------------------------

_NORMALIZE_KEY_RE = re.compile(r"[^A-Z0-9]+")


def _normalize_key(text: str) -> str:
    """Normalize a filename or volume key for lookup."""
    return _NORMALIZE_KEY_RE.sub("", text.upper())


_BUILTIN_PROFILES: dict[str, VolumeProfile] = {}


def _register(profile: VolumeProfile) -> None:
    _BUILTIN_PROFILES[_normalize_key(profile.pdf_filename)] = profile
    _BUILTIN_PROFILES[_normalize_key(profile.volume_key)] = profile
    if profile.name != profile.volume_key:
        _BUILTIN_PROFILES[_normalize_key(profile.name)] = profile


# --- AKT 1 (Turkish, sequential, AKT1-heading detection) ---
_register(
    AKT1Profile(
        name="AKT 1",
        volume_key="AKT 1",
        pdf_filename="AKT_1_1990_fixed.pdf",
        page_start=34,
        page_end=120,
        detect_profile_name="akt1_detect_v1",
        canonicalize_fn="akt1_heading",
        use_text_layer=True,
        heading_examples='"No. 1", "No. 27"',
    )
)

# --- AKT 2 (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 2",
        volume_key="AKT 2",
        pdf_filename="AKT_2_1995_fixed.pdf",
        page_start=21,
        page_end=119,
        heading_examples='"Kt n/k 34", "Kt n/k 1648"',
    )
)

# --- AKT 3 (German, sequential) ---
_register(
    VolumeProfile(
        name="AKT 3",
        volume_key="AKT 3",
        pdf_filename="AKT_3_1995_fixed.pdf",
        page_start=15,
        page_end=193,
        heading_examples='"Kt a/k 423a", "Kt a/k 554b"',
        extract_profile_name="combined_akt3_extraction_v1",
        source_language="German",
    )
)

# --- AKT 4 (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 4",
        volume_key="AKT 4",
        pdf_filename="AKT_4_2006.pdf",
        page_start=37,
        page_end=159,
        heading_examples='"Kt 87/k 253", "Kt 87/k 450"',
    )
)

# --- AKT 5 (English, side-by-side) ---
_register(
    VolumeProfile(
        name="AKT 5",
        volume_key="AKT 5",
        pdf_filename="AKT_5_2008.pdf",
        page_start=46,
        page_end=168,
        heading_examples='"Kt 93/k 58", "Kt 93/k 168"',
        extract_profile_name="combined_side_by_side_extraction_v1",
        source_language="English",
        skip_translation=True,
    )
)

# --- AKT 6A–6E (English, side-by-side) ---
for _sub, _page_range in [
    ("A", (52, 436)),
    ("B", (59, 353)),
    ("C", (46, 297)),
    ("D", (26, 147)),
    ("E", (99, 281)),
]:
    _register(
        VolumeProfile(
            name=f"AKT 6{_sub}",
            volume_key=f"AKT 6{_sub}",
            pdf_filename=f"AKT_6{_sub}.pdf",
            page_start=_page_range[0],
            page_end=_page_range[1],
            heading_examples='"Kt 94/k 100", "Kt 94/k 943"',
            extract_profile_name="combined_side_by_side_extraction_v1",
            source_language="English",
            skip_translation=True,
        )
    )

# --- AKT 7A, 7B (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 7A",
        volume_key="AKT 7A",
        pdf_filename="AKT_7A.pdf",
        page_start=101,
        page_end=579,
        heading_examples='"Kt 91/k 104", "Kt 91/k 471"',
    )
)
_register(
    VolumeProfile(
        name="AKT 7B",
        volume_key="AKT 7B",
        pdf_filename="AKT_7B.pdf",
        page_start=102,
        page_end=420,
        heading_examples='"Kt 91/k 350", "Kt 91/k 550"',
    )
)

# --- AKT 8 (English, side-by-side) ---
_register(
    VolumeProfile(
        name="AKT 8",
        volume_key="AKT 8",
        pdf_filename="AKT_8_2015.pdf",
        page_start=49,
        page_end=515,
        heading_examples='"Kt 92/k 239", "Kt 92/k 1050"',
        extract_profile_name="combined_side_by_side_extraction_v1",
        source_language="English",
        skip_translation=True,
    )
)

# --- AKT 9A (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 9A",
        volume_key="AKT 9A",
        pdf_filename="AKT_9A.pdf",
        page_start=47,
        page_end=237,
        heading_examples='"Kt 88/k 110", "Kt 88/k 583"',
    )
)

# --- AKT 10 (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 10",
        volume_key="AKT 10",
        pdf_filename="AKT_10.pdf",
        page_start=28,
        page_end=154,
        heading_examples='"Kt 89/k 289", "Kt 89/k 370"',
    )
)

# --- AKT 11A, 11B (Turkish, sequential) ---
_register(
    VolumeProfile(
        name="AKT 11A",
        volume_key="AKT 11A",
        pdf_filename="AKT_11A.pdf",
        page_start=82,
        page_end=261,
        heading_examples='"Kt 90/k 100", "Kt 90/k 337b"',
    )
)
_register(
    VolumeProfile(
        name="AKT 11B",
        volume_key="AKT 11B",
        pdf_filename="AKT_11B.pdf",
        page_start=37,
        page_end=180,
        heading_examples='"Kt 00/k 17", "Kt 00/k 80"',
    )
)

# --- AKT 12 (English, side-by-side) ---
_register(
    VolumeProfile(
        name="AKT 12",
        volume_key="AKT 12",
        pdf_filename="AKT_12.pdf",
        page_start=43,
        page_end=174,
        heading_examples='"Kt 08/k 10", "Kt 08/k 200"',
        extract_profile_name="combined_side_by_side_extraction_v1",
        source_language="English",
        skip_translation=True,
    )
)

# --- Larsen 2002 (English, side-by-side, Larsen heading detection) ---
_register(
    LarsenProfile(
        name="Larsen 2002",
        volume_key="LARSEN 2002 - THE ASSUR-NADA ARCHIVE. PIHANS 96 2002",
        pdf_filename="Larsen_2002_-_The_Assur-nada_archive._PIHANS_96_2002.pdf",
        page_start=51,
        page_end=282,
        detect_profile_name="larsen_detect_v1",
        canonicalize_fn="larsen_heading",
        heading_examples='"1. CCT 5, 6a", "4. C 33", "6. TC 3, 95"',
        extract_profile_name="combined_side_by_side_extraction_v1",
        source_language="English",
        skip_translation=True,
    )
)

# --- ICK 4 (German, ICK4-style detection + extraction) ---
_register(
    VolumeProfile(
        name="ICK 4",
        volume_key="ICK 4",
        pdf_filename="ICK 4.pdf",
        page_start=24,
        page_end=406,
        detect_profile_name="ick4_detect_v1",
        detect_json_key="headings",
        canonicalize_fn="ick4_heading",
        heading_examples='"I 427", "I 552", "I 637(b)"',
        extract_profile_name="ick4_extract_v1",
        source_language="German",
    )
)


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


def resolve_profile(
    pdf_path: Path,
    override: str | None = None,
) -> VolumeProfile:
    """Resolve a VolumeProfile for the given PDF.

    Priority:
    1. ``override`` — built-in profile key
    2. Built-in profile registry (by PDF filename or stem)
    3. Default profile
    """
    if override:
        key = _normalize_key(override)
        if key in _BUILTIN_PROFILES:
            return _BUILTIN_PROFILES[key]
        raise ValueError(
            f"Unknown profile {override!r}. "
            f"Available: {sorted(set(p.name for p in _BUILTIN_PROFILES.values()))}"
        )

    # By PDF filename
    pdf_key = _normalize_key(pdf_path.name)
    if pdf_key in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[pdf_key]

    # By stem only
    stem_key = _normalize_key(pdf_path.stem)
    if stem_key in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[stem_key]

    # Default
    return VolumeProfile(
        name=pdf_path.stem,
        volume_key=pdf_path.stem,
        pdf_filename=pdf_path.name,
    )
