"""Unified 4-phase extraction pipeline for Akkadian scholarly PDFs.

Replaces the previous multi-script pipeline (find_excavation_number_pages.py,
extract_translation_pairs.py, extract_transliteration_pairs.py, extract_ick4.py,
extract_pairs.py) with a single entry point driven by YAML-based volume profiles.

Phases:
    1. detect   — Detect excavation numbers / text headings per page
    2. link     — Link detections to published_texts (get transliteration_orig)
    3. extract  — Extract transliteration + translation (multi-sample, ChrF++ consensus)
    4. translate — Translate to English (skip for English volumes)

Usage:
    python -m extraction_pipeline.extract_unified --pdf-path input/pdfs/AKT_12.pdf
    python -m extraction_pipeline.extract_unified --pdf-path ... --profile ick4
    python -m extraction_pipeline.extract_unified --all-pdfs --pdf-root input/pdfs
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import tyro
from tqdm import tqdm

from extraction_pipeline._consensus import select_best_candidate
from extraction_pipeline._excavation import (
    canonicalize_akt1_heading_number,
    canonicalize_ick4_heading,
    canonicalize_volume_identifier,
    expand_detected_excavation_texts,

    extract_akt1_heading_numbers_from_text_layer,
    normalize_detected_excavation_numbers,
)
from extraction_pipeline._json_utils import (
    extract_final_answer,
    normalize_ws,
    parse_first_json_object,
)
from extraction_pipeline._llm_client import LLMClient, build_visual_context
from extraction_pipeline._pdf_renderer import list_pdf_pages, render_pdf_pages_by_list
from extraction_pipeline._prompts import (
    COMBINED_PROFILE_NAMES,
    PromptProfile,
)
from extraction_pipeline._volume_profiles import (
    VolumeProfile,
    get_detect_prompt_profile,
    get_extraction_prompt_profile,
    resolve_profile,
)

# Re-use translate_to_english utilities
from extraction_pipeline.translate_to_english import (
    USER_PROMPT_FEW_SHOT_BLOCK_TEMPLATE,
    USER_PROMPT_TEMPLATE as TRANSLATE_EN_USER_PROMPT_TEMPLATE,
    build_similarity_retriever,
    extract_translation,
    format_few_shot_examples,
    has_valid_translation_json,
    load_few_shot_pool,
    normalize_gap_token_forms,
    normalize_non_gap_angle_tokens,
    normalize_translation_fractions,
    normalize_whitespace,
    retrieve_similar_few_shot,
    sample_few_shot_examples,
)

# Re-use preprocess_extracted utilities
from extraction_pipeline.preprocess_extracted import (
    TRANSLITERATION_BRACKET_STRIP_TABLE,
    collapse_gap_tokens,
    convert_sup_sub_html_to_unicode,
    ensure_spaces_around_fraction_glyphs,
    normalize_transliteration_fraction_glyphs,
    normalize_transliteration_gap_candidates,
    preprocess_translation_text,
    preprocess_transliteration_before_normalization,
    remove_transliteration_noise_markers,
)
from extraction_pipeline.utils.preprocess import normalize_transliteration as normalize_translit_text

# ---------------------------------------------------------------------------
# Regex patterns for text normalization (from extract_translation_pairs.py)
# ---------------------------------------------------------------------------

INTRO_SUMMARY_CUE_RE = re.compile(
    r"(?i)\b(?:mektuptur|yazdığı|tarafından|maktubun sahibi|yayınlanmış|belgesidir)\b"
)
ENGLISH_DESCRIPTION_RE = re.compile(
    r"(?i)^(?:unopened\s+envelope|an?\s+(?:unopened|damaged|fragmentary)\s+)"
)
LINE_MARKER_RE = re.compile(r"\b\d{1,3}(?:\s*['′]?\s*[-)\.]|['′])")
TRANSLATION_START_CUE_RE = re.compile(
    r"(?i)\b(?:şöyle\s*\(söyler\)\s*:|buzutaya['']?ya\s+söyle!?|um-ma)\b"
)
TRANSLITERATION_SECTION_CUE_RE = re.compile(r"(?i)\b(?:öy|oy|ak|ay)\.\s*\d{1,3}\b")
TRANSLATION_LINE_RANGE_RE = re.compile(r"(?m)^\s*(\d{1,3}\s*-\s*\d{1,3}\s*\))")
# Inline line-range markers embedded in translation text, e.g. "3-6)", "17-21)"
INLINE_LINE_RANGE_RE = re.compile(r"(?<!\d)\d{1,3}\s*[-–]\s*\d{1,3}\s*\)\s*")

TRANSLATE_EN_SYSTEM_PROMPTS = {
    "Turkish": """You are an expert translator for Turkish translations of Akkadian texts.
Translate Turkish text into English under strict constraints.

Rules:
1) Keep proper nouns exactly as written (people, places, deities, institutions).
2) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
3) If Akkadian transliteration fragments are mixed into the Turkish text, omit those fragments from the English output.
4) Prefer faithful/literal translation over paraphrase.
5) Respond in json format only with this schema:
{"translation_en": string}
""",
    "German": """You are an expert translator for German translations of Akkadian texts.
Translate German text into English under strict constraints.

Rules:
1) Keep proper nouns exactly as written (people, places, deities, institutions).
2) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
3) If Akkadian transliteration fragments are mixed into the German text, omit those fragments from the English output.
4) Prefer faithful/literal translation over paraphrase.
5) Respond in json format only with this schema:
{"translation_en": string}
""",
}


# ---------------------------------------------------------------------------
# CLI Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """Unified extraction pipeline arguments."""

    # Input
    pdf_path: str = ""
    all_pdfs: bool = False
    pdf_root: str = "./input/pdfs"
    profile: str = ""  # Built-in profile key, or empty for auto-resolve
    published_texts_csv: str = "./input/deep-past-initiative-machine-translation/published_texts.csv"

    # Output
    output_root: str = "./data/extract_unified"

    # Phase skipping
    skip_detect: bool = False
    skip_extract: bool = False
    skip_translate: bool = False
    retry_empty: bool = False
    extract_max_retries: int = 1  # max validation-based retries per record (0=disable)

    # LLM settings
    model: str = "openrouter/qwen/qwen3.5-plus-02-15"
    translate_model: str = "openrouter/openai/gpt-5.4-mini"
    max_concurrency: int = 32
    temperature: float = 0.3
    top_p: float = 0.95
    detect_max_tokens: int = 512
    extract_max_tokens: int = 16384
    translate_max_tokens: int = 4096
    use_pdf: bool = False
    render_workers: int = 4
    reasoning: bool = False  # Enable reasoning (extra_body for OpenRouter)

    # Few-shot for Phase 4
    few_shot_path: str = "./data/train_processed.csv"
    few_shot_count: int = 3
    few_shot_max_chars: int = 0  # 0 = no truncation
    few_shot_min_chars: int = 200  # exclude short pairs from pool
    seed: int = 42

    # Overrides (take precedence over profile)
    dpi: int = 200  # 0 = use profile default
    page_start: int = 0  # 0 = use profile default
    page_end: int = 0  # 0 = use profile default
    max_pages: int = 0  # 0 = unlimited, >0 = limit pages from page_start
    detect_num_samples: int = 0  # 0 = use profile default, >1 = self-consistency
    num_samples: int = 0  # 0 = use profile default
    context_next_pages: int = -1  # -1 = use profile default


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _expand_record_to_lines(row: dict, tr_col: str = "translation") -> list[dict]:
    """Expand a single record into line-level rows based on translation_ranges.

    Grouping follows translation_ranges keys (e.g. "1", "1-3").
    Transliteration lines within each range are joined with space.
    Falls back to flat (single row) if line-level data is unavailable.
    """
    tl_lines_raw = row.get("transliteration_lines", "") or ""
    tr_ranges_raw = row.get("translation_ranges", "") or ""

    try:
        tl_lines: dict[str, str] = json.loads(tl_lines_raw) if tl_lines_raw else {}
    except (json.JSONDecodeError, TypeError):
        tl_lines = {}
    try:
        tr_ranges: dict[str, str] = json.loads(tr_ranges_raw) if tr_ranges_raw else {}
    except (json.JSONDecodeError, TypeError):
        tr_ranges = {}

    exc_num = str(row.get("excavation_number", ""))
    page = row.get("page", "")

    # If no line-level data, fall back to flat record
    if not tr_ranges:
        tl_flat = str(row.get("transliteration", "") or "")
        tr_flat = str(row.get(tr_col, "") or "")
        if tl_flat.strip() or tr_flat.strip():
            return [{
                "excavation_number": exc_num,
                "page": page,
                "line_range": "",
                "transliteration": tl_flat,
                "translation": tr_flat,
            }]
        return []

    # Expand: iterate translation_ranges keys, gather matching transliteration lines
    sorted_tr_keys = sorted(tr_ranges.keys(), key=_parse_line_number_key)
    expanded: list[dict] = []
    for tr_key in sorted_tr_keys:
        tr_text = normalize_ws(str(tr_ranges[tr_key]))

        # Parse range to collect transliteration lines
        start, end = _parse_line_number_key(tr_key)
        if start >= 999999:
            continue

        tl_parts: list[str] = []
        for line_num in range(start, end + 1):
            line_key = str(line_num)
            if line_key in tl_lines:
                part = normalize_ws(str(tl_lines[line_key]))
                if part:
                    tl_parts.append(part)

        tl_text = " ".join(tl_parts)

        expanded.append({
            "excavation_number": exc_num,
            "page": page,
            "line_range": tr_key,
            "transliteration": tl_text,
            "translation": tr_text,
        })

    return expanded


def _parse_line_number_key(key: str) -> tuple[int, int]:
    """Parse a line key like '1', '1-3' into (start, end) for sorting."""
    key = key.strip()
    if "-" in key:
        parts = key.split("-", 1)
        try:
            return int(parts[0].strip()), int(parts[1].strip().rstrip(")"))
        except ValueError:
            return (999999, 999999)
    try:
        num = int(key.rstrip(")"))
        return (num, num)
    except ValueError:
        return (999999, 999999)


def parse_combined_extraction_response(
    parsed: dict, fallback_excavation_number: str
) -> tuple[str, str, str, dict[str, str], dict[str, str]]:
    """Parse combined extraction JSON.

    Returns:
        (excavation_number, transliteration_flat, translation_flat,
         transliteration_lines, translation_ranges)
    """
    excavation_number = normalize_ws(str(parsed.get("excavation_number", "")))
    if not excavation_number:
        excavation_number = fallback_excavation_number

    transliteration_lines: dict[str, str] = {}
    raw_translit = parsed.get("transliteration", {})
    if isinstance(raw_translit, dict) and raw_translit:
        sorted_lines = sorted(raw_translit.items(), key=lambda kv: _parse_line_number_key(kv[0]))
        transliteration_lines = {k: normalize_ws(str(v)) for k, v in sorted_lines}
        transliteration = " ".join(v for v in transliteration_lines.values() if v)
    elif isinstance(raw_translit, str):
        transliteration = normalize_ws(raw_translit)
    else:
        transliteration = ""

    translation_ranges: dict[str, str] = {}
    raw_translation = parsed.get("translation", {})
    if isinstance(raw_translation, dict) and raw_translation:
        sorted_groups = sorted(raw_translation.items(), key=lambda kv: _parse_line_number_key(kv[0]))
        translation_ranges = {k: normalize_ws(str(v)) for k, v in sorted_groups}
        translation = " ".join(v for v in translation_ranges.values() if v)
    elif isinstance(raw_translation, str):
        translation = normalize_ws(raw_translation)
    else:
        translation = ""

    translation = _strip_leading_summary(translation)
    if _is_likely_summary_only(translation):
        translation = ""

    return excavation_number, transliteration, translation, transliteration_lines, translation_ranges


def _has_summary_cue(text: str) -> bool:
    return bool(INTRO_SUMMARY_CUE_RE.search(text) or ENGLISH_DESCRIPTION_RE.search(text))


def _is_likely_summary_only(text: str) -> bool:
    normalized = normalize_ws(text)
    if not normalized:
        return False
    return bool(_has_summary_cue(normalized) and not LINE_MARKER_RE.search(normalized))


def _strip_leading_summary(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    range_match = TRANSLATION_LINE_RANGE_RE.search(raw)
    if range_match:
        marker_start = range_match.start(1)
        prefix = raw[:marker_start]
        if _has_summary_cue(prefix) or TRANSLITERATION_SECTION_CUE_RE.search(prefix):
            return normalize_ws(raw[marker_start:])

    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) == 2:
        head = normalize_ws(parts[0])
        if _has_summary_cue(head):
            return normalize_ws(parts[1])

    start_match = TRANSLATION_START_CUE_RE.search(raw)
    if start_match and _has_summary_cue(raw[: start_match.start()]):
        return normalize_ws(raw[start_match.start():])

    return normalize_ws(raw)



def build_image_context(
    rendered_pages: dict[int, Path | None],
    target_page: int,
    next_pages: int,
) -> list[tuple[str, int, Path | None]]:
    """Build image context list for target page + next context pages."""
    if target_page not in rendered_pages:
        return []
    context: list[tuple[str, int, Path | None]] = []
    context.append(("target", target_page, rendered_pages[target_page]))
    for offset in range(1, next_pages + 1):
        next_page = target_page + offset
        if next_page in rendered_pages:
            context.append(("next", next_page, rendered_pages[next_page]))
    return context


def _autocrop_vertical(img: "Image.Image", threshold: int = 250, min_margin: int = 10) -> "Image.Image":
    """Crop top/bottom white margins from an image.

    Scans rows from top and bottom to find the first row with
    non-white pixels (average < threshold), then crops with min_margin padding.
    """
    import numpy as np

    arr = np.asarray(img)
    if arr.ndim == 3:
        row_means = arr.mean(axis=(1, 2))
    else:
        row_means = arr.mean(axis=1)

    non_white = np.where(row_means < threshold)[0]
    if len(non_white) == 0:
        return img

    top = max(0, non_white[0] - min_margin)
    bottom = min(img.height, non_white[-1] + 1 + min_margin)
    return img.crop((0, top, img.width, bottom))


def stitch_pages_vertically(
    image_paths: list[Path],
    output_path: Path,
    pad: int = 20,
) -> Path | None:
    """Vertically concatenate page images into a single image.

    Each page is auto-cropped to remove top/bottom white margins before stitching.
    A small white padding is added at the top and bottom of the final image.
    Returns output_path on success, None if any input is missing.
    """
    from PIL import Image

    images: list[Image.Image] = []
    for p in image_paths:
        if p is None or not p.exists():
            return None
        images.append(_autocrop_vertical(Image.open(p)))

    if not images:
        return None

    total_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + pad * 2
    stitched = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    y_offset = pad
    for img in images:
        stitched.paste(img, (0, y_offset))
        y_offset += img.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, "PNG")
    for img in images:
        img.close()
    return output_path


# ---------------------------------------------------------------------------
# Phase 1: Detect excavation numbers / headings
# ---------------------------------------------------------------------------


def _majority_vote_detections(
    all_sample_detections: list[list[dict]],
    threshold: float = 0.5,
) -> list[dict]:
    """Select detections that appear in >= threshold fraction of samples (majority vote).

    Each sample is a list of detection dicts with 'excavation_number' and 'detected_text'.
    Returns deduplicated detections ordered by vote count (descending).
    """
    from collections import Counter

    n_samples = len(all_sample_detections)
    if n_samples <= 1:
        return all_sample_detections[0] if all_sample_detections else []

    # Count how many samples each canonical excavation_number appears in
    vote_counts: Counter[str] = Counter()
    # Keep first detected_text per excavation_number
    first_text: dict[str, str] = {}
    for sample in all_sample_detections:
        seen_in_sample: set[str] = set()
        for det in sample:
            key = det["excavation_number"]
            if key not in seen_in_sample:
                seen_in_sample.add(key)
                vote_counts[key] += 1
                if key not in first_text:
                    first_text[key] = det.get("detected_text", key)

    import math
    min_votes = max(1, math.ceil(n_samples * threshold))
    results: list[dict] = []
    for exc_num, count in vote_counts.most_common():
        if count >= min_votes:
            results.append({
                "excavation_number": exc_num,
                "detected_text": first_text.get(exc_num, exc_num),
            })

    return results


def phase1_detect(
    args: Args,
    profile: VolumeProfile,
    client: LLMClient,
    pdf_path: Path,
    output_dir: Path,
) -> pl.DataFrame:
    """Phase 1: Detect excavation numbers per page."""
    detections_csv = output_dir / "detections.csv"

    if args.skip_detect and detections_csv.exists():
        print(f"[detect] Skipped. Loading from {detections_csv}")
        return pl.read_csv(detections_csv)

    dpi = args.dpi if args.dpi > 0 else profile.dpi
    page_start = args.page_start if args.page_start > 0 else profile.page_start
    page_end = args.page_end if args.page_end > 0 else profile.page_end
    if args.max_pages > 0:
        page_end = min(page_end, page_start + args.max_pages - 1)

    pages = list_pdf_pages(pdf_path, page_start, page_end)
    print(f"[detect] Scanning {len(pages)} pages (p{page_start}–p{page_end})...")

    raw_dir = output_dir / "raw_detect"
    raw_dir.mkdir(parents=True, exist_ok=True)

    detect_profile = get_detect_prompt_profile(profile.detect_profile_name)
    json_key = profile.detect_json_key
    canonicalize_fn = profile.canonicalize_fn
    use_text_layer = profile.use_text_layer
    heading_examples_raw = profile.heading_examples
    heading_examples = (
        f"Expected heading format examples for this volume: {heading_examples_raw}"
        if heading_examples_raw else ""
    )

    # Pre-render images if needed
    rendered_map: dict[int, Path | None] = {}
    if not args.use_pdf:
        images_dir = output_dir / "images"
        rendered_map, _ = render_pdf_pages_by_list(
            pdf_path=pdf_path,
            images_dir=images_dir,
            pages=pages,
            dpi=dpi,
            render_workers=args.render_workers,
        )

    all_detections: list[dict] = []
    pending: list[tuple[int, Path]] = []

    for page_num in pages:
        cache_path = raw_dir / f"page_{page_num:03d}.json"

        # AKT1 text-layer fast path
        if use_text_layer:
            text_layer_results = extract_akt1_heading_numbers_from_text_layer(pdf_path, page_num)
            if text_layer_results:
                for row in text_layer_results:
                    all_detections.append({
                        "excavation_number": row["canonical_excavation_number"],
                        "detected_text": row["detected_text"],
                        "page": page_num,
                    })
                continue

        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                for item in payload.get("detections", []):
                    all_detections.append({
                        "excavation_number": str(item.get("excavation_number", "")),
                        "detected_text": str(item.get("detected_text", "")),
                        "page": page_num,
                    })
                continue
            except Exception:
                pass
        pending.append((page_num, cache_path))

    detect_n = args.detect_num_samples if args.detect_num_samples > 0 else profile.detect_num_samples
    if detect_n > 1:
        print(f"[detect] Self-consistency: {detect_n} samples per page (majority vote)")

    if pending:
        def _detect_one(page_num: int, cache_path: Path) -> tuple[int, Path, list[str]]:
            image_path = rendered_map.get(page_num)
            visual = build_visual_context(
                pdf_path=pdf_path,
                entries=[("", page_num, image_path)],
                use_pdf=args.use_pdf,
            )
            messages = [
                {"role": "system", "content": detect_profile.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": detect_profile.user_prompt_template.format(
                                page_number=page_num,
                                heading_examples=heading_examples,
                            ),
                        },
                        *visual,
                    ],
                },
            ]
            resp_list = client.chat(
                messages,
                temperature=args.temperature,
                max_tokens=args.detect_max_tokens,
                top_p=args.top_p,
                response_format={"type": "json_object"},
                n=detect_n,
            )
            return page_num, cache_path, resp_list

        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(pending))) as executor:
            futures = {
                executor.submit(_detect_one, page_num, cache_path): (page_num, cache_path)
                for page_num, cache_path in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="detect"):
                page_num, cache_path, resp_list = future.result()

                # Parse each sample
                all_sample_detections: list[list[dict]] = []
                raw_texts: list[str] = []
                for raw_text in resp_list:
                    raw_texts.append(raw_text)
                    parsed = parse_first_json_object(extract_final_answer(raw_text))
                    sample_dets = _normalize_detections(
                        parsed, json_key, canonicalize_fn, profile.volume_key
                    )
                    all_sample_detections.append(sample_dets)

                # Majority vote if multiple samples, otherwise use single result
                if len(all_sample_detections) > 1:
                    detections = _majority_vote_detections(all_sample_detections)
                else:
                    detections = all_sample_detections[0] if all_sample_detections else []

                payload = {
                    "page": page_num,
                    "raw_texts": raw_texts,
                    "num_samples": len(resp_list),
                    "detections": detections,
                }
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                for det in detections:
                    all_detections.append({
                        "excavation_number": det["excavation_number"],
                        "detected_text": det["detected_text"],
                        "page": page_num,
                    })

    # Deduplicate: keep first occurrence of each excavation_number
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in all_detections:
        key = str(row["excavation_number"]).lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)

    df = pl.DataFrame(deduped)
    if df.height == 0:
        df = pl.DataFrame(schema={
            "excavation_number": pl.String,
            "detected_text": pl.String,
            "page": pl.Int64,
        })

    df.write_csv(detections_csv)
    print(f"[detect] Found {df.height} excavation numbers. Saved to {detections_csv}")
    client.print_usage_summary(label="detect")
    client.reset_usage()
    return df


def _normalize_detections(
    parsed: dict,
    json_key: str,
    canonicalize_fn: str,
    volume_key: str,
) -> list[dict]:
    """Normalize raw LLM detection output into canonical forms."""
    raw_values = parsed.get(json_key, [])
    if not isinstance(raw_values, list):
        return []

    rows: list[dict] = []
    seen: set[str] = set()

    for value in raw_values:
        detected_text = normalize_ws(str(value))
        if not detected_text:
            continue

        if canonicalize_fn == "ick4_heading":
            canonical = canonicalize_ick4_heading(detected_text)
        elif canonicalize_fn == "akt1_heading":
            canonical = canonicalize_akt1_heading_number(detected_text)
        else:
            # standard + larsen_heading go through the volume-aware path
            canonical = canonicalize_volume_identifier(volume_key, detected_text)

        if not canonical:
            continue

        dedupe_key = canonical.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append({
            "excavation_number": canonical,
            "detected_text": detected_text,
        })
    return rows


# ---------------------------------------------------------------------------
# Phase 2: Link to published_texts
# ---------------------------------------------------------------------------


def phase2_link(
    profile: VolumeProfile,
    detections_df: pl.DataFrame,
    published_texts_csv: Path,
    output_dir: Path,
    train_csv: Path | None = None,
) -> pl.DataFrame:
    """Phase 2: Link detections to published_texts.csv to get transliteration_orig.

    If *train_csv* is provided, also look up known English translations
    (``translation_ref``) via ``oare_id`` joining published_texts → train.
    """
    linked_csv = output_dir / "linked.csv"

    if not published_texts_csv.exists():
        print(f"[link] published_texts.csv not found at {published_texts_csv}, skipping link phase")
        return detections_df.with_columns(
            pl.lit("").alias("transliteration_orig"),
            pl.lit("").alias("translation_ref"),
        )

    # Load published_texts
    pub_df = pl.read_csv(published_texts_csv, infer_schema_length=10000)
    translit_lookup: dict[str, str] = {}
    _exc_to_oare: dict[str, str] = {}  # excavation_no (casefolded) → oare_id

    # Build oare_id → translation from train_csv
    _oare_to_translation: dict[str, str] = {}
    if train_csv and train_csv.exists():
        train_df = pl.read_csv(train_csv, infer_schema_length=10000)
        for row in train_df.iter_rows(named=True):
            oid = str(row.get("oare_id", "") or "").strip()
            tr = str(row.get("translation", "") or "").strip()
            if oid and tr:
                _oare_to_translation[oid] = tr

    _PRAGUE_I_RE = re.compile(r"Prague (I \d+)", re.IGNORECASE)

    def _add_to_lookup(key: str, translit: str, oare_id: str = "") -> None:
        key = key.casefold()
        existing = translit_lookup.get(key, "")
        if not existing or len(translit) > len(existing):
            translit_lookup[key] = translit
        if oare_id and key not in _exc_to_oare:
            _exc_to_oare[key] = oare_id

    for row in pub_df.iter_rows(named=True):
        translit = normalize_ws(str(row.get("transliteration_orig", "") or ""))
        oare_id = str(row.get("oare_id", "") or "").strip()
        exc_no = normalize_ws(str(row.get("excavation_no", "") or ""))
        if exc_no and translit:
            _add_to_lookup(exc_no, translit, oare_id)
        elif exc_no and oare_id:
            # No transliteration but has oare_id — still index for translation lookup
            _exc_to_oare.setdefault(exc_no.casefold(), oare_id)
        # Index by each pipe-separated alias (e.g. "CCT 5 50f | BM 14377bis")
        aliases = normalize_ws(str(row.get("aliases", "") or ""))
        for alias in aliases.split("|"):
            alias = normalize_ws(alias)
            if alias:
                if translit:
                    _add_to_lookup(alias, translit, oare_id)
                elif oare_id:
                    _exc_to_oare.setdefault(alias.casefold(), oare_id)
        # Also extract structured keys from label
        label = normalize_ws(str(row.get("label", "") or ""))
        combined_label = f"{label} | {aliases}"
        # ICK 4: "Prague I 445" → "I 445"
        m = _PRAGUE_I_RE.search(combined_label)
        if m:
            if translit:
                _add_to_lookup(m.group(1), translit, oare_id)
            elif oare_id:
                _exc_to_oare.setdefault(m.group(1).casefold(), oare_id)

    # Link each detection
    records: list[dict] = []
    for row in detections_df.iter_rows(named=True):
        exc_num = str(row.get("excavation_number", ""))
        match_keys = profile.expand_link_keys(exc_num)
        translit_orig = ""
        translation_ref = ""
        for mk in match_keys:
            mk_lower = mk.casefold()
            translit_orig = translit_lookup.get(mk_lower, "")
            if translit_orig:
                # Also look up translation via oare_id
                oare_id = _exc_to_oare.get(mk_lower, "")
                if oare_id:
                    translation_ref = _oare_to_translation.get(oare_id, "")
                break

        record = dict(row)
        record["transliteration_orig"] = translit_orig
        record["translation_ref"] = translation_ref
        records.append(record)

    df = pl.DataFrame(records)
    df.write_csv(linked_csv)
    linked_count = df.filter(pl.col("transliteration_orig").str.strip_chars() != "").height
    ref_count = df.filter(pl.col("translation_ref").str.strip_chars() != "").height
    print(f"[link] Linked {linked_count}/{df.height} detections to published_texts. Saved to {linked_csv}")
    if ref_count:
        print(f"[link] Found reference translations for {ref_count}/{df.height} detections (from train_old.csv)")
    return df


# ---------------------------------------------------------------------------
# Phase 3: Extract transliteration + translation
# ---------------------------------------------------------------------------


def phase3_extract(
    args: Args,
    profile: VolumeProfile,
    client: LLMClient,
    pdf_path: Path,
    linked_df: pl.DataFrame,
    output_dir: Path,
) -> pl.DataFrame:
    """Phase 3: Extract transliteration + translation for each record."""
    pairs_raw_csv = output_dir / "pairs_raw.csv"

    if args.skip_extract and pairs_raw_csv.exists():
        print(f"[extract] Skipped. Loading from {pairs_raw_csv}")
        return pl.read_csv(pairs_raw_csv)

    dpi = args.dpi if args.dpi > 0 else profile.dpi
    num_samples = args.num_samples if args.num_samples > 0 else profile.num_samples
    context_next = args.context_next_pages if args.context_next_pages >= 0 else profile.context_next_pages

    extract_profile = get_extraction_prompt_profile(profile.extract_profile_name)
    is_combined = extract_profile.name in COMBINED_PROFILE_NAMES

    excavation_numbers = linked_df.get_column("excavation_number").cast(pl.String).fill_null("").to_list()
    pages = linked_df.get_column("page").cast(pl.Int64).to_list()

    translit_origs: list[str] = []
    if "transliteration_orig" in linked_df.columns:
        translit_origs = linked_df.get_column("transliteration_orig").cast(pl.String).fill_null("").to_list()
    else:
        translit_origs = [""] * len(excavation_numbers)

    translation_refs: list[str] = []
    if "translation_ref" in linked_df.columns:
        translation_refs = linked_df.get_column("translation_ref").cast(pl.String).fill_null("").to_list()
    else:
        translation_refs = [""] * len(excavation_numbers)

    total_pdf_pages = list_pdf_pages(pdf_path)[-1]

    # Pre-render images
    rendered_map: dict[int, Path | None] = {}
    if not args.use_pdf:
        all_needed: set[int] = set()
        for page in pages:
            for offset in range(0, context_next + 1):
                p = page + offset
                if p <= total_pdf_pages:
                    all_needed.add(p)
        images_dir = output_dir / "images"
        rendered_map, _ = render_pdf_pages_by_list(
            pdf_path=pdf_path,
            images_dir=images_dir,
            pages=sorted(all_needed),
            dpi=dpi,
            render_workers=args.render_workers,
        )

    raw_dir = output_dir / "raw_extract"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    pending: list[tuple[int, str, int, str, str, Path]] = []

    for idx, (exc_num, page) in enumerate(zip(excavation_numbers, pages)):
        cache_path = raw_dir / f"record_{idx:04d}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                tl = str(payload.get("transliteration", "")).strip()
                tr = str(payload.get("translation", "")).strip()
                if args.retry_empty and (not tl or not tr):
                    cache_path.unlink()
                else:
                    cached_tl_lines = payload.get("transliteration_lines", {}) or {}
                    cached_tr_ranges = payload.get("translation_ranges", {}) or {}
                    records.append({
                        "excavation_number": str(payload.get("excavation_number", exc_num)),
                        "page": int(payload.get("page", page)),
                        "transliteration": tl,
                        "translation": tr,
                        "transliteration_lines": json.dumps(cached_tl_lines, ensure_ascii=False),
                        "translation_ranges": json.dumps(cached_tr_ranges, ensure_ascii=False),
                    })
                    continue
            except Exception:
                pass
        translit_orig = translit_origs[idx] if idx < len(translit_origs) else ""
        trans_ref = translation_refs[idx] if idx < len(translation_refs) else ""
        pending.append((idx, str(exc_num), int(page), translit_orig, trans_ref, cache_path))

    # Pre-stitch multi-page images
    stitched_dir = output_dir / "images_stitched"
    if not args.use_pdf and context_next > 0:
        stitched_dir.mkdir(parents=True, exist_ok=True)

    def _build_visual(page: int) -> list[dict]:
        """Build visual context entries for a given page."""
        image_context = build_image_context(
            rendered_map if not args.use_pdf else {p: None for p in range(1, total_pdf_pages + 1)},
            page,
            context_next,
        )
        if not args.use_pdf and len(image_context) > 1:
            page_images = [img for _, _, img in image_context if img is not None]
            page_nums = [p for _, p, _ in image_context]
            stitched_name = f"p{'_'.join(str(p) for p in page_nums)}.png"
            stitched_path = stitched_dir / stitched_name
            if not stitched_path.exists():
                stitch_pages_vertically(page_images, stitched_path)
            if stitched_path.exists():
                label = f"Pages {page_nums[0]}-{page_nums[-1]} (stitched)"
                entries = [(label, page, stitched_path)]
            else:
                role_to_label = {"target": "TARGET page", "next": "NEXT context page"}
                entries = [(role_to_label[role], p, img) for role, p, img in image_context]
        else:
            role_to_label = {"target": "TARGET page", "next": "NEXT context page"}
            entries = [(role_to_label[role], p, img) for role, p, img in image_context]
        return build_visual_context(pdf_path=pdf_path, entries=entries, use_pdf=args.use_pdf)

    def _call_extract(
        exc_num: str, page: int, translit_orig: str, visual: list[dict],
        retry_reason: str = "", translation_ref: str = "",
    ) -> list[str]:
        """Send extraction request to LLM, optionally with retry context."""
        if is_combined:
            format_kwargs = {
                "excavation_number": exc_num,
                "page_number": page,
                "transliteration_orig": translit_orig or "(unavailable)",
            }
        else:
            format_kwargs = {
                "excavation_number": exc_num,
                "page_number": page,
            }

        user_text = extract_profile.user_prompt_template.format(**format_kwargs)
        # Append reference translation if available
        if translation_ref:
            user_text += (
                f"\n\nPublished English translation (from OARE database, for reference):\n"
                f"{translation_ref}"
            )
        if retry_reason:
            user_text += f"\n\nIMPORTANT — Your previous extraction had these issues:\n{retry_reason}"

        messages = [
            {"role": "system", "content": extract_profile.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *visual,
                ],
            },
        ]
        chat_kwargs: dict = {}
        if args.reasoning:
            chat_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        return client.chat(
            messages,
            temperature=args.temperature,
            max_tokens=args.extract_max_tokens,
            top_p=args.top_p,
            response_format={"type": "json_object"},
            n=num_samples,
            **chat_kwargs,
        )

    def _parse_candidates(resp_list: list[str], exc_num: str) -> list[dict]:
        """Parse LLM responses into candidate dicts."""
        candidates: list[dict] = []
        for ci, raw_text in enumerate(resp_list):
            parsed = parse_first_json_object(extract_final_answer(raw_text))
            if is_combined:
                _, tl, tr, tl_lines, tr_ranges = parse_combined_extraction_response(parsed, exc_num)
                candidates.append({
                    "candidate_index": ci,
                    "raw_text": raw_text,
                    "transliteration": tl,
                    "translation": tr,
                    "transliteration_lines": tl_lines,
                    "translation_ranges": tr_ranges,
                })
            else:
                tl = normalize_ws(str(parsed.get("transliteration", "")))
                tr = normalize_ws(str(parsed.get("translation", "")))
                candidates.append({
                    "candidate_index": ci,
                    "raw_text": raw_text,
                    "transliteration": tl,
                    "translation": tr,
                    "transliteration_lines": {},
                    "translation_ranges": {},
                })
        return candidates

    def _select_and_save(
        candidates: list[dict], exc_num: str, page: int, cache_path: Path,
    ) -> dict:
        """Select best candidate, save to cache, return record dict."""
        selected, meta = select_best_candidate(
            candidates, text_key="translation",
            fallback_fields={"transliteration": "", "translation": ""},
        )
        transliteration = str(selected.get("transliteration", ""))
        translation = str(selected.get("translation", ""))
        tl_lines = selected.get("transliteration_lines", {}) or {}
        tr_ranges = selected.get("translation_ranges", {}) or {}

        payload = {
            "excavation_number": exc_num,
            "page": page,
            "transliteration": transliteration,
            "translation": translation,
            "transliteration_lines": tl_lines,
            "translation_ranges": tr_ranges,
            **meta,
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "excavation_number": exc_num,
            "page": page,
            "transliteration": transliteration,
            "translation": translation,
            "transliteration_lines": json.dumps(tl_lines, ensure_ascii=False),
            "translation_ranges": json.dumps(tr_ranges, ensure_ascii=False),
        }

    # --- Initial extraction ---
    if pending:
        def _extract_one(
            idx: int, exc_num: str, page: int, translit_orig: str, trans_ref: str, cache_path: Path,
        ) -> tuple[int, str, int, str, Path, list[str]]:
            visual = _build_visual(page)
            resp_list = _call_extract(exc_num, page, translit_orig, visual, translation_ref=trans_ref)
            return idx, exc_num, page, translit_orig, cache_path, resp_list

        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(pending))) as executor:
            futures = {
                executor.submit(_extract_one, idx, exc_num, page, translit_orig, trans_ref, cache_path): None
                for idx, exc_num, page, translit_orig, trans_ref, cache_path in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="extract"):
                _idx, exc_num, page, translit_orig, cache_path, resp_list = future.result()
                candidates = _parse_candidates(resp_list, exc_num)
                record = _select_and_save(candidates, exc_num, page, cache_path)
                records.append(record)

    # --- Validation-based retry loop ---
    max_retries = args.extract_max_retries
    # Build exc_num → (exc_num, page, translit_orig, translation_ref, cache_path) for retry lookup
    pending_lookup: dict[str, tuple[str, int, str, str, Path]] = {}
    for idx, (en, pg) in enumerate(zip(excavation_numbers, pages)):
        en_str = str(en)
        to = translit_origs[idx] if idx < len(translit_origs) else ""
        tr = translation_refs[idx] if idx < len(translation_refs) else ""
        pending_lookup[en_str] = (en_str, int(pg), to, tr, raw_dir / f"record_{idx:04d}.json")

    for retry_round in range(max_retries):
        # Validate all records and find those needing retry
        to_retry: list[tuple[int, dict, str]] = []
        for rec_idx, rec in enumerate(records):
            tl_json = str(rec.get("transliteration_lines", "") or "")
            tr_json = str(rec.get("translation_ranges", "") or "")
            flags = validate_extraction(
                tl_lines_json=tl_json,
                tr_ranges_json=tr_json,
                transliteration=str(rec.get("transliteration", "") or ""),
                translation=str(rec.get("translation", "") or ""),
            )
            if flags:
                reason = _flags_to_retry_message(flags, tl_json, tr_json)
                to_retry.append((rec_idx, rec, reason))

        if not to_retry:
            break

        print(f"[extract] Retry round {retry_round + 1}: {len(to_retry)} records with retryable flags")

        def _retry_one(rec_idx: int, rec: dict, reason: str) -> tuple[int, dict]:
            exc_num = str(rec["excavation_number"])
            page = int(rec["page"])
            info = pending_lookup.get(exc_num)
            if not info:
                return rec_idx, rec
            _, _, translit_orig, trans_ref, cache_path = info
            visual = _build_visual(page)
            resp_list = _call_extract(exc_num, page, translit_orig, visual, retry_reason=reason, translation_ref=trans_ref)
            candidates = _parse_candidates(resp_list, exc_num)
            new_record = _select_and_save(candidates, exc_num, page, cache_path)
            return rec_idx, new_record

        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(to_retry))) as executor:
            futures = {
                executor.submit(_retry_one, rec_idx, rec, reason): None
                for rec_idx, rec, reason in to_retry
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"retry-{retry_round + 1}"):
                rec_idx, new_record = future.result()
                records[rec_idx] = new_record

        client.print_usage_summary(label=f"retry-{retry_round + 1}")
        client.reset_usage()

    # Run final validation per record, then expand to line-level rows
    expanded_rows: list[dict] = []
    flagged = 0
    for rec in records:
        flags = validate_extraction(
            tl_lines_json=str(rec.get("transliteration_lines", "") or ""),
            tr_ranges_json=str(rec.get("translation_ranges", "") or ""),
            transliteration=str(rec.get("transliteration", "") or ""),
            translation=str(rec.get("translation", "") or ""),
        )
        if flags:
            flagged += 1

        lines = _expand_record_to_lines(rec)
        if lines:
            for line in lines:
                line["validation_flags"] = flags
                expanded_rows.append(line)
        else:
            expanded_rows.append({
                "excavation_number": rec.get("excavation_number", ""),
                "page": rec.get("page", ""),
                "line_range": "",
                "transliteration": rec.get("transliteration", ""),
                "translation": rec.get("translation", ""),
                "validation_flags": flags,
            })

    if flagged > 0:
        print(f"[extract] Validation: {flagged}/{len(records)} records have flags")

    df = pl.DataFrame(expanded_rows)
    if df.height == 0:
        df = pl.DataFrame(schema={
            "excavation_number": pl.String,
            "page": pl.Int64,
            "line_range": pl.String,
            "transliteration": pl.String,
            "translation": pl.String,
            "validation_flags": pl.String,
        })

    df.write_csv(pairs_raw_csv)
    non_empty = df.filter(
        (pl.col("transliteration").str.strip_chars() != "")
        & (pl.col("translation").str.strip_chars() != "")
    ).height
    print(f"[extract] Extracted {df.height} line-level pairs from {len(records)} records ({non_empty} non-empty). Saved to {pairs_raw_csv}")
    client.print_usage_summary(label="extract")
    client.reset_usage()
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(
    tl_lines_json: str,
    tr_ranges_json: str,
    transliteration: str,
    translation: str,
) -> str:
    """Validate line-level extraction data and return pipe-separated flags.

    Flags (separated by ``|``):
        missing_translation  — transliteration present but no translation
        missing_transliteration — translation present but no transliteration
        missing_translation_range:N,N,... — transliteration lines with no
            corresponding translation range
    """
    flags: list[str] = []

    try:
        tl_lines: dict[str, str] = json.loads(tl_lines_json) if tl_lines_json else {}
    except (json.JSONDecodeError, TypeError):
        tl_lines = {}
    try:
        tr_ranges: dict[str, str] = json.loads(tr_ranges_json) if tr_ranges_json else {}
    except (json.JSONDecodeError, TypeError):
        tr_ranges = {}

    has_tl = bool(transliteration and transliteration.strip())
    has_tr = bool(translation and translation.strip())

    if has_tl and not has_tr:
        flags.append("missing_translation")
    if has_tr and not has_tl:
        flags.append("missing_transliteration")

    # Translation coverage: which transliteration lines lack translation?
    if tl_lines and tr_ranges:
        tl_line_nums: set[int] = set()
        for key in tl_lines:
            start, end = _parse_line_number_key(key)
            if start < 999999:
                tl_line_nums.update(range(start, end + 1))

        tr_covered_nums: set[int] = set()
        for key in tr_ranges:
            start, end = _parse_line_number_key(key)
            if start < 999999:
                tr_covered_nums.update(range(start, end + 1))

        if tl_line_nums:
            uncovered = tl_line_nums - tr_covered_nums
            if uncovered:
                missing_keys = ",".join(str(n) for n in sorted(uncovered))
                flags.append(f"missing_translation_range:{missing_keys}")

    return "|".join(flags)


def _flags_to_retry_message(flags: str, tl_lines_json: str, tr_ranges_json: str) -> str:
    """Convert validation flags into a natural-language message for LLM retry."""
    parts: list[str] = []

    try:
        tl_lines: dict[str, str] = json.loads(tl_lines_json) if tl_lines_json else {}
    except (json.JSONDecodeError, TypeError):
        tl_lines = {}
    try:
        tr_ranges: dict[str, str] = json.loads(tr_ranges_json) if tr_ranges_json else {}
    except (json.JSONDecodeError, TypeError):
        tr_ranges = {}

    tl_max = 0
    for key in tl_lines:
        _, end = _parse_line_number_key(key)
        if end < 999999:
            tl_max = max(tl_max, end)
    tr_max = 0
    for key in tr_ranges:
        _, end = _parse_line_number_key(key)
        if end < 999999:
            tr_max = max(tr_max, end)

    for flag in flags.split("|"):
        flag = flag.strip()
        key = flag.split(":")[0]
        if key == "missing_translation":
            parts.append(
                "You extracted transliteration but NO translation. "
                "The page contains a translation for this text — please extract it."
            )
        elif key == "missing_transliteration":
            parts.append(
                "You extracted a translation but NO transliteration lines. "
                "Please extract the transliteration as well."
            )
        elif key == "missing_translation_range":
            missing_lines = flag.split(":", 1)[1] if ":" in flag else ""
            parts.append(
                f"Your transliteration has {len(tl_lines)} lines (up to line {tl_max}), "
                f"but your translation only covers up to line {tr_max}. "
                f"Lines {missing_lines} have no translation. "
                "Please include translations for ALL lines — do not stop early."
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 4: Translate to English
# ---------------------------------------------------------------------------


def phase4_translate(
    args: Args,
    profile: VolumeProfile,
    client: LLMClient,
    pairs_df: pl.DataFrame,
    output_dir: Path,
) -> pl.DataFrame:
    """Phase 4: Translate non-English translations to English."""
    pairs_en_csv = output_dir / "pairs_en.csv"

    source_language = profile.source_language
    if profile.skip_translation:
        print(f"[translate] Skipped (source_language={source_language}, skip=True)")
        # For English volumes, translation column already contains English
        return pairs_df.with_columns(
            pl.col("translation").alias("translation_en")
        )

    if args.skip_translate and pairs_en_csv.exists():
        print(f"[translate] Skipped. Loading from {pairs_en_csv}")
        return pl.read_csv(pairs_en_csv)

    system_prompt = TRANSLATE_EN_SYSTEM_PROMPTS.get(source_language)
    if system_prompt is None:
        print(f"[translate] Unknown source_language={source_language}, skipping")
        return pairs_df.with_columns(pl.lit("").alias("translation_en"))

    few_shot_count = args.few_shot_count if args.few_shot_count > 0 else profile.few_shot_count

    transliterations = pairs_df.get_column("transliteration").cast(pl.String).fill_null("").to_list()
    translations_src = pairs_df.get_column("translation").cast(pl.String).fill_null("").to_list()
    exc_nums = pairs_df.get_column("excavation_number").cast(pl.String).fill_null("").to_list()
    translations_en: list[str] = [""] * len(exc_nums)

    translate_cache_dir = output_dir / "raw_translate"
    translate_cache_dir.mkdir(parents=True, exist_ok=True)

    # Few-shot setup (similarity-based retrieval per record)
    few_shot_pool: list[tuple[str, str]] = []
    retriever = None
    if few_shot_count > 0:
        few_shot_pool = load_few_shot_pool(
            csv_path=args.few_shot_path,
            transliteration_column="transliteration",
            translation_column="translation",
            max_chars=args.few_shot_max_chars,
            min_chars=args.few_shot_min_chars,
        )
        retriever = build_similarity_retriever(few_shot_pool)
        if retriever:
            print(f"[translate] Few-shot: similarity retriever built ({len(few_shot_pool)} pool)")
        elif few_shot_pool:
            print(f"[translate] Few-shot: falling back to random sampling ({len(few_shot_pool)} pool)")

    # Identify rows needing translation (skip empty transliteration or translation)
    to_translate: list[int] = []
    pending: list[int] = []
    for i, src_text in enumerate(translations_src):
        if not normalize_ws(src_text) or not normalize_ws(transliterations[i]):
            continue
        to_translate.append(i)
        cache_path = translate_cache_dir / f"record_{i:04d}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_en = str(payload.get("translation_en", "")).strip()
                if args.retry_empty and not cached_en:
                    cache_path.unlink()
                else:
                    translations_en[i] = cached_en
                    continue
            except Exception:
                pass
        pending.append(i)

    cached = len(to_translate) - len(pending)
    if cached > 0:
        print(f"[translate] Loaded {cached} from cache, {len(pending)} pending")

    if pending:
        def _translate_one(idx: int) -> tuple[int, str]:
            # Per-record similarity-based few-shot
            few_shot_block = ""
            if few_shot_count > 0 and transliterations[idx]:
                if retriever:
                    examples = retrieve_similar_few_shot(
                        retriever, transliterations[idx], few_shot_count
                    )
                else:
                    examples = sample_few_shot_examples(
                        few_shot_pool, few_shot_count, args.seed
                    )
                if examples:
                    few_shot_block = USER_PROMPT_FEW_SHOT_BLOCK_TEMPLATE.format(
                        few_shot_examples=format_few_shot_examples(examples)
                    )

            user_text = TRANSLATE_EN_USER_PROMPT_TEMPLATE.format(
                transliteration=transliterations[idx] or "(none)",
                few_shot_block=few_shot_block,
                source_language=source_language,
                source_text=translations_src[idx],
            )
            resp = client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=args.temperature,
                max_tokens=args.translate_max_tokens,
                top_p=args.top_p,
                response_format={"type": "json_object"},
            )
            return idx, resp[0] if resp else ""

        with ThreadPoolExecutor(max_workers=min(args.max_concurrency, len(pending))) as executor:
            futures = {executor.submit(_translate_one, idx): idx for idx in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="translate"):
                idx, raw = future.result()
                final_text = extract_final_answer(raw)
                parsed = parse_first_json_object(final_text)

                if has_valid_translation_json(parsed):
                    translated = extract_translation(parsed, final_text)
                else:
                    # Fallback: strip wrapping {"translation_en": "..."} if JSON parse failed
                    m = re.search(r'\{\s*"translation_en"\s*:\s*"', final_text)
                    if m:
                        translated = final_text[m.end():].rstrip().rstrip("}").rstrip().rstrip('"')
                    else:
                        translated = final_text.strip()

                translated = normalize_gap_token_forms(translated)
                translated = normalize_non_gap_angle_tokens(translated)
                translated = normalize_translation_fractions(translated)
                translated = normalize_whitespace(translated)

                translations_en[idx] = translated

                cache_path = translate_cache_dir / f"record_{idx:04d}.json"
                cache_path.write_text(
                    json.dumps({
                        "excavation_number": exc_nums[idx],
                        "translation_src": translations_src[idx],
                        "translation_en": translated,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    out_df = pairs_df.with_columns(pl.Series("translation_en", translations_en))
    out_df.write_csv(pairs_en_csv)

    non_empty_en = sum(1 for t in translations_en if normalize_ws(t))
    print(f"[translate] Translated {non_empty_en}/{len(to_translate)} texts. Saved to {pairs_en_csv}")
    client.print_usage_summary(label="translate")
    client.reset_usage()
    return out_df


# ---------------------------------------------------------------------------
# Final preprocessing
# ---------------------------------------------------------------------------


def postprocess(pairs_en_df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Normalize and produce final training CSV with line-level rows."""
    pairs_final_csv = output_dir / "pairs_final.csv"

    def _normalize_tl(text: str) -> str:
        preprocessed = preprocess_transliteration_before_normalization("", text)
        preprocessed = convert_sup_sub_html_to_unicode(preprocessed)
        preprocessed = normalize_transliteration_fraction_glyphs(preprocessed)
        preprocessed = preprocessed.translate(TRANSLITERATION_BRACKET_STRIP_TABLE)
        preprocessed = normalize_transliteration_gap_candidates(preprocessed)
        normalized = remove_transliteration_noise_markers(normalize_translit_text(preprocessed))
        normalized = collapse_gap_tokens(normalized)
        normalized = ensure_spaces_around_fraction_glyphs(normalized)
        return normalized

    def _normalize_tr(text: str) -> str:
        text = INLINE_LINE_RANGE_RE.sub("", text)
        return preprocess_translation_text(text)

    # Determine translation column — use translation_en if available
    tr_col = "translation_en" if "translation_en" in pairs_en_df.columns else "translation"

    # pairs_raw.csv is already line-level; just select the right translation column
    if tr_col != "translation" and tr_col in pairs_en_df.columns:
        df = pairs_en_df.with_columns(pl.col(tr_col).alias("translation"))
    else:
        df = pairs_en_df.clone()

    # Ensure line_range column exists (backward compat with old pairs_raw.csv)
    if "line_range" not in df.columns:
        df = df.with_columns(pl.lit("").alias("line_range"))

    if df.height == 0:
        final_df = pl.DataFrame(schema={
            "excavation_number": pl.String,
            "page": pl.Int64,
            "line_range": pl.String,
            "transliteration": pl.String,
            "translation": pl.String,
            "fold": pl.Int64,
        })
        final_df.write_csv(pairs_final_csv)
        print(f"[postprocess] Final: 0 pairs. Saved to {pairs_final_csv}")
        return final_df

    # Normalize
    df = df.with_columns(
        pl.col("transliteration")
        .map_elements(_normalize_tl, return_dtype=pl.String)
        .alias("transliteration_norm"),
        pl.col("translation")
        .map_elements(_normalize_tr, return_dtype=pl.String)
        .alias("translation_norm"),
    )

    df = df.filter(
        (pl.col("transliteration_norm").str.strip_chars() != "")
        & (pl.col("translation_norm").str.strip_chars() != "")
    )

    pre_filter = df.height
    placeholder_re = r"(?i)not worthwhile|is omitted|translation omitted|keine [Üü]bersetzung"
    df = df.filter(~pl.col("translation_norm").str.contains(placeholder_re))

    df = df.with_columns(
        pl.col("transliteration_norm").str.len_chars().alias("_tl_len"),
        pl.col("translation_norm").str.len_chars().alias("_tr_len"),
    ).with_columns(
        (pl.col("_tr_len") / pl.col("_tl_len").clip(1, None)).alias("_ratio"),
    )
    df = df.filter((pl.col("_ratio") >= 0.3) & (pl.col("_ratio") <= 4.0))
    filtered_out = pre_filter - df.height
    if filtered_out > 0:
        print(f"[postprocess] Filtered out {filtered_out} noisy pairs")

    select_cols = [pl.col("excavation_number")]
    if "page" in df.columns:
        select_cols.append(pl.col("page"))
    select_cols.extend([
        pl.col("line_range"),
        pl.col("transliteration_norm").alias("transliteration"),
        pl.col("translation_norm").alias("translation"),
        pl.lit(-1).alias("fold"),
    ])
    final_df = df.select(select_cols)

    final_df.write_csv(pairs_final_csv)
    print(f"[postprocess] Final: {final_df.height} line-level pairs. Saved to {pairs_final_csv}")
    return final_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: Args, pdf_path: Path) -> None:
    """Run the full 4-phase pipeline for a single PDF."""
    profile = resolve_profile(pdf_path, args.profile or None)

    output_dir = Path(args.output_root) / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved profile as YAML (output artifact for reproducibility)
    import dataclasses
    import yaml
    profile_yaml = output_dir / "profile.yaml"
    profile_data = {
        k: v for k, v in dataclasses.asdict(profile).items()
        if not k.startswith("_")
    }
    profile_data["profile_class"] = type(profile).__name__
    profile_yaml.write_text(
        yaml.dump(profile_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Per-PDF cost/usage log — reset for each run
    cost_log_path = output_dir / "run_log.jsonl"
    if cost_log_path.exists():
        cost_log_path.unlink()
    os.environ["COST_LOG_PATH"] = str(cost_log_path)

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"Profile: {profile.name} (volume_key={profile.volume_key})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    client = LLMClient(model=args.model, max_concurrency=args.max_concurrency)

    published_texts = Path(args.published_texts_csv)

    # Phase 1: Detect
    detections_df = phase1_detect(args, profile, client, pdf_path, output_dir)
    if detections_df.height == 0:
        print("[warn] No excavation numbers detected. Exiting.")
        return

    # Phase 2: Link
    train_csv = Path(args.few_shot_path) if args.few_shot_path else None
    linked_df = phase2_link(profile, detections_df, published_texts, output_dir, train_csv=train_csv)

    # Phase 3: Extract
    pairs_df = phase3_extract(args, profile, client, pdf_path, linked_df, output_dir)

    # Phase 4: Translate (use dedicated translate model)
    translate_client = LLMClient(model=args.translate_model, max_concurrency=args.max_concurrency)
    pairs_en_df = phase4_translate(args, profile, translate_client, pairs_df, output_dir)

    # Postprocess
    final_df = postprocess(pairs_en_df, output_dir)

    # Write pipeline summary to run_log.jsonl
    summary = {
        "label": "summary",
        "pdf": pdf_path.name,
        "profile": profile.name,
        "detections": detections_df.height,
        "pairs_raw": pairs_df.height,
        "pairs_final": final_df.height,
    }
    # Aggregate costs from phase logs
    if cost_log_path.exists():
        total_cost = 0.0
        total_tokens = 0
        for line in cost_log_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                rec = json.loads(line)
                total_cost += rec.get("total_cost", 0.0)
                total_tokens += rec.get("total_tokens", 0)
            except Exception:
                pass
        summary["total_cost"] = round(total_cost, 6)
        summary["total_tokens"] = total_tokens
    with cost_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"\nDone. detections={detections_df.height}, "
        f"pairs={pairs_df.height}, "
        f"final={final_df.height}"
    )
    if "total_cost" in summary:
        print(f"Total cost: ${summary['total_cost']:.4f}, tokens: {summary['total_tokens']:,}")


def main(args: Args) -> None:
    if args.all_pdfs:
        pdf_root = Path(args.pdf_root)
        if not pdf_root.is_dir():
            raise FileNotFoundError(f"PDF root not found: {pdf_root}")
        pdfs = sorted(pdf_root.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError(f"No PDF files found in {pdf_root}")
        print(f"Found {len(pdfs)} PDFs in {pdf_root}")
        for pdf_path in pdfs:
            run_pipeline(args, pdf_path)
    elif args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        run_pipeline(args, pdf_path)
    else:
        raise ValueError("Must specify --pdf-path or --all-pdfs")


if __name__ == "__main__":
    main(tyro.cli(Args))
