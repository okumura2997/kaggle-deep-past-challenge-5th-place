from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
import tyro
from tqdm import tqdm
from vllm import LLM

from extract_excavation_translation_pairs import (
    TranslationPromptProfile as PromptProfile,
    build_image_context,
    build_sampling_params,
    extract_final_answer,
    normalize_filter_key,
    normalize_prompt_lookup_key,
    normalize_ws,
    pairwise_chrfpp_score,
    parse_bool,
    parse_first_json_object,
    parse_page_list,
    render_pdf_pages,
    safe_int,
    shutdown_runtime,
)


DEFAULT_TRANSLITERATION_SYSTEM_PROMPT = """You extract Akkadian transliteration text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is usually:
  1) excavation heading
  2) summary/introduction prose (must be excluded)
  3) Akkadian transliteration block (extract this only)
  4) translation block (must be excluded)
- Extract only Akkadian transliteration text for the specified excavation number.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order.
- Start extraction at the beginning of the transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Exclude section cue tokens such as "öy.", "oy.", "ak.", and "ay.".
- Exclude marginal or inline line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude editorial section markers such as "K.", "A.y.", "Ü.k.", and "S.k." if they appear inline.
- Stop before the translation block starts.
- Do not include translation text, headings, commentary, bibliography, metadata, or section titles.
- If only summary prose is visible and the transliteration block is not visible, return empty transliteration.
- If no Akkadian transliteration is readable, return an empty transliteration string.
"""

DEFAULT_TRANSLITERATION_USER_PROMPT_TEMPLATE = """Extract the Akkadian transliteration for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}

Important:
- Extract only the transliteration corresponding to this excavation number.
- Exclude summary/introduction prose before the transliteration block.
- Exclude translation text that appears after the transliteration block.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "öy.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "Ü.k.", and "S.k.".

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": "string"
}}
"""

SIDE_BY_SIDE_TRANSLITERATION_SYSTEM_PROMPT = """You extract Akkadian transliteration text for a single excavation number from page images.
Return JSON only.

Rules:
- These pages use a side-by-side layout:
  1) LEFT side = Akkadian transliteration
  2) RIGHT side = translation
- Extract only the transliteration for the specified excavation number.
- Read only the LEFT-HAND transliteration column/area.
- Ignore the RIGHT-HAND translation column entirely, even if it is clearer or more complete.
- Do not interleave left-column transliteration with right-column translation.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order within the transliteration column, page by page.
- Header lines at the top of the entry may contain metadata, references, or edition notes. Exclude those unless they are clearly part of the transliteration itself.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "öy.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "Ü.k.", and "S.k.".
- Do not include translation text, metadata, references, bibliography, headings, commentary, or section titles.
- If no readable transliteration is visible on the left side, return an empty transliteration string.
"""

SIDE_BY_SIDE_TRANSLITERATION_USER_PROMPT_TEMPLATE = """Extract the Akkadian transliteration for this excavation number from the side-by-side page layout.
Excavation number: {excavation_number}
Target page number: {page_number}

Important:
- Extract only the transliteration corresponding to this excavation number.
- Read only the LEFT-SIDE transliteration column.
- Ignore the RIGHT-SIDE translation column completely.
- Exclude top metadata/reference lines before the transliteration begins.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "öy.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "Ü.k.", and "S.k.".

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": "string"
}}
"""

TRANSLITERATION_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_number": {"type": "string"},
        "transliteration": {"type": "string"},
    },
    "required": ["excavation_number", "transliteration"],
    "additionalProperties": False,
}

INTRO_SUMMARY_CUE_RE = re.compile(
    r"(?i)\b(?:mektuptur|yazdığı|tarafından|maktubun sahibi|yayınlanmış|belgesidir)\b"
)
LINE_MARKER_RE = re.compile(r"\b\d{1,3}(?:\s*['′]?\s*[-)\.]|['′])")
TRANSLITERATION_LINE_NUMBER_TOKEN_RE = re.compile(r"(?<!\S)(\d{1,3})\.(?=\s|$)")
TRANSLITERATION_SECTION_MARKER_RE = re.compile(
    r"(?<!\S)(?:k|a\.\s*y|[uü]\.\s*k|s\.\s*k)\.(?=\s|$)",
    flags=re.IGNORECASE,
)
TRANSLITERATION_SECTION_CUE_TOKEN_RE = re.compile(
    r"(?<!\S)(?:öy|oy|ak|ay)\.(?=\s|$)",
    flags=re.IGNORECASE,
)
TRANSLATION_START_CUE_RE = re.compile(
    r"(?i)\b(?:şöyle\s*\(söyler\)\s*:|buzutaya['’]?ya\s+söyle!?|um-ma)\b"
)
TRANSLITERATION_SECTION_CUE_RE = re.compile(r"(?i)\b(?:öy|oy|ak|ay)\.\s*\d{1,3}\b")
TRANSLATION_LINE_RANGE_RE = re.compile(r"(?m)^\s*(\d{1,3}\s*-\s*\d{1,3}\s*\))")

DEFAULT_TRANSLITERATION_PROMPT_PROFILE = PromptProfile(
    name="default_transliteration_v1",
    system_prompt=DEFAULT_TRANSLITERATION_SYSTEM_PROMPT,
    user_prompt_template=DEFAULT_TRANSLITERATION_USER_PROMPT_TEMPLATE,
)
SIDE_BY_SIDE_TRANSLITERATION_PROMPT_PROFILE = PromptProfile(
    name="side_by_side_transliteration_v1",
    system_prompt=SIDE_BY_SIDE_TRANSLITERATION_SYSTEM_PROMPT,
    user_prompt_template=SIDE_BY_SIDE_TRANSLITERATION_USER_PROMPT_TEMPLATE,
)


HECKER_ICK4_PDF_NAME = (
    "Hecker Kryszat Matous - Kappadokische Keilschrifttafeln aus den "
    "Sammlungen der Karlsuniversitat Prag. ICK 4 1998.pdf"
)
HECKER_ICK4_VOLUME_NAME = (
    "HECKER KRYSZAT MATOUS - KAPPADOKISCHE KEILSCHRIFTTAFELN AUS DEN "
    "SAMMLUNGEN DER KARLSUNIVERSITAT PRAG. ICK 4 1998"
)

TRANSLITERATION_PROMPT_PROFILES_BY_PDF_KEY: dict[str, PromptProfile] = {
    "AKT12PDF": SIDE_BY_SIDE_TRANSLITERATION_PROMPT_PROFILE,
    "LARSEN2002THEASSURNADAARCHIVEPIHANS962002PDF": SIDE_BY_SIDE_TRANSLITERATION_PROMPT_PROFILE,
    normalize_prompt_lookup_key(HECKER_ICK4_PDF_NAME): DEFAULT_TRANSLITERATION_PROMPT_PROFILE,
}
TRANSLITERATION_PROMPT_PROFILES_BY_VOLUME_KEY: dict[str, PromptProfile] = {
    "AKT12": SIDE_BY_SIDE_TRANSLITERATION_PROMPT_PROFILE,
    "LARSEN2002THEASSURNADAARCHIVEPIHANS962002": SIDE_BY_SIDE_TRANSLITERATION_PROMPT_PROFILE,
    normalize_prompt_lookup_key(HECKER_ICK4_VOLUME_NAME): DEFAULT_TRANSLITERATION_PROMPT_PROFILE,
}

TRANSLITERATION_CACHE_VERSION = "transliteration_prompt_profiles_v1"


@dataclass
class Args:
    input_csv: str = (
        "./data/find_excavation_number_pages/"
        "pdf_only_excavation_numbers_with_locations.csv"
    )
    pdf_root: str = "./input/pdfs"
    output_root: str = "./data/extract_excavation_transliteration_pairs_from_locations"
    run_name: str = ""
    dpi: int = 220
    render_workers: int = 32
    context_next_pages: int = 2
    require_found: bool = True
    target_volumes: list[str] = field(default_factory=list)
    target_pdf_names: list[str] = field(default_factory=list)
    input_row_indices: list[int] = field(default_factory=list)
    limit_rows: int = -1
    dryrun: bool = False

    model: str = "Qwen/Qwen3.5-27B-FP8"
    transliterate_batch_size: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    transliterate_max_tokens: int = 8192
    transliterate_num_samples: int = 3
    use_structured_outputs: bool = True

    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 24576
    seed: int = 42
    trust_remote_code: bool = False
    enforce_eager: bool = False


def select_transliteration_prompt_profile(pdf_name: str, akt_volume: str) -> PromptProfile:
    pdf_key = normalize_prompt_lookup_key(pdf_name)
    prompt_profile = TRANSLITERATION_PROMPT_PROFILES_BY_PDF_KEY.get(pdf_key)
    if prompt_profile is not None:
        return prompt_profile

    volume_key = normalize_prompt_lookup_key(akt_volume)
    prompt_profile = TRANSLITERATION_PROMPT_PROFILES_BY_VOLUME_KEY.get(volume_key)
    if prompt_profile is not None:
        return prompt_profile

    return DEFAULT_TRANSLITERATION_PROMPT_PROFILE


def strip_transliteration_line_numbers(text: str) -> str:
    raw = text or ""
    if not raw:
        return ""

    def _replace(match: re.Match[str]) -> str:
        try:
            number = int(match.group(1))
        except ValueError:
            return match.group(0)
        if number >= 5 and number % 5 == 0:
            return ""
        return match.group(0)

    cleaned = TRANSLITERATION_LINE_NUMBER_TOKEN_RE.sub(_replace, raw)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    return cleaned.strip()


def strip_transliteration_section_markers(text: str) -> str:
    raw = text or ""
    if not raw:
        return ""
    cleaned = TRANSLITERATION_SECTION_MARKER_RE.sub("", raw)
    cleaned = TRANSLITERATION_SECTION_CUE_TOKEN_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    return cleaned.strip()


def is_likely_summary_only(text: str) -> bool:
    normalized = normalize_ws(text)
    if not normalized:
        return False
    if INTRO_SUMMARY_CUE_RE.search(normalized) and not LINE_MARKER_RE.search(normalized):
        return True
    return False


def strip_summary_and_trailing_translation(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    cue_match = TRANSLITERATION_SECTION_CUE_RE.search(raw)
    if cue_match:
        prefix = raw[: cue_match.start()]
        if INTRO_SUMMARY_CUE_RE.search(prefix):
            raw = raw[cue_match.start() :]

    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) == 2:
        head = normalize_ws(parts[0])
        tail = normalize_ws(parts[1])
        if INTRO_SUMMARY_CUE_RE.search(head) and TRANSLITERATION_SECTION_CUE_RE.search(tail):
            raw = parts[1]

    range_match = TRANSLATION_LINE_RANGE_RE.search(raw)
    if range_match:
        raw = raw[: range_match.start()].rstrip()

    start_match = TRANSLATION_START_CUE_RE.search(raw)
    if start_match and TRANSLITERATION_SECTION_CUE_RE.search(raw[: start_match.start()]):
        raw = raw[: start_match.start()].rstrip()

    raw = strip_transliteration_line_numbers(raw)
    raw = strip_transliteration_section_markers(raw)
    return normalize_ws(raw)


def normalize_transliteration(parsed: dict, fallback_excavation_number: str) -> tuple[str, str]:
    excavation_number = normalize_ws(str(parsed.get("excavation_number", "")))
    transliteration = strip_summary_and_trailing_translation(
        str(parsed.get("transliteration", ""))
    )
    if is_likely_summary_only(transliteration):
        transliteration = ""
    if not excavation_number:
        excavation_number = fallback_excavation_number
    return excavation_number, transliteration


def make_transliteration_messages(
    excavation_number: str,
    page_number: int,
    image_context: list[tuple[str, int, Path]],
    prompt_profile: PromptProfile,
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": prompt_profile.user_prompt_template.format(
                excavation_number=excavation_number,
                page_number=page_number,
            ),
        }
    ]
    role_to_label = {"target": "TARGET page", "next": "NEXT context page"}
    for role, ctx_page, image_path in image_context:
        content.append({"type": "text", "text": f"{role_to_label[role]}: page {ctx_page}"})
        content.append({"type": "image_url", "image_url": {"url": f"file://{image_path.resolve()}"}})

    return [
        {"role": "system", "content": prompt_profile.system_prompt},
        {"role": "user", "content": content},
    ]


def build_transliteration_candidates(out, fallback_excavation_number: str) -> list[dict]:
    candidates: list[dict] = []
    for candidate_index, output in enumerate(getattr(out, "outputs", []) or []):
        raw_text = output.text if output is not None else ""
        parsed = parse_first_json_object(extract_final_answer(raw_text))
        response_excavation_number, transliteration = normalize_transliteration(
            parsed=parsed,
            fallback_excavation_number=fallback_excavation_number,
        )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "raw_text": raw_text,
                "parsed": parsed,
                "response_excavation_number": response_excavation_number,
                "transliteration": transliteration,
            }
        )
    return candidates


def select_best_transliteration_candidate(
    candidates: list[dict],
    fallback_excavation_number: str,
) -> tuple[dict, dict]:
    if not candidates:
        empty_candidate = {
            "candidate_index": -1,
            "raw_text": "",
            "parsed": {},
            "response_excavation_number": fallback_excavation_number,
            "transliteration": "",
        }
        return empty_candidate, {
            "transliteration_candidate_count": 0,
            "transliteration_selected_candidate_index": -1,
            "transliteration_selected_candidate_mean_chrfpp": 0.0,
        }

    best_candidate: dict | None = None
    best_key: tuple[float, float, int, int] | None = None

    for candidate in candidates:
        transliteration = normalize_ws(str(candidate.get("transliteration", "")))
        peer_scores = [
            pairwise_chrfpp_score(transliteration, str(other.get("transliteration", "")))
            for other in candidates
            if other is not candidate
        ]
        mean_chrfpp = sum(peer_scores) / len(peer_scores) if peer_scores else 0.0
        candidate["mean_chrfpp"] = mean_chrfpp
        candidate_key = (
            1.0 if transliteration else 0.0,
            mean_chrfpp,
            len(transliteration),
            -int(candidate.get("candidate_index", 0)),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate, {
        "transliteration_candidate_count": len(candidates),
        "transliteration_selected_candidate_index": int(best_candidate.get("candidate_index", -1)),
        "transliteration_selected_candidate_mean_chrfpp": float(
            best_candidate.get("mean_chrfpp", 0.0)
        ),
    }


def empty_transliterations_df(input_columns: list[str]) -> pl.DataFrame:
    schema = {column: pl.String for column in input_columns}
    schema.update(
        {
            "page_candidates": pl.String,
            "selected_target_page": pl.Int64,
            "context_pages": pl.String,
            "image_path": pl.String,
            "pdf_path": pl.String,
            "transliteration_request_excavation_number": pl.String,
            "transliteration_prompt_profile": pl.String,
            "transliteration_candidate_count": pl.Int64,
            "transliteration_selected_candidate_index": pl.Int64,
            "transliteration_selected_candidate_mean_chrfpp": pl.Float64,
            "response_excavation_number": pl.String,
            "transliteration": pl.String,
            "transliteration_found": pl.Boolean,
            "cache_hit": pl.Boolean,
            "error": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)


def empty_skipped_df(input_columns: list[str]) -> pl.DataFrame:
    schema = {column: pl.String for column in input_columns}
    schema.update({"skip_reason": pl.String})
    return pl.DataFrame(schema=schema)


def merge_transliterations_by_excavation(records: list[dict]) -> list[dict]:
    ordered_rows = sorted(records, key=lambda row: safe_int(str(row.get("input_row_index", "")), 10**9))
    merged: OrderedDict[str, dict] = OrderedDict()

    for row in ordered_rows:
        transliteration = normalize_ws(str(row.get("transliteration", "")))
        if not transliteration:
            continue

        key = normalize_ws(str(row.get("canonical_excavation_number", "")))
        if not key:
            key = normalize_ws(str(row.get("response_excavation_number", "")))
        if not key:
            key = normalize_ws(str(row.get("excavation_no", "")))
        if not key:
            continue

        current = {
            "excavation_number": key,
            "transliteration": transliteration,
            "source_input_row_index": str(row.get("input_row_index", "")),
            "oare_id": str(row.get("oare_id", "")),
            "akt_volume": str(row.get("akt_volume", "")),
            "pdf_names": str(row.get("pdf_names", "")),
            "selected_target_page": str(row.get("selected_target_page", "")),
        }
        existing = merged.get(key)
        if existing is None or len(transliteration) > len(str(existing.get("transliteration", ""))):
            merged[key] = current

    return list(merged.values())


def load_input_rows(args: Args) -> tuple[list[str], list[dict], list[dict]]:
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    volume_filter = {normalize_filter_key(value) for value in args.target_volumes if normalize_ws(value)}
    pdf_filter = {normalize_ws(value) for value in args.target_pdf_names if normalize_ws(value)}
    row_index_filter = set(args.input_row_indices)

    working_rows: list[dict] = []
    skipped_rows: list[dict] = []

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        input_columns = list(reader.fieldnames)

        for raw_row in reader:
            row = {key: (value if value is not None else "") for key, value in raw_row.items()}
            input_row_index = safe_int(row.get("input_row_index", ""), 10**9)

            if row_index_filter and input_row_index not in row_index_filter:
                continue
            if volume_filter and normalize_filter_key(row.get("akt_volume", "")) not in volume_filter:
                continue
            if pdf_filter and normalize_ws(row.get("pdf_names", "")) not in pdf_filter:
                continue

            if args.require_found and not parse_bool(row.get("found", "")):
                skipped_rows.append({**row, "skip_reason": "found_is_false"})
                continue

            pdf_name = normalize_ws(row.get("pdf_names", ""))
            if not pdf_name:
                skipped_rows.append({**row, "skip_reason": "missing_pdf_name"})
                continue

            page_candidates = parse_page_list(row.get("pages", ""))
            if not page_candidates:
                skipped_rows.append({**row, "skip_reason": "invalid_pages"})
                continue

            pdf_path = Path(args.pdf_root) / pdf_name
            if not pdf_path.exists():
                skipped_rows.append({**row, "skip_reason": "missing_pdf_file"})
                continue

            excavation_hint = normalize_ws(row.get("canonical_excavation_number", ""))
            if not excavation_hint:
                excavation_hint = normalize_ws(row.get("excavation_no", ""))
            if not excavation_hint:
                excavation_hint = normalize_ws(row.get("oare_id", ""))

            transliteration_prompt_profile = select_transliteration_prompt_profile(
                pdf_name=pdf_name,
                akt_volume=row.get("akt_volume", ""),
            )

            working_rows.append(
                {
                    "source_row": row,
                    "input_row_index": input_row_index,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_candidates": page_candidates,
                    "target_page": page_candidates[0],
                    "excavation_hint": excavation_hint,
                    "transliteration_prompt_profile": transliteration_prompt_profile,
                }
            )

    working_rows.sort(key=lambda row: row["input_row_index"])
    if args.limit_rows > 0:
        working_rows = working_rows[: args.limit_rows]
    return input_columns, working_rows, skipped_rows


def build_transliteration_record(
    work_row: dict,
    image_context: list[tuple[str, int, Path]],
    response_excavation_number: str,
    transliteration: str,
    selection_metadata: dict | None,
    cache_hit: bool,
    error: str,
) -> dict:
    source_row = dict(work_row["source_row"])
    image_path = ""
    if image_context:
        image_path = str(image_context[0][2])
    selection_metadata = selection_metadata or {}

    return {
        **source_row,
        "page_candidates": ",".join(str(page) for page in work_row["page_candidates"]),
        "selected_target_page": int(work_row["target_page"]),
        "context_pages": ",".join(str(page) for _, page, _ in image_context),
        "image_path": image_path,
        "pdf_path": str(work_row["pdf_path"]),
        "transliteration_request_excavation_number": work_row["excavation_hint"],
        "transliteration_prompt_profile": work_row["transliteration_prompt_profile"].name,
        "transliteration_candidate_count": int(
            selection_metadata.get("transliteration_candidate_count", 0)
        ),
        "transliteration_selected_candidate_index": int(
            selection_metadata.get("transliteration_selected_candidate_index", -1)
        ),
        "transliteration_selected_candidate_mean_chrfpp": float(
            selection_metadata.get("transliteration_selected_candidate_mean_chrfpp", 0.0)
        ),
        "response_excavation_number": response_excavation_number,
        "transliteration": transliteration,
        "transliteration_found": bool(normalize_ws(transliteration)),
        "cache_hit": cache_hit,
        "error": error,
    }


def main(args: Args) -> None:
    if args.transliterate_batch_size < 1:
        raise ValueError("transliterate_batch_size must be >= 1")
    if args.transliterate_num_samples < 1:
        raise ValueError("transliterate_num_samples must be >= 1")
    if args.render_workers < 1:
        raise ValueError("render_workers must be >= 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if args.context_next_pages < 0:
        raise ValueError("context_next_pages must be >= 0")

    input_columns, working_rows, skipped_rows = load_input_rows(args)
    if args.dryrun:
        dryrun_n = min(5, len(working_rows))
        working_rows = working_rows[:dryrun_n]
        print(f"[dryrun] Enabled. Using first {dryrun_n} rows from {args.input_csv}.")

    input_csv = Path(args.input_csv)
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        base_run_name = input_csv.stem or "run"
        run_name = f"{base_run_name}_dryrun" if args.dryrun else base_run_name

    run_dir = Path(args.output_root) / run_name
    images_root = run_dir / "images"
    transliterations_dir = run_dir / "transliterations_by_record"
    output_transliterations_path = run_dir / "transliterations_by_record.csv"
    output_unique_path = run_dir / "transliterations_unique_by_excavation.csv"
    output_pairs_path = run_dir / "pairs.csv"
    output_missing_path = run_dir / "missing_transliterations.csv"
    output_skipped_path = run_dir / "skipped_rows.csv"
    transliterate_raw_output_path = run_dir / "raw_transliterate_responses.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    transliterations_dir.mkdir(parents=True, exist_ok=True)

    pdf_to_pages: dict[str, set[int]] = defaultdict(set)
    for work_row in working_rows:
        for offset in range(0, args.context_next_pages + 1):
            pdf_to_pages[work_row["pdf_name"]].add(work_row["target_page"] + offset)

    rendered_by_pdf: dict[str, dict[int, Path]] = {}
    invalid_pages_by_pdf: dict[str, set[int]] = {}
    for pdf_name in tqdm(sorted(pdf_to_pages), desc="render_pdfs"):
        pdf_path = Path(args.pdf_root) / pdf_name
        images_dir = images_root / pdf_path.stem
        rendered, invalid_pages = render_pdf_pages(
            pdf_path=pdf_path,
            images_dir=images_dir,
            pages=sorted(pdf_to_pages[pdf_name]),
            dpi=args.dpi,
            render_workers=args.render_workers,
        )
        rendered_by_pdf[pdf_name] = rendered
        invalid_pages_by_pdf[pdf_name] = set(invalid_pages)

    if torch.cuda.is_available() and args.tensor_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"tensor_parallel_size={args.tensor_parallel_size} exceeds CUDA devices={torch.cuda.device_count()}"
        )

    quantization = (args.quantization or "").strip() or None
    if quantization is None and "fp8" in args.model.lower():
        quantization = "fp8"

    llm: LLM | None = None
    transliteration_records: list[dict] = []
    transliterate_raw_records: list[dict] = []
    try:
        if working_rows:
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
                allowed_local_media_path=str(images_root.resolve()),
                limit_mm_per_prompt={"image": 1 + args.context_next_pages},
            )

        transliteration_sampling = build_sampling_params(
            args=args,
            schema=TRANSLITERATION_OUTPUT_JSON_SCHEMA,
            max_tokens=args.transliterate_max_tokens,
            num_samples=args.transliterate_num_samples,
        )

        pending_transliterate: list[tuple[dict, Path, Path, list[tuple[str, int, Path]]]] = []
        for work_row in working_rows:
            row_idx = work_row["input_row_index"]
            transliteration_csv_path = transliterations_dir / f"row_{row_idx:05d}_transliteration.csv"
            raw_json_path = transliterations_dir / f"row_{row_idx:05d}_raw.json"

            if transliteration_csv_path.exists() and raw_json_path.exists():
                try:
                    raw_payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
                    cached_df = pl.read_csv(transliteration_csv_path)
                    cached_rows = cached_df.to_dicts()
                    if raw_payload.get("transliteration_cache_version") == TRANSLITERATION_CACHE_VERSION and len(
                        cached_rows
                    ) == 1:
                        cached_row = {**cached_rows[0], "cache_hit": True}
                        cached_target_page = safe_int(str(cached_row.get("selected_target_page", "")), -1)
                        cached_pdf_name = normalize_ws(str(cached_row.get("pdf_names", "")))
                        cached_request_excavation = normalize_ws(
                            str(cached_row.get("transliteration_request_excavation_number", ""))
                        )
                        cached_prompt_profile = normalize_ws(
                            str(
                                cached_row.get(
                                    "transliteration_prompt_profile",
                                    raw_payload.get("transliteration_prompt_profile", ""),
                                )
                            )
                        )
                        cached_num_samples = safe_int(
                            str(raw_payload.get("transliterate_num_samples", "")),
                            -1,
                        )
                        if (
                            cached_target_page == int(work_row["target_page"])
                            and cached_pdf_name == work_row["pdf_name"]
                            and cached_request_excavation == work_row["excavation_hint"]
                            and cached_prompt_profile == work_row["transliteration_prompt_profile"].name
                            and cached_num_samples == args.transliterate_num_samples
                        ):
                            transliteration_records.append(cached_row)
                            transliterate_raw_records.append(raw_payload)
                            continue
                    print(
                        f"[info] Transliteration cache is outdated or mismatched for row={row_idx}. Re-running."
                    )
                except Exception as e:
                    print(
                        f"[warn] Failed to load cached transliteration for row={row_idx}: {e}. Re-running."
                    )

            if work_row["target_page"] in invalid_pages_by_pdf.get(work_row["pdf_name"], set()):
                transliteration_record = build_transliteration_record(
                    work_row=work_row,
                    image_context=[],
                    response_excavation_number=work_row["excavation_hint"],
                    transliteration="",
                    selection_metadata=None,
                    cache_hit=False,
                    error="target_page_out_of_range",
                )
                pl.DataFrame([transliteration_record]).write_csv(transliteration_csv_path)
                transliteration_records.append(transliteration_record)
                continue

            rendered_pages = rendered_by_pdf.get(work_row["pdf_name"], {})
            image_context = build_image_context(
                rendered_pages=rendered_pages,
                target_page=work_row["target_page"],
                next_pages=args.context_next_pages,
            )
            if not image_context:
                transliteration_record = build_transliteration_record(
                    work_row=work_row,
                    image_context=[],
                    response_excavation_number=work_row["excavation_hint"],
                    transliteration="",
                    selection_metadata=None,
                    cache_hit=False,
                    error="target_page_image_missing",
                )
                pl.DataFrame([transliteration_record]).write_csv(transliteration_csv_path)
                transliteration_records.append(transliteration_record)
                continue

            pending_transliterate.append(
                (work_row, transliteration_csv_path, raw_json_path, image_context)
            )

        if pending_transliterate and llm is not None:
            for start in tqdm(
                range(0, len(pending_transliterate), args.transliterate_batch_size),
                desc="extract_transliterations",
            ):
                end = min(start + args.transliterate_batch_size, len(pending_transliterate))
                batch = pending_transliterate[start:end]
                batch_messages = [
                    make_transliteration_messages(
                        excavation_number=work_row["excavation_hint"],
                        page_number=work_row["target_page"],
                        image_context=image_context,
                        prompt_profile=work_row["transliteration_prompt_profile"],
                    )
                    for work_row, _transliteration_csv_path, _raw_json_path, image_context in batch
                ]
                outputs = llm.chat(batch_messages, sampling_params=transliteration_sampling, use_tqdm=False)

                for (work_row, transliteration_csv_path, raw_json_path, image_context), out in zip(
                    batch, outputs
                ):
                    candidates = build_transliteration_candidates(
                        out=out,
                        fallback_excavation_number=work_row["excavation_hint"],
                    )
                    selected_candidate, selection_metadata = select_best_transliteration_candidate(
                        candidates=candidates,
                        fallback_excavation_number=work_row["excavation_hint"],
                    )
                    response_excavation_number = str(selected_candidate.get("response_excavation_number", ""))
                    transliteration = str(selected_candidate.get("transliteration", ""))
                    transliteration_record = build_transliteration_record(
                        work_row=work_row,
                        image_context=image_context,
                        response_excavation_number=response_excavation_number,
                        transliteration=transliteration,
                        selection_metadata=selection_metadata,
                        cache_hit=False,
                        error="",
                    )
                    pl.DataFrame([transliteration_record]).write_csv(transliteration_csv_path)
                    transliteration_records.append(transliteration_record)

                    raw_payload = {
                        "input_row_index": work_row["input_row_index"],
                        "pdf_name": work_row["pdf_name"],
                        "pdf_path": str(work_row["pdf_path"]),
                        "target_page": work_row["target_page"],
                        "page_candidates": work_row["page_candidates"],
                        "excavation_hint": work_row["excavation_hint"],
                        "transliteration_prompt_profile": work_row["transliteration_prompt_profile"].name,
                        "transliterate_num_samples": args.transliterate_num_samples,
                        "context_pages": [page for _, page, _ in image_context],
                        "candidate_outputs": candidates,
                        "selected_candidate_index": selection_metadata[
                            "transliteration_selected_candidate_index"
                        ],
                        "selected_candidate_mean_chrfpp": selection_metadata[
                            "transliteration_selected_candidate_mean_chrfpp"
                        ],
                        "transliteration_cache_version": TRANSLITERATION_CACHE_VERSION,
                    }
                    raw_json_path.write_text(
                        json.dumps(raw_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    transliterate_raw_records.append(raw_payload)

        if transliteration_records:
            transliterations_df = pl.DataFrame(transliteration_records).sort("input_row_index")
        else:
            transliterations_df = empty_transliterations_df(input_columns)
        transliterations_df.write_csv(output_transliterations_path)

        missing_df = (
            transliterations_df.filter(
                pl.col("transliteration").fill_null("").str.strip_chars() == ""
            ).sort("input_row_index")
            if transliteration_records
            else empty_transliterations_df(input_columns)
        )
        missing_df.write_csv(output_missing_path)

        skipped_df = (
            pl.DataFrame(skipped_rows).sort("input_row_index")
            if skipped_rows
            else empty_skipped_df(input_columns)
        )
        skipped_df.write_csv(output_skipped_path)

        unique_rows = merge_transliterations_by_excavation(transliterations_df.to_dicts())
        unique_df = (
            pl.DataFrame(unique_rows).sort("excavation_number")
            if unique_rows
            else pl.DataFrame(
                schema={
                    "excavation_number": pl.String,
                    "transliteration": pl.String,
                    "source_input_row_index": pl.String,
                    "oare_id": pl.String,
                    "akt_volume": pl.String,
                    "pdf_names": pl.String,
                    "selected_target_page": pl.String,
                }
            )
        )
        unique_df.write_csv(output_unique_path)

        pairs_df = (
            unique_df.select(["excavation_number", "transliteration"])
            if len(unique_df) > 0
            else pl.DataFrame(
                schema={
                    "excavation_number": pl.String,
                    "transliteration": pl.String,
                }
            )
        )
        pairs_df.write_csv(output_pairs_path)

        transliterate_raw_records.sort(key=lambda row: safe_int(str(row.get("input_row_index", "")), 10**9))
        with transliterate_raw_output_path.open("w", encoding="utf-8") as f:
            for rec in transliterate_raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"Done. input_rows={len(working_rows)}, transliterated_rows={len(transliterations_df)}, "
            f"missing_transliterations={len(missing_df)}, skipped_rows={len(skipped_df)}, "
            f"unique_excavations={len(unique_df)}"
        )
        print(f"Run directory: {run_dir}")
        print(f"Per-record transliterations: {output_transliterations_path}")
        print(f"Unique transliterations: {output_unique_path}")
        print(f"Pairs: {output_pairs_path}")
        print(f"Missing transliterations: {output_missing_path}")
        print(f"Skipped rows: {output_skipped_path}")
        print(f"Transliterate raw responses: {transliterate_raw_output_path}")
    finally:
        shutdown_runtime(llm)


if __name__ == "__main__":
    main(tyro.cli(Args))
