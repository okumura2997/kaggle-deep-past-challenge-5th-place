"""Translate Turkish/German translations of Akkadian texts into English via litellm.

Ported from deep-past-initiative-machine-translation/src/translate.py.
Replaces vLLM with LLMClient (litellm + OpenRouter).
Incorporates per-sample few-shot from translate_turkish_translation_to_english_v4.py.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tqdm import tqdm

from extraction_pipeline._json_utils import (
    extract_final_answer,
    parse_first_json_object,
)
from extraction_pipeline._llm_client import LLMClient
from extraction_pipeline.utils.similar import SimilarTransliterationRetriever

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TURKISH = """You are an expert translator for Turkish translations of Akkadian texts.
Translate Turkish text into English under strict constraints.

Rules:
1) Keep proper nouns exactly as written (people, places, deities, institutions).
2) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
3) If Akkadian transliteration fragments are mixed into the Turkish text, omit those fragments from the English output.
4) Prefer faithful/literal translation over paraphrase.
5) Output JSON only with this schema:
{"translation_en": string}
"""

SYSTEM_PROMPT_GERMAN = """You are an expert translator for German translations of Akkadian texts.
Translate German text into English under strict constraints.

Rules:
1) Keep proper nouns exactly as written (people, places, deities, institutions).
2) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
3) If Akkadian transliteration fragments are mixed into the German text, omit those fragments from the English output.
4) Prefer faithful/literal translation over paraphrase.
5) Output JSON only with this schema:
{"translation_en": string}
"""

USER_PROMPT_TEMPLATE = """Akkadian transliteration (reference only, do not translate):
{transliteration}
{few_shot_block}

{source_language} text to translate into English:
{source_text}
"""

USER_PROMPT_FEW_SHOT_BLOCK_TEMPLATE = """
Reference Akkadian transliteration -> English translation examples (few-shot style guidance only):
{few_shot_examples}"""

RETRY_USER_PROMPT_SUFFIX = """

Important: The previous response could not be parsed as JSON.
Return only one valid JSON object matching exactly this schema:
{"translation_en": string}
Do not add explanations, notes, or markdown.
"""

RETRY_SHORT_TRANSLATION_SUFFIX = """

Important: The previous translation was too short or incomplete.
Translate the entire source text faithfully.
Do not stop after the opening formula, first clause, or first sentence.
Return only one valid JSON object matching exactly this schema:
{"translation_en": string}
Do not add explanations, notes, or markdown.
"""

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translation_en": {"type": "string"},
    },
    "required": ["translation_en"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Regex / constants
# ---------------------------------------------------------------------------

GAP_TOKEN_RE = re.compile(r"<\s*gap\s*>|\[\s*gap\s*\]", flags=re.IGNORECASE)
ANGLE_WRAPPED_TOKEN_RE = re.compile(r"<\s*([^<>]+?)\s*>")
WS_RE = re.compile(r"\s+")
SIMPLE_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s*[/⁄]\s*(\d+)(?!\d)")
FRACTION_TO_GLYPH: dict[tuple[int, int], str] = {
    (0, 3): "↉",
    (1, 10): "⅒",
    (1, 9): "⅑",
    (1, 7): "⅐",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (5, 8): "⅝",
    (7, 8): "⅞",
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
}
TURKISH_VOLUME_KEYS = {
    "AKT 1",
    "AKT 2",
    "AKT 4",
    "AKT 7A",
    "AKT 7B",
    "AKT 9A",
    "AKT 10",
    "AKT 11A",
    "AKT 11B",
}
GERMAN_VOLUME_KEYS = {"AKT 3"}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    input_path: str = (
        "./data/extract_excavation_translation_pairs_from_locations/"
        "translations_by_record_merged_processed.csv"
    )
    output_path: str = (
        "./data/extract_excavation_translation_pairs_from_locations/"
        "translations_by_record_merged_processed_en.csv"
    )
    model: str = "openrouter/openai/gpt-5.4-mini"

    source_column: str = "translation"
    transliteration_column: str | None = "transliteration"
    output_column: str = "translation_en"
    add_llm_raw_column: bool = False
    overwrite_output_column: bool = False
    few_shot_path: str = "./data/train_processed.csv"
    few_shot_transliteration_column: str = "transliteration"
    few_shot_translation_column: str = "translation"
    few_shot_count: int = 3
    few_shot_max_chars: int = 400
    few_shot_per_sample: bool = False
    few_shot_similarity: bool = False
    dryrun: bool = False

    max_concurrency: int = 8
    request_batch_size: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 8192
    json_parse_retries: int = 2
    short_translation_retries: int = 1
    short_translation_min_source_chars: int = 80
    short_translation_min_output_chars: int = 40
    short_translation_min_ratio: float = 0.25
    use_structured_outputs: bool = True

    seed: int = 42


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def normalize_gap_token_forms(text: str) -> str:
    return GAP_TOKEN_RE.sub("<gap>", text or "")


def count_gap_tokens(text: str) -> int:
    return len(GAP_TOKEN_RE.findall(text or ""))


def normalize_whitespace(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def normalize_translation_fractions(text: str) -> str:
    def replace_simple(match: re.Match[str]) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return glyph

    return SIMPLE_FRACTION_RE.sub(replace_simple, text or "")


def normalize_non_gap_angle_tokens(text: str) -> str:
    def replace_token(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)
        if inner.lower() == "gap":
            return "<gap>"
        return f"({inner})"

    return ANGLE_WRAPPED_TOKEN_RE.sub(replace_token, text or "")


def is_translation_too_short(
    source_text: str,
    translated_text: str,
    min_source_chars: int,
    min_output_chars: int,
    min_ratio: float,
) -> bool:
    source_len = len(normalize_whitespace(source_text))
    if source_len < min_source_chars:
        return False
    translated_len = len(normalize_whitespace(translated_text))
    if translated_len == 0:
        return False
    length_ratio = translated_len / source_len
    return translated_len < min_output_chars or length_ratio < min_ratio


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


def enforce_gap_token_count(source_text: str, translated_text: str) -> str:
    source_count = count_gap_tokens(source_text)
    text = normalize_gap_token_forms(translated_text)
    if source_count == 0:
        return text

    matches = list(re.finditer(r"<gap>", text))
    output_count = len(matches)
    if output_count == source_count:
        return text

    kept = 0
    parts: list[str] = []
    last = 0
    for match in matches:
        parts.append(text[last : match.start()])
        if kept < source_count:
            parts.append("<gap>")
            kept += 1
        last = match.end()
    parts.append(text[last:])
    return "".join(parts).strip()


def extract_translation(parsed: dict, final_text: str) -> str:
    if parsed:
        value = parsed.get("translation_en", "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return final_text.strip()


def has_valid_translation_json(parsed: dict) -> bool:
    return isinstance(parsed.get("translation_en"), str)


# ---------------------------------------------------------------------------
# Language routing
# ---------------------------------------------------------------------------


def get_source_language_from_volume_key(volume_key: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (volume_key or "").strip()).upper()
    if normalized in GERMAN_VOLUME_KEYS:
        return "German"
    if normalized in TURKISH_VOLUME_KEYS:
        return "Turkish"
    return None


def get_system_prompt_for_language(source_language: str) -> str:
    if source_language == "German":
        return SYSTEM_PROMPT_GERMAN
    return SYSTEM_PROMPT_TURKISH


def get_retry_prompt_suffix(retry_reason: str | None) -> str:
    if retry_reason == "json":
        return RETRY_USER_PROMPT_SUFFIX
    if retry_reason == "short":
        return RETRY_SHORT_TRANSLATION_SUFFIX
    return ""


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------


def load_few_shot_pool(
    csv_path: str,
    transliteration_column: str,
    translation_column: str,
    max_chars: int,
    min_chars: int = 0,
) -> list[tuple[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        print(
            f"[warn] few-shot CSV not found: {path}. Continuing without few-shot examples."
        )
        return []

    few_df = pl.read_csv(path)
    if transliteration_column not in few_df.columns:
        print(
            f"[warn] few-shot transliteration column '{transliteration_column}' not found in {path}. "
            "Continuing without few-shot examples."
        )
        return []
    if translation_column not in few_df.columns:
        print(
            f"[warn] few-shot translation column '{translation_column}' not found in {path}. "
            "Continuing without few-shot examples."
        )
        return []

    transliterations = (
        few_df.get_column(transliteration_column)
        .cast(pl.String)
        .fill_null("")
        .to_list()
    )
    translations = (
        few_df.get_column(translation_column).cast(pl.String).fill_null("").to_list()
    )
    cleaned: list[tuple[str, str]] = []
    for trlit, tr in zip(transliterations, translations):
        trlit_clean = truncate_text(normalize_whitespace(trlit), max_chars)
        tr_clean = truncate_text(normalize_whitespace(tr), max_chars)
        if not trlit_clean or not tr_clean:
            continue
        if min_chars > 0 and (len(trlit_clean) < min_chars or len(tr_clean) < min_chars):
            continue
        cleaned.append((trlit_clean, tr_clean))
    if not cleaned:
        print(
            "[warn] No usable few-shot pairs found in "
            f"{path}:{transliteration_column},{translation_column}."
        )
        return []

    return cleaned


def sample_few_shot_examples(
    few_shot_pool: list[tuple[str, str]],
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    if count <= 0 or not few_shot_pool:
        return []
    rng = random.Random(seed)
    k = min(count, len(few_shot_pool))
    return rng.sample(few_shot_pool, k=k)


def format_few_shot_examples(examples: list[tuple[str, str]]) -> str:
    lines = [
        (
            f"Example {idx}\n"
            f"Transliteration: {transliteration}\n"
            f"English translation: {translation}"
        )
        for idx, (transliteration, translation) in enumerate(examples, start=1)
    ]
    return "\n\n".join(lines)


def build_similarity_retriever(
    pool: list[tuple[str, str]],
) -> SimilarTransliterationRetriever | None:
    if not pool:
        return None
    corpus = [t for t, _ in pool]
    metadata = [{"translation": tr} for _, tr in pool]
    return SimilarTransliterationRetriever(corpus=corpus, metadata=metadata)


def retrieve_similar_few_shot(
    retriever: SimilarTransliterationRetriever,
    query_transliteration: str,
    count: int,
) -> list[tuple[str, str]]:
    query = (query_transliteration or "").strip()
    if not query or query.lower() == "(unavailable)":
        return []
    results = retriever.query(query, k=count)
    return [(r["transliteration"], r["translation"]) for r in results]


# ---------------------------------------------------------------------------
# Message construction (litellm messages format)
# ---------------------------------------------------------------------------


def make_messages(
    source_text: str,
    transliteration: str,
    source_language: str,
    few_shot_examples: list[tuple[str, str]],
    retry_reason: str | None = None,
) -> list[dict]:
    system_prompt = get_system_prompt_for_language(source_language)
    few_shot_block = ""
    if few_shot_examples:
        few_shot_block = USER_PROMPT_FEW_SHOT_BLOCK_TEMPLATE.format(
            few_shot_examples=format_few_shot_examples(few_shot_examples)
        )
    user_text = USER_PROMPT_TEMPLATE.format(
        transliteration=(transliteration or "").strip() or "(none)",
        few_shot_block=few_shot_block,
        source_language=source_language,
        source_text=(source_text or "").strip(),
    )
    user_text += get_retry_prompt_suffix(retry_reason)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.few_shot_count < 0:
        raise ValueError("few_shot_count must be >= 0")
    if args.json_parse_retries < 0:
        raise ValueError("json_parse_retries must be >= 0")
    if args.short_translation_retries < 0:
        raise ValueError("short_translation_retries must be >= 0")
    if args.request_batch_size < 1:
        raise ValueError("request_batch_size must be >= 1")

    df = pl.read_csv(args.input_path)
    if args.source_column not in df.columns:
        raise ValueError(
            f"Missing required source column: {args.source_column}. "
            f"Available columns={df.columns}"
        )
    if args.output_column in df.columns and not args.overwrite_output_column:
        raise ValueError(
            f"Output column already exists: {args.output_column}. "
            "Use --overwrite-output-column to replace it."
        )

    if args.dryrun:
        dryrun_n = min(5, len(df))
        df = df.head(dryrun_n)
        print(f"[dryrun] Enabled. Using first {dryrun_n} rows from {args.input_path}.")

    if len(df) == 0:
        df.write_csv(args.output_path)
        print("Input is empty. Wrote empty output.")
        return

    # Few-shot setup
    few_shot_pool: list[tuple[str, str]] = []
    retriever: SimilarTransliterationRetriever | None = None
    if args.few_shot_count > 0:
        few_shot_pool = load_few_shot_pool(
            csv_path=args.few_shot_path,
            transliteration_column=args.few_shot_transliteration_column,
            translation_column=args.few_shot_translation_column,
            max_chars=args.few_shot_max_chars,
        )
    if args.few_shot_similarity:
        retriever = build_similarity_retriever(few_shot_pool)
    few_shot_examples = sample_few_shot_examples(
        few_shot_pool, args.few_shot_count, args.seed
    )
    if args.few_shot_similarity:
        print(
            f"Few-shot pool loaded: {len(few_shot_pool)} candidates from "
            f"{args.few_shot_path} (mode=similarity, count={args.few_shot_count})"
        )
    elif args.few_shot_per_sample:
        print(
            f"Few-shot pool loaded: {len(few_shot_pool)} candidates from "
            f"{args.few_shot_path} (mode=per-sample, count={args.few_shot_count}, "
            f"base_seed={args.seed})"
        )
    elif args.few_shot_count == 0:
        print("Few-shot examples disabled (--few-shot-count=0).")
    else:
        print(
            f"Few-shot examples loaded: {len(few_shot_examples)} from "
            f"{args.few_shot_path} (mode=global, seed={args.seed})"
        )

    # Extract columns
    source_texts = (
        df.get_column(args.source_column).cast(pl.String).fill_null("").to_list()
    )

    if args.transliteration_column and args.transliteration_column in df.columns:
        transliterations = (
            df.get_column(args.transliteration_column)
            .cast(pl.String)
            .fill_null("")
            .to_list()
        )
    else:
        transliterations = [""] * len(df)
        if args.transliteration_column:
            print(
                f"Warning: transliteration column '{args.transliteration_column}' not found. "
                "Prompts will be generated without transliteration context."
            )

    # Language routing
    if "volume_key" in df.columns:
        volume_keys = (
            df.get_column("volume_key").cast(pl.String).fill_null("").to_list()
        )
        source_languages = [get_source_language_from_volume_key(v) for v in volume_keys]
    elif "pairs_csv" in df.columns:
        # Fallback: infer from pairs_csv column (v4 compat)
        pairs_csv_values = (
            df.get_column("pairs_csv").cast(pl.String).fill_null("").to_list()
        )
        akt3_re = re.compile(r"^\s*AKT\s*3\b", flags=re.IGNORECASE)
        source_languages = [
            "German" if akt3_re.match(v) else "Turkish" for v in pairs_csv_values
        ]
    else:
        print(
            "Warning: Neither 'volume_key' nor 'pairs_csv' column found. "
            "Defaulting to Turkish->English for all rows."
        )
        source_languages = ["Turkish"] * len(df)

    german_rows = sum(1 for lang in source_languages if lang == "German")
    turkish_rows = sum(1 for lang in source_languages if lang == "Turkish")
    skipped_rows = sum(1 for lang in source_languages if lang is None)
    print(
        f"Language routing: German->English={german_rows}, "
        f"Turkish->English={turkish_rows}, skipped={skipped_rows}"
    )

    # LLM client
    client = LLMClient(model=args.model, max_concurrency=args.max_concurrency)
    response_format = {"type": "json_object"} if args.use_structured_outputs else None

    # Translation loop
    translated_texts: list[str] = []
    llm_raw: list[str] = []
    gap_mismatch_rows = 0
    json_retry_rows: set[int] = set()
    short_retry_rows: set[int] = set()
    json_parse_failures_after_retries = 0
    short_translation_failures_after_retries = 0
    total = len(df)
    json_retry_counts = [0] * total
    short_retry_counts = [0] * total

    for start in tqdm(range(0, total, args.request_batch_size), desc="translate"):
        end = min(start + args.request_batch_size, total)
        batch_indices = list(range(start, end))
        batch_translations: dict[int, str] = {}
        batch_raw: dict[int, str] = {}
        pending_indices = [i for i in batch_indices if source_languages[i] is not None]
        pending_retry_reasons: dict[int, str | None] = {
            i: None for i in pending_indices
        }

        while pending_indices:
            batch_messages = [
                make_messages(
                    source_text=source_texts[i],
                    transliteration=transliterations[i],
                    source_language=source_languages[i],
                    few_shot_examples=(
                        retrieve_similar_few_shot(
                            retriever, transliterations[i], args.few_shot_count
                        )
                        if retriever is not None
                        else sample_few_shot_examples(
                            few_shot_pool, args.few_shot_count, args.seed + i
                        )
                        if args.few_shot_per_sample
                        else few_shot_examples
                    ),
                    retry_reason=pending_retry_reasons.get(i),
                )
                for i in pending_indices
            ]
            responses = client.batch_chat(
                batch_messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                response_format=response_format,
            )

            next_pending_indices: list[int] = []
            next_pending_retry_reasons: dict[int, str | None] = {}
            for source_idx, response_texts in zip(pending_indices, responses):
                raw = response_texts[0] if response_texts else ""
                final_text = extract_final_answer(raw)
                parsed = parse_first_json_object(final_text)
                batch_raw[source_idx] = raw

                if not has_valid_translation_json(parsed):
                    if json_retry_counts[source_idx] < args.json_parse_retries:
                        json_retry_counts[source_idx] += 1
                        json_retry_rows.add(source_idx)
                        next_pending_indices.append(source_idx)
                        next_pending_retry_reasons[source_idx] = "json"
                        continue

                    json_parse_failures_after_retries += 1
                    batch_translations[source_idx] = final_text.strip()
                    continue

                translated = extract_translation(parsed, final_text)
                if is_translation_too_short(
                    source_text=source_texts[source_idx],
                    translated_text=translated,
                    min_source_chars=args.short_translation_min_source_chars,
                    min_output_chars=args.short_translation_min_output_chars,
                    min_ratio=args.short_translation_min_ratio,
                ):
                    if short_retry_counts[source_idx] < args.short_translation_retries:
                        short_retry_counts[source_idx] += 1
                        short_retry_rows.add(source_idx)
                        next_pending_indices.append(source_idx)
                        next_pending_retry_reasons[source_idx] = "short"
                        continue

                    short_translation_failures_after_retries += 1
                batch_translations[source_idx] = translated

            pending_indices = next_pending_indices
            pending_retry_reasons = next_pending_retry_reasons

        # Post-process batch
        for source_idx in batch_indices:
            if source_languages[source_idx] is None:
                translated_texts.append("")
                llm_raw.append("")
                continue
            translated = batch_translations.get(source_idx, "")
            translated_fixed = enforce_gap_token_count(
                source_texts[source_idx], translated
            )
            translated_fixed = normalize_non_gap_angle_tokens(translated_fixed)
            translated_fixed = normalize_translation_fractions(translated_fixed)
            if count_gap_tokens(source_texts[source_idx]) != count_gap_tokens(
                translated_fixed
            ):
                gap_mismatch_rows += 1

            translated_texts.append(translated_fixed)
            llm_raw.append(batch_raw.get(source_idx, ""))

    # Write output
    out_df = df.with_columns(pl.Series(args.output_column, translated_texts))
    if args.add_llm_raw_column:
        raw_col = "llm_raw_translation_en"
        if raw_col in out_df.columns and not args.overwrite_output_column:
            raise ValueError(
                f"Raw output column already exists: {raw_col}. "
                "Use --overwrite-output-column to replace it."
            )
        out_df = out_df.with_columns(pl.Series(raw_col, llm_raw))

    out_df.write_csv(args.output_path)
    print(f"Done. translated={len(out_df)}, output={args.output_path}")
    print(f"Output column: {args.output_column}")
    print(f"Rows retried due to JSON parse failure: {len(json_retry_rows)}")
    print(f"Rows retried due to short translation: {len(short_retry_rows)}")
    print(
        f"Rows still failing JSON parse after retries: {json_parse_failures_after_retries}"
    )
    print(
        f"Rows still too short after retries: {short_translation_failures_after_retries}"
    )
    print(
        f"Rows with unresolved <gap> count mismatch after repair: {gap_mismatch_rows}"
    )
    client.print_usage_summary(label="translate_to_english")


if __name__ == "__main__":
    main(tyro.cli(Args))
