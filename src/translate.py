from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


SYSTEM_PROMPT_TURKISH = """You are an expert translator for Turkish translations of Akkadian texts.
Translate Turkish text into English under strict constraints.

Rules:
1) Keep every `<gap>` token exactly as `<gap>`.
2) Keep proper nouns exactly as written (people, places, deities, institutions).
3) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
4) If Akkadian transliteration fragments are mixed into the Turkish text, omit those fragments from the English output.
5) Prefer faithful/literal translation over paraphrase.
6) Output JSON only with this schema:
{"translation_en": string}
"""

SYSTEM_PROMPT_GERMAN = """You are an expert translator for German translations of Akkadian texts.
Translate German text into English under strict constraints.

Rules:
1) Keep every `<gap>` token exactly as `<gap>`.
2) Keep proper nouns exactly as written (people, places, deities, institutions).
3) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
4) If Akkadian transliteration fragments are mixed into the German text, omit those fragments from the English output.
5) Prefer faithful/literal translation over paraphrase.
6) Output JSON only with this schema:
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
CHRF_PLUS_PLUS = CHRF(word_order=2)
BLEU_SCORE = BLEU(effective_order=True)
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
GERMAN_VOLUME_KEYS = {
    "AKT 3",
    "HECKER KRYSZAT MATOUS - KAPPADOKISCHE KEILSCHRIFTTAFELN AUS DEN SAMMLUNGEN DER KARLSUNIVERSITAT PRAG. ICK 4 1998",
}
HECKER_VOLUME_KEYS = {
    "HECKER KRYSZAT MATOUS - KAPPADOKISCHE KEILSCHRIFTTAFELN AUS DEN SAMMLUNGEN DER KARLSUNIVERSITAT PRAG. ICK 4 1998",
}

HECKER_UNIT_INSTRUCTION = (
    "For this Hecker volume only: when the source uses the abbreviated metrological units "
    "`M.`, `T.`, and `S.`, expand them in English as `mina`/`minas`, "
    "`talent`/`talents`, and `shekel`/`shekels`, using singular or plural as appropriate."
)


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
    model: str = "Qwen/Qwen3.5-27B-FP8"

    source_column: str = "translation"
    transliteration_column: str | None = "transliteration"
    output_column: str = "translation_en"
    add_llm_raw_column: bool = False
    overwrite_output_column: bool = False
    few_shot_path: str = "./data/train_processed.csv"
    few_shot_transliteration_column: str = "transliteration"
    few_shot_translation_column: str = "translation"
    few_shot_count: int = 5
    few_shot_max_chars: int = 400
    dryrun: bool = False

    request_batch_size: int = 64
    translate_num_samples: int = 5
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 8192
    json_parse_retries: int = 2
    short_translation_retries: int = 1
    short_translation_min_source_chars: int = 80
    short_translation_min_output_chars: int = 40
    short_translation_min_ratio: float = 0.25
    enable_thinking: bool = True

    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    seed: int = 42
    trust_remote_code: bool = False
    enforce_eager: bool = False


def parse_llm_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def extract_final_answer(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


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
    return text[: max_chars - 1].rstrip() + "…"


def load_few_shot_pool(
    csv_path: str,
    transliteration_column: str,
    translation_column: str,
    max_chars: int,
) -> list[tuple[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        print(f"[warn] few-shot CSV not found: {path}. Continuing without few-shot examples.")
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
        few_df.get_column(translation_column)
        .cast(pl.String)
        .fill_null("")
        .to_list()
    )
    cleaned: list[tuple[str, str]] = []
    for trlit, tr in zip(transliterations, translations):
        trlit_clean = truncate_text(normalize_whitespace(trlit), max_chars)
        tr_clean = truncate_text(normalize_whitespace(tr), max_chars)
        if not trlit_clean or not tr_clean:
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
    sampled = rng.sample(few_shot_pool, k=k)
    return sampled


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


def pairwise_chrfpp_score(left: str, right: str) -> float:
    left_norm = normalize_whitespace(left)
    right_norm = normalize_whitespace(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(CHRF_PLUS_PLUS.sentence_score(left_norm, [right_norm]).score)


def pairwise_bleu_score(left: str, right: str) -> float:
    left_norm = normalize_whitespace(left)
    right_norm = normalize_whitespace(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(BLEU_SCORE.sentence_score(left_norm, [right_norm]).score)


def compute_geo_mean(chrfpp: float, bleu: float) -> float:
    if chrfpp <= 0.0 or bleu <= 0.0:
        return 0.0
    return math.sqrt(chrfpp * bleu)


def build_translation_candidates(out) -> list[dict]:
    candidates: list[dict] = []
    for candidate_index, output in enumerate(getattr(out, "outputs", []) or []):
        raw_text = output.text if output is not None else ""
        final_text = extract_final_answer(raw_text)
        parsed = parse_llm_json(final_text)
        valid_json = has_valid_translation_json(parsed)
        translation = extract_translation(parsed, final_text) if valid_json else final_text.strip()
        candidates.append(
            {
                "candidate_index": candidate_index,
                "raw_text": raw_text,
                "final_text": final_text,
                "parsed": parsed,
                "valid_json": valid_json,
                "translation": translation,
            }
        )
    return candidates


def select_best_translation_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        return {
            "candidate_index": -1,
            "raw_text": "",
            "final_text": "",
            "parsed": {},
            "valid_json": False,
            "translation": "",
            "mean_chrfpp": 0.0,
            "mean_bleu": 0.0,
            "geo_mean": 0.0,
        }

    best_candidate: dict | None = None
    best_key: tuple[float, float, float, float, int, int] | None = None

    for candidate in candidates:
        translation = normalize_whitespace(str(candidate.get("translation", "")))
        peer_chrfpp_scores = [
            pairwise_chrfpp_score(translation, str(other.get("translation", "")))
            for other in candidates
            if other is not candidate
        ]
        peer_bleu_scores = [
            pairwise_bleu_score(translation, str(other.get("translation", "")))
            for other in candidates
            if other is not candidate
        ]
        mean_chrfpp = (
            sum(peer_chrfpp_scores) / len(peer_chrfpp_scores) if peer_chrfpp_scores else 0.0
        )
        mean_bleu = sum(peer_bleu_scores) / len(peer_bleu_scores) if peer_bleu_scores else 0.0
        geo_mean = compute_geo_mean(mean_chrfpp, mean_bleu)
        candidate["mean_chrfpp"] = mean_chrfpp
        candidate["mean_bleu"] = mean_bleu
        candidate["geo_mean"] = geo_mean
        candidate_key = (
            1.0 if translation else 0.0,
            geo_mean,
            mean_chrfpp,
            mean_bleu,
            len(translation),
            -int(candidate.get("candidate_index", 0)),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate


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


def is_hecker_volume_key(volume_key: str) -> bool:
    normalized = re.sub(r"\s+", " ", (volume_key or "").strip()).upper()
    return normalized in HECKER_VOLUME_KEYS


def get_volume_specific_prompt_instruction(volume_key: str) -> str:
    if is_hecker_volume_key(volume_key):
        return HECKER_UNIT_INSTRUCTION
    return ""


def get_retry_prompt_suffix(retry_reason: str | None) -> str:
    if retry_reason == "json":
        return RETRY_USER_PROMPT_SUFFIX
    if retry_reason == "short":
        return RETRY_SHORT_TRANSLATION_SUFFIX
    return ""


def make_prompt(
    tokenizer,
    source_text: str,
    transliteration: str,
    source_language: str,
    volume_key: str,
    few_shot_examples: list[tuple[str, str]],
    enable_thinking: bool,
    retry_reason: str | None = None,
) -> str:
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
    volume_specific_instruction = get_volume_specific_prompt_instruction(volume_key)
    if volume_specific_instruction:
        user_text += "\n\nAdditional instruction for this source volume:\n" + volume_specific_instruction
    user_text += get_retry_prompt_suffix(retry_reason)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    base_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    if enable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **base_kwargs)
        except TypeError:
            pass

    # Compatibility path for tokenizer versions that require positional conversation arg.
    return tokenizer.apply_chat_template(messages, **base_kwargs)


def main(args: Args) -> None:
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.few_shot_count < 0:
        raise ValueError("few_shot_count must be >= 0")
    if args.json_parse_retries < 0:
        raise ValueError("json_parse_retries must be >= 0")
    if args.short_translation_retries < 0:
        raise ValueError("short_translation_retries must be >= 0")
    if args.short_translation_min_source_chars < 1:
        raise ValueError("short_translation_min_source_chars must be >= 1")
    if args.short_translation_min_output_chars < 1:
        raise ValueError("short_translation_min_output_chars must be >= 1")
    if not 0 < args.short_translation_min_ratio <= 1:
        raise ValueError("short_translation_min_ratio must be in (0, 1]")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if args.request_batch_size < 1:
        raise ValueError("request_batch_size must be >= 1")
    if args.translate_num_samples < 1:
        raise ValueError("translate_num_samples must be >= 1")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.tensor_parallel_size > num_gpus:
            raise ValueError(
                f"tensor_parallel_size={args.tensor_parallel_size} is larger than available GPUs={num_gpus}. "
                "Set --tensor-parallel-size accordingly."
            )

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

    few_shot_pool: list[tuple[str, str]] = []
    if args.few_shot_count > 0:
        few_shot_pool = load_few_shot_pool(
            csv_path=args.few_shot_path,
            transliteration_column=args.few_shot_transliteration_column,
            translation_column=args.few_shot_translation_column,
            max_chars=args.few_shot_max_chars,
        )
    few_shot_examples = sample_few_shot_examples(few_shot_pool, args.few_shot_count, args.seed)
    if args.few_shot_count == 0:
        print("Few-shot examples disabled (--few-shot-count=0).")
    else:
        print(
            "Few-shot examples loaded: "
            f"{len(few_shot_examples)} from "
            f"{args.few_shot_path}:{args.few_shot_transliteration_column},{args.few_shot_translation_column} "
            f"(seed={args.seed})"
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
    )
    tokenizer = llm.get_tokenizer()
    sampling_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(
            json=OUTPUT_JSON_SCHEMA,
            disable_fallback=True,
        ),
    )
    if args.translate_num_samples > 1:
        sampling_kwargs["n"] = args.translate_num_samples
    sampling_params = SamplingParams(**sampling_kwargs)

    source_texts = df.get_column(args.source_column).cast(pl.String).fill_null("").to_list()

    if args.transliteration_column and args.transliteration_column in df.columns:
        transliterations = (
            df.get_column(args.transliteration_column).cast(pl.String).fill_null("").to_list()
        )
    else:
        transliterations = [""] * len(df)
        if args.transliteration_column:
            print(
                f"Warning: transliteration column '{args.transliteration_column}' not found. "
                "Prompts will be generated without transliteration context."
            )

    if "volume_key" not in df.columns:
        raise ValueError(
            "Missing required column: volume_key. "
            f"Available columns={df.columns}"
        )
    volume_keys = df.get_column("volume_key").cast(pl.String).fill_null("").to_list()
    source_languages = [get_source_language_from_volume_key(v) for v in volume_keys]
    german_rows = sum(1 for lang in source_languages if lang == "German")
    turkish_rows = sum(1 for lang in source_languages if lang == "Turkish")
    skipped_rows = sum(1 for lang in source_languages if lang is None)
    print(
        "Language routing by volume_key: "
        f"German->English={german_rows}, Turkish->English={turkish_rows}, skipped={skipped_rows}"
    )

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
        pending_retry_reasons = {i: None for i in pending_indices}

        while pending_indices:
            prompts = [
                make_prompt(
                    tokenizer=tokenizer,
                    source_text=source_texts[i],
                    transliteration=transliterations[i],
                    source_language=source_languages[i],
                    volume_key=volume_keys[i],
                    few_shot_examples=few_shot_examples,
                    enable_thinking=args.enable_thinking,
                    retry_reason=pending_retry_reasons.get(i),
                )
                for i in pending_indices
            ]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            next_pending_indices: list[int] = []
            next_pending_retry_reasons: dict[int, str] = {}
            for source_idx, out in zip(pending_indices, outputs):
                candidates = build_translation_candidates(out)
                valid_candidates = [candidate for candidate in candidates if candidate["valid_json"]]

                if not valid_candidates:
                    if json_retry_counts[source_idx] < args.json_parse_retries:
                        json_retry_counts[source_idx] += 1
                        json_retry_rows.add(source_idx)
                        next_pending_indices.append(source_idx)
                        next_pending_retry_reasons[source_idx] = "json"
                        continue

                    json_parse_failures_after_retries += 1
                    selected_candidate = select_best_translation_candidate(candidates)
                else:
                    selected_candidate = select_best_translation_candidate(valid_candidates)

                batch_raw[source_idx] = str(selected_candidate.get("raw_text", ""))
                translated = str(selected_candidate.get("translation", "")).strip()
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

        for source_idx in batch_indices:
            if source_languages[source_idx] is None:
                translated_texts.append("")
                llm_raw.append("")
                continue
            translated = batch_translations.get(source_idx, "")
            source_count = count_gap_tokens(source_texts[source_idx])
            translated_fixed = enforce_gap_token_count(source_texts[source_idx], translated)
            translated_fixed = normalize_non_gap_angle_tokens(translated_fixed)
            translated_fixed = normalize_translation_fractions(translated_fixed)
            if source_count != count_gap_tokens(translated_fixed):
                gap_mismatch_rows += 1

            translated_texts.append(translated_fixed)
            llm_raw.append(batch_raw.get(source_idx, ""))

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
    print(f"Rows still failing JSON parse after retries: {json_parse_failures_after_retries}")
    print(f"Rows still too short after retries: {short_translation_failures_after_retries}")
    print(f"Rows with unresolved <gap> count mismatch after repair: {gap_mismatch_rows}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
