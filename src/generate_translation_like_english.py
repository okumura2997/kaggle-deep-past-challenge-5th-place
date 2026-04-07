from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import polars as pl
import torch
import tyro
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


SYSTEM_PROMPT = """You generate synthetic English passages that plausibly look like scholarly translations of Akkadian transliterations.

Rules:
1) Write only one English passage, not transliteration, commentary, or notes.
2) The passage should sound like a faithful or literal translation of an ancient text.
3) Use plausible personal names, place names, deities, commodities, silver or gold amounts, measures, kinship terms, caravan or trade details, and administrative formulas when natural.
4) You may use `<gap>` to mark damaged text only when it fits naturally.
5) Do not copy or closely paraphrase any reference example.
6) Avoid modern idioms, analysis, and explanations.
7) Output JSON only with this schema:
{"translation": string}
"""

USER_PROMPT_TEMPLATE = """Reference English translations of Akkadian transliterations (style guidance only; do not copy them):
{few_shot_block}

Generate one new English passage that could plausibly be the translation of an Akkadian transliteration.

Loose profile:
- Approximate length: {length_instruction}
- Use `<gap>`: {gap_instruction}

The passage may be terse, fragmentary, formulaic, or more expansive, as long as it feels plausible.
Concrete names, quantities, and actions are welcome when natural, but do not force them all into every passage.
A lightly literal or uneven translation style is fine, and variation in structure or texture is welcome.

Return only one valid JSON object matching exactly this schema:
{{"translation": string}}
"""

RETRY_USER_PROMPT_SUFFIX = """

Important: The previous response could not be parsed as JSON.
Return only one valid JSON object matching exactly this schema:
{"translation": string}
Do not add explanations, notes, or markdown.
"""

RETRY_SHORT_OUTPUT_SUFFIX = """

Important: The previous passage was empty or too short.
Generate a somewhat fuller passage, but keep it plausible and do not force extra details unnaturally.
Return only one valid JSON object matching exactly this schema:
{"translation": string}
Do not add explanations, notes, or markdown.
"""

RETRY_DUPLICATE_SUFFIX = """

Important: The previous passage was too close to a reference example or duplicated an earlier generation.
Generate a substantially different passage by varying the structure, focus, level of detail, or scenario.
Return only one valid JSON object matching exactly this schema:
{"translation": string}
Do not add explanations, notes, or markdown.
"""

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
    },
    "required": ["translation"],
    "additionalProperties": False,
}

WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class GenerationProfile:
    target_chars: int
    include_gap: bool


@dataclass(frozen=True)
class FewShotPoolStats:
    source_rows: int
    rows_after_fold_filter: int
    usable_examples: int


@dataclass
class ExistingOutputState:
    fieldnames: list[str]
    row_count: int
    next_synthetic_id: int
    generated_norms: set[str] = field(default_factory=set)
    needs_header: bool = False


@dataclass
class Args:
    few_shot_paths: list[str] = field(
        default_factory=lambda: ["./data/train_processed_merged_with_extract.csv"]
    )
    output_path: str = "./data/generated_translation_like_english.csv"
    few_shot_translation_column: str = "translation_en"
    few_shot_translation_fallback_column: str | None = "translation"
    few_shot_fold_column: str | None = "fold"
    few_shot_exclude_folds: list[int] = field(default_factory=list)
    output_column: str = "translation"
    output_flush_rows: int = 64
    num_generations: int = 1000
    few_shot_count: int = 6
    few_shot_max_chars: int = 2048
    min_example_chars: int = 20
    min_output_chars: int = 20
    max_similarity_to_few_shot: float = 0.96
    add_llm_raw_column: bool = False
    dryrun: bool = False

    request_batch_size: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 8192
    json_parse_retries: int = 2
    short_output_retries: int = 1
    duplicate_retries: int = 2
    enable_thinking: bool = True

    model: str = "Qwen/Qwen3.5-27B-FP8"
    tensor_parallel_size: int = 2
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    seed: int = 42
    trust_remote_code: bool = False
    enforce_eager: bool = False


def normalize_whitespace(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


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


def resolve_few_shot_path(csv_paths: list[str]) -> Path:
    candidates = [Path(path) for path in csv_paths]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"few-shot CSV not found. Checked: {searched}")


def load_translation_pool(
    csv_paths: list[str],
    primary_translation_column: str,
    fallback_translation_column: str | None,
    fold_column: str | None,
    exclude_folds: list[int],
    max_chars: int,
    min_chars: int,
) -> tuple[Path, list[str], str, FewShotPoolStats]:
    path = resolve_few_shot_path(csv_paths)
    few_df = pl.read_csv(path)
    source_rows = len(few_df)

    excluded_folds = sorted(set(exclude_folds))
    if excluded_folds:
        if fold_column is None:
            raise ValueError(
                "few_shot_exclude_folds was specified, but few_shot_fold_column is None."
            )
        if fold_column not in few_df.columns:
            raise ValueError(
                "few_shot_exclude_folds was specified, but "
                f"{fold_column!r} is missing in {path}. Available columns={few_df.columns}"
            )

        excluded_fold_values = [str(fold) for fold in excluded_folds]
        few_df = few_df.filter(
            ~pl.col(fold_column)
            .cast(pl.String)
            .fill_null("")
            .str.strip_chars()
            .is_in(excluded_fold_values)
        )
    rows_after_fold_filter = len(few_df)

    primary_exists = primary_translation_column in few_df.columns
    fallback_exists = (
        fallback_translation_column is not None
        and fallback_translation_column in few_df.columns
    )
    if not primary_exists and not fallback_exists:
        raise ValueError(
            "Few-shot translation columns not found in "
            f"{path}. Requested primary='{primary_translation_column}', "
            f"fallback='{fallback_translation_column}'. "
            f"Available columns={few_df.columns}"
        )

    primary_values = [""] * len(few_df)
    if primary_exists:
        primary_values = (
            few_df.get_column(primary_translation_column)
            .cast(pl.String)
            .fill_null("")
            .to_list()
        )

    fallback_values = [""] * len(few_df)
    if fallback_exists and fallback_translation_column != primary_translation_column:
        fallback_values = (
            few_df.get_column(fallback_translation_column)
            .cast(pl.String)
            .fill_null("")
            .to_list()
        )
    elif primary_exists:
        fallback_values = primary_values

    cleaned: list[str] = []
    seen: set[str] = set()
    for primary_translation, fallback_translation in zip(primary_values, fallback_values):
        translation = normalize_whitespace(primary_translation)
        if not translation:
            translation = normalize_whitespace(fallback_translation)

        translation_clean = truncate_text(translation, max_chars)
        if len(translation_clean) < min_chars:
            continue
        key = translation_clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(translation_clean)

    if not cleaned:
        raise ValueError(
            "No usable few-shot translations were found in "
            f"{path}:{primary_translation_column} (fallback={fallback_translation_column})."
        )

    used_columns = primary_translation_column
    if fallback_exists and fallback_translation_column != primary_translation_column:
        used_columns = f"{primary_translation_column}->{fallback_translation_column}"
    elif not primary_exists and fallback_translation_column is not None:
        used_columns = fallback_translation_column
    if excluded_folds:
        used_columns += f" | excluded {fold_column} in {excluded_folds}"

    return (
        path,
        cleaned,
        used_columns,
        FewShotPoolStats(
            source_rows=source_rows,
            rows_after_fold_filter=rows_after_fold_filter,
            usable_examples=len(cleaned),
        ),
    )


def sample_few_shot_examples(
    few_shot_pool: list[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    if count <= 0 or not few_shot_pool:
        return []
    k = min(count, len(few_shot_pool))
    return rng.sample(few_shot_pool, k=k)


def format_few_shot_examples(examples: list[str]) -> str:
    return "\n\n".join(
        f"Example {idx}\nEnglish translation: {example}"
        for idx, example in enumerate(examples, start=1)
    )


def sample_generation_profile(
    few_shot_pool: list[str],
    rng: random.Random,
) -> GenerationProfile:
    anchor = few_shot_pool[rng.randrange(len(few_shot_pool))]
    target_chars = min(max(len(anchor), 40), 2400)

    return GenerationProfile(
        target_chars=target_chars,
        include_gap="<gap>" in anchor,
    )


def format_length_instruction(target_chars: int) -> str:
    lower = max(20, int(target_chars * 0.6))
    upper = max(lower + 40, int(target_chars * 1.5))
    return f"about {lower}-{upper} characters"


def get_retry_prompt_suffix(retry_reason: str | None) -> str:
    if retry_reason == "json":
        return RETRY_USER_PROMPT_SUFFIX
    if retry_reason == "short":
        return RETRY_SHORT_OUTPUT_SUFFIX
    if retry_reason == "duplicate":
        return RETRY_DUPLICATE_SUFFIX
    return ""


def make_prompt(
    tokenizer,
    few_shot_examples: list[str],
    profile: GenerationProfile,
    enable_thinking: bool,
    retry_reason: str | None = None,
) -> str:
    few_shot_block = "(none)"
    if few_shot_examples:
        few_shot_block = format_few_shot_examples(few_shot_examples)

    user_text = USER_PROMPT_TEMPLATE.format(
        few_shot_block=few_shot_block,
        length_instruction=format_length_instruction(profile.target_chars),
        gap_instruction=(
            "allowed if it fits naturally"
            if profile.include_gap
            else "usually avoid it, unless it clearly feels natural"
        ),
    )
    user_text += get_retry_prompt_suffix(retry_reason)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    base_kwargs = dict(tokenize=False, add_generation_prompt=True)

    if enable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **base_kwargs,
            )
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, **base_kwargs)


def extract_translation(parsed: dict, final_text: str) -> str:
    value = parsed.get("translation", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return final_text.strip()


def has_valid_translation_json(parsed: dict) -> bool:
    return isinstance(parsed.get("translation"), str)


def get_output_fieldnames(output_column: str, add_llm_raw_column: bool) -> list[str]:
    fieldnames = [
        "synthetic_id",
        output_column,
        "prompt_target_chars",
        "prompt_include_gap",
        "json_retry_count",
        "short_retry_count",
        "duplicate_retry_count",
    ]
    if add_llm_raw_column:
        fieldnames.append("llm_raw")
    return fieldnames


def initialize_output_csv(output_path: str | Path, fieldnames: list[str]) -> None:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def load_existing_output_state(
    output_path: str | Path,
    expected_fieldnames: list[str],
    output_column: str,
) -> ExistingOutputState:
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return ExistingOutputState(
            fieldnames=expected_fieldnames,
            row_count=0,
            next_synthetic_id=0,
            needs_header=True,
        )

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        if not existing_fieldnames:
            return ExistingOutputState(
                fieldnames=expected_fieldnames,
                row_count=0,
                next_synthetic_id=0,
                needs_header=True,
            )

        if set(existing_fieldnames) != set(expected_fieldnames):
            raise ValueError(
                "Existing output CSV columns do not match requested configuration. "
                f"path={path}, existing_columns={existing_fieldnames}, "
                f"expected_columns={expected_fieldnames}"
            )

        row_count = 0
        max_synthetic_id = -1
        generated_norms: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            synthetic_id_raw = normalize_whitespace(row.get("synthetic_id", ""))
            if not synthetic_id_raw:
                raise ValueError(
                    f"Existing output row {row_number} in {path} is missing synthetic_id."
                )

            try:
                synthetic_id = int(synthetic_id_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Existing output row {row_number} in {path} has non-integer "
                    f"synthetic_id={synthetic_id_raw!r}."
                ) from exc

            max_synthetic_id = max(max_synthetic_id, synthetic_id)

            translation = normalize_whitespace(row.get(output_column, ""))
            if translation:
                generated_norms.add(translation.casefold())
            row_count += 1

    return ExistingOutputState(
        fieldnames=existing_fieldnames,
        row_count=row_count,
        next_synthetic_id=max_synthetic_id + 1 if row_count else 0,
        generated_norms=generated_norms,
        needs_header=False,
    )


def flush_output_rows(
    output_path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    rows_written: int,
    total_rows: int,
) -> int:
    if not rows:
        return rows_written

    path = Path(output_path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)

    rows_written += len(rows)
    print(f"[flush] wrote {len(rows)} rows ({rows_written}/{total_rows}) to {path}")
    rows.clear()
    return rows_written


def is_near_duplicate(
    text: str,
    few_shot_examples: list[str],
    existing_norms: set[str],
    generated_norms: set[str],
    similarity_threshold: float,
) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return True

    key = normalized.casefold()
    if key in existing_norms or key in generated_norms:
        return True

    if similarity_threshold <= 0:
        return False

    for example in few_shot_examples:
        example_key = normalize_whitespace(example).casefold()
        if not example_key:
            continue
        if SequenceMatcher(None, key, example_key).ratio() >= similarity_threshold:
            return True

    return False


def main(args: Args) -> None:
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.few_shot_count < 0:
        raise ValueError("few_shot_count must be >= 0")
    if not args.few_shot_paths:
        raise ValueError("few_shot_paths must not be empty")
    if args.few_shot_exclude_folds and args.few_shot_fold_column is None:
        raise ValueError(
            "few_shot_exclude_folds requires few_shot_fold_column to be set."
        )
    if args.num_generations < 0:
        raise ValueError("num_generations must be >= 0")
    if args.request_batch_size < 1:
        raise ValueError("request_batch_size must be >= 1")
    if args.output_flush_rows < 1:
        raise ValueError("output_flush_rows must be >= 1")
    if args.min_output_chars < 1:
        raise ValueError("min_output_chars must be >= 1")
    if args.min_example_chars < 1:
        raise ValueError("min_example_chars must be >= 1")
    if args.json_parse_retries < 0:
        raise ValueError("json_parse_retries must be >= 0")
    if args.short_output_retries < 0:
        raise ValueError("short_output_retries must be >= 0")
    if args.duplicate_retries < 0:
        raise ValueError("duplicate_retries must be >= 0")
    if not 0 <= args.max_similarity_to_few_shot <= 1:
        raise ValueError("max_similarity_to_few_shot must be in [0, 1]")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.tensor_parallel_size > num_gpus:
            raise ValueError(
                f"tensor_parallel_size={args.tensor_parallel_size} is larger than available GPUs={num_gpus}. "
                "Set --tensor-parallel-size accordingly."
            )

    (
        resolved_few_shot_path,
        few_shot_pool,
        few_shot_column_spec,
        few_shot_pool_stats,
    ) = load_translation_pool(
        csv_paths=args.few_shot_paths,
        primary_translation_column=args.few_shot_translation_column,
        fallback_translation_column=args.few_shot_translation_fallback_column,
        fold_column=args.few_shot_fold_column,
        exclude_folds=args.few_shot_exclude_folds,
        max_chars=args.few_shot_max_chars,
        min_chars=args.min_example_chars,
    )

    total = args.num_generations or len(few_shot_pool)
    if args.dryrun:
        total = min(total, 8)
        print(f"[dryrun] Enabled. Generating {total} rows.")

    print(
        "Few-shot pool loaded: "
        f"{len(few_shot_pool)} examples from "
        f"{resolved_few_shot_path}:{few_shot_column_spec}"
    )
    print(
        "Few-shot pool stats: "
        f"source_rows={few_shot_pool_stats.source_rows}, "
        f"rows_after_fold_filter={few_shot_pool_stats.rows_after_fold_filter}, "
        f"usable_examples={few_shot_pool_stats.usable_examples}, "
        f"sampled_per_prompt={min(args.few_shot_count, few_shot_pool_stats.usable_examples)}"
    )
    print(f"Target total rows: {total}")

    requested_output_fieldnames = get_output_fieldnames(
        output_column=args.output_column,
        add_llm_raw_column=args.add_llm_raw_column,
    )
    existing_output_state = load_existing_output_state(
        output_path=args.output_path,
        expected_fieldnames=requested_output_fieldnames,
        output_column=args.output_column,
    )
    output_fieldnames = existing_output_state.fieldnames

    if existing_output_state.needs_header:
        initialize_output_csv(args.output_path, output_fieldnames)

    existing_row_count = existing_output_state.row_count
    if existing_row_count:
        print(
            f"Resuming from {args.output_path}: "
            f"{existing_row_count} existing rows, "
            f"next synthetic_id={existing_output_state.next_synthetic_id}"
        )

    if existing_row_count >= total:
        print(
            f"Output already has {existing_row_count} rows, "
            f"which meets or exceeds the target of {total}. Nothing to do."
        )
        return

    print(
        f"Streaming output to {args.output_path} "
        f"(flush every {args.output_flush_rows} rows)"
    )
    print(f"Rows remaining to generate: {total - existing_row_count}")

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
    sampling_params = SamplingParams(**sampling_kwargs)

    rng = random.Random(args.seed)
    existing_norms = {text.casefold() for text in few_shot_pool}
    generated_norms: set[str] = set(existing_output_state.generated_norms)

    remaining_rows = total - existing_row_count
    generation_start = existing_output_state.next_synthetic_id
    generation_end = generation_start + remaining_rows

    json_retry_counts = [0] * generation_end
    short_retry_counts = [0] * generation_end
    duplicate_retry_counts = [0] * generation_end

    json_retry_rows: set[int] = set()
    short_retry_rows: set[int] = set()
    duplicate_retry_rows: set[int] = set()
    json_failures_after_retries = 0
    short_failures_after_retries = 0
    duplicate_failures_after_retries = 0
    output_rows_buffer: list[dict[str, object]] = []
    rows_written = existing_row_count

    for start in tqdm(
        range(generation_start, generation_end, args.request_batch_size),
        desc="generate",
    ):
        end = min(start + args.request_batch_size, generation_end)
        batch_indices = list(range(start, end))
        batch_profiles = {
            idx: sample_generation_profile(few_shot_pool, rng) for idx in batch_indices
        }
        batch_examples = {
            idx: sample_few_shot_examples(few_shot_pool, args.few_shot_count, rng)
            for idx in batch_indices
        }
        pending_indices = batch_indices[:]
        pending_retry_reasons = {idx: None for idx in batch_indices}
        batch_generated_norms: set[str] = set()
        batch_translations = {idx: "" for idx in batch_indices}
        batch_llm_raw = {idx: "" for idx in batch_indices}

        while pending_indices:
            prompts = [
                make_prompt(
                    tokenizer=tokenizer,
                    few_shot_examples=batch_examples[idx],
                    profile=batch_profiles[idx],
                    enable_thinking=args.enable_thinking,
                    retry_reason=pending_retry_reasons.get(idx),
                )
                for idx in pending_indices
            ]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            next_pending_indices: list[int] = []
            next_pending_retry_reasons: dict[int, str] = {}

            for idx, out in zip(pending_indices, outputs):
                raw = out.outputs[0].text if out.outputs else ""
                batch_llm_raw[idx] = raw
                final_text = extract_final_answer(raw)
                parsed = parse_llm_json(final_text)

                if not has_valid_translation_json(parsed):
                    if json_retry_counts[idx] < args.json_parse_retries:
                        json_retry_counts[idx] += 1
                        json_retry_rows.add(idx)
                        batch_profiles[idx] = sample_generation_profile(few_shot_pool, rng)
                        batch_examples[idx] = sample_few_shot_examples(
                            few_shot_pool,
                            args.few_shot_count,
                            rng,
                        )
                        next_pending_indices.append(idx)
                        next_pending_retry_reasons[idx] = "json"
                        continue
                    json_failures_after_retries += 1

                translation = normalize_whitespace(extract_translation(parsed, final_text))
                if len(translation) < args.min_output_chars:
                    if short_retry_counts[idx] < args.short_output_retries:
                        short_retry_counts[idx] += 1
                        short_retry_rows.add(idx)
                        batch_profiles[idx] = sample_generation_profile(few_shot_pool, rng)
                        batch_examples[idx] = sample_few_shot_examples(
                            few_shot_pool,
                            args.few_shot_count,
                            rng,
                        )
                        next_pending_indices.append(idx)
                        next_pending_retry_reasons[idx] = "short"
                        continue
                    short_failures_after_retries += 1

                current_generated_norms = generated_norms | batch_generated_norms
                if is_near_duplicate(
                    text=translation,
                    few_shot_examples=batch_examples[idx],
                    existing_norms=existing_norms,
                    generated_norms=current_generated_norms,
                    similarity_threshold=args.max_similarity_to_few_shot,
                ):
                    if duplicate_retry_counts[idx] < args.duplicate_retries:
                        duplicate_retry_counts[idx] += 1
                        duplicate_retry_rows.add(idx)
                        batch_profiles[idx] = sample_generation_profile(few_shot_pool, rng)
                        batch_examples[idx] = sample_few_shot_examples(
                            few_shot_pool,
                            args.few_shot_count,
                            rng,
                        )
                        next_pending_indices.append(idx)
                        next_pending_retry_reasons[idx] = "duplicate"
                        continue
                    duplicate_failures_after_retries += 1

                batch_translations[idx] = translation
                batch_generated_norms.add(translation.casefold())

            pending_indices = next_pending_indices
            pending_retry_reasons = next_pending_retry_reasons

        for idx in batch_indices:
            translation = batch_translations[idx]
            if translation:
                generated_norms.add(translation.casefold())

            row = {
                "synthetic_id": idx,
                args.output_column: translation,
                "prompt_target_chars": batch_profiles[idx].target_chars,
                "prompt_include_gap": batch_profiles[idx].include_gap,
                "json_retry_count": json_retry_counts[idx],
                "short_retry_count": short_retry_counts[idx],
                "duplicate_retry_count": duplicate_retry_counts[idx],
            }
            if args.add_llm_raw_column:
                row["llm_raw"] = batch_llm_raw[idx]
            output_rows_buffer.append(row)

        if len(output_rows_buffer) >= args.output_flush_rows:
            rows_written = flush_output_rows(
                output_path=args.output_path,
                fieldnames=output_fieldnames,
                rows=output_rows_buffer,
                rows_written=rows_written,
                total_rows=total,
            )

    rows_written = flush_output_rows(
        output_path=args.output_path,
        fieldnames=output_fieldnames,
        rows=output_rows_buffer,
        rows_written=rows_written,
        total_rows=total,
    )
    print(
        f"Done. total_rows={rows_written}, "
        f"newly_generated={rows_written - existing_row_count}, "
        f"output={args.output_path}"
    )
    print(f"Output column: {args.output_column}")
    print(f"Rows retried due to JSON parse failure: {len(json_retry_rows)}")
    print(f"Rows retried due to short output: {len(short_retry_rows)}")
    print(f"Rows retried due to duplicate-like output: {len(duplicate_retry_rows)}")
    print(f"Rows still failing JSON parse after retries: {json_failures_after_retries}")
    print(f"Rows still short after retries: {short_failures_after_retries}")
    print(f"Rows still duplicate-like after retries: {duplicate_failures_after_retries}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
