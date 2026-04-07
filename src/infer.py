import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import sacrebleu
import torch
import tyro
from datasets import Dataset
from preprocess import normalize_transliteration
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from utils import seed_everything


@dataclass
class Args:
    model: str = "./outputs/finetune/fresh-night-48/best_model/"
    input_path: str = "./input/deep-past-initiative-machine-translation/test.csv"
    output_path: str = "./outputs/submission.csv"
    sample_scores_path: str = "./outputs/eval_sample_scores.csv"

    translation_direction: Literal["akkadian_to_english", "english_to_akkadian"] = "akkadian_to_english"
    input_transliteration_column: str = "transliteration"
    input_translation_column: str = "translation"
    output_id_column: str = "id"
    output_confidence_column: str = "confidence"
    add_transliteration: bool = False
    add_translation: bool = False
    apply_transliteration_char_normalization: bool = True

    compute_confidence: bool = False
    dryrun: bool = False
    fold: int | None = None

    preprocess_num_proc: int = 1
    dataloader_num_workers: int = 0

    max_length: int = 1024
    per_device_eval_batch_size: int = 1
    generation_num_beams: int = 8
    generation_num_return_sequences: int = 1
    generation_do_sample: bool = False
    generation_max_new_tokens: int = 1024
    generation_max_new_tokens_input_ratio: float = 1.7
    generation_length_penalty: float = 1.0
    generation_early_stopping: bool = True
    no_repeat_ngram_size: int | None = None
    rerank_strategy: Literal["first", "mbr_chrf"] = "first"
    gpu_ids: str | None = None
    write_output_per_batch: bool = False


@dataclass(frozen=True)
class BatchOutputConfig:
    output_path: str
    output_id_column: str
    prediction_output_column: str
    source_column: str
    source_output_column: str
    add_transliteration: bool
    add_translation: bool
    compute_confidence: bool
    output_confidence_column: str


_BATCH_WRITE_LOCK = None


def resolve_output_id_column(df: pl.DataFrame, configured_column: str) -> str:
    if configured_column in df.columns:
        return configured_column
    if "oare_id" in df.columns:
        return "oare_id"
    if "id" in df.columns:
        return "id"
    raise ValueError(f"ID column not found. configured={configured_column}, columns={df.columns}")


def validate_gpu_ids(gpu_ids: list[int]) -> list[int]:
    if not torch.cuda.is_available():
        raise ValueError("`--gpu-ids` was specified, but CUDA is not available.")

    if not gpu_ids:
        raise ValueError("`--gpu-ids` requires at least one GPU ID.")

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"`--gpu-ids` contains duplicates: {gpu_ids}")

    available = torch.cuda.device_count()

    # If CUDA_VISIBLE_DEVICES is set with numeric physical IDs (e.g. "0,7"),
    # allow both physical IDs (0,7) and PyTorch logical IDs (0,1).
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env:
        visible_parts = [part.strip() for part in visible_env.split(",") if part.strip()]
        if visible_parts and all(part.isdigit() for part in visible_parts):
            visible_physical_ids = [int(part) for part in visible_parts]
            if set(gpu_ids).issubset(set(visible_physical_ids)):
                physical_to_logical = {physical_id: logical_id for logical_id, physical_id in enumerate(visible_physical_ids)}
                gpu_ids = [physical_to_logical[gpu_id] for gpu_id in gpu_ids]

    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available]
    if invalid:
        raise ValueError(
            f"Invalid GPU IDs: {invalid}. available_logical_gpu_ids={list(range(available))}. "
            f"CUDA_VISIBLE_DEVICES={visible_env!r}"
        )
    return gpu_ids


def parse_gpu_ids(gpu_ids: str | None) -> list[int] | None:
    if gpu_ids is None:
        return None

    parts = [part for part in re.split(r"[,\s]+", gpu_ids.strip()) if part]
    if not parts:
        raise ValueError("`--gpu-ids` is empty. Example: `--gpu-ids 0,7`.")

    try:
        return [int(part) for part in parts]
    except ValueError as e:
        raise ValueError(f"`--gpu-ids` must be comma-separated integers. got={gpu_ids!r}") from e


def split_records(records: list[dict], num_splits: int) -> list[list[dict]]:
    base_size, extra = divmod(len(records), num_splits)
    shards = []
    start = 0
    for i in range(num_splits):
        shard_size = base_size + (1 if i < extra else 0)
        end = start + shard_size
        shards.append(records[start:end])
        start = end
    return shards


def resolve_direction_config(
    args: Args,
) -> tuple[str, str, str, str, bool]:
    if args.translation_direction == "akkadian_to_english":
        return (
            args.input_transliteration_column,
            "translation",
            "translation",
            "transliteration",
            args.apply_transliteration_char_normalization,
        )
    if args.translation_direction == "english_to_akkadian":
        return (
            args.input_translation_column,
            args.input_transliteration_column,
            "transliteration",
            "translation",
            False,
        )
    raise ValueError(f"Unknown translation_direction: {args.translation_direction}")


def build_submission_output_columns(
    args: Args,
    *,
    output_id_column: str,
    prediction_output_column: str,
    source_column: str,
    source_output_column: str,
) -> list[pl.Expr]:
    output_columns = [
        pl.col(output_id_column),
        pl.col("prediction").alias(prediction_output_column),
    ]
    if args.add_transliteration and source_output_column == "transliteration":
        output_columns.append(pl.col(source_column).alias("transliteration"))
    if args.add_translation and source_output_column == "translation":
        output_columns.append(pl.col(source_column).alias("translation"))
    if args.compute_confidence:
        output_columns.append(pl.col(args.output_confidence_column))
    return output_columns


def get_submission_output_column_names(batch_output_config: BatchOutputConfig) -> list[str]:
    column_names = [
        batch_output_config.output_id_column,
        batch_output_config.prediction_output_column,
    ]
    if batch_output_config.add_transliteration and batch_output_config.source_output_column == "transliteration":
        column_names.append("transliteration")
    if batch_output_config.add_translation and batch_output_config.source_output_column == "translation":
        column_names.append("translation")
    if batch_output_config.compute_confidence:
        column_names.append(batch_output_config.output_confidence_column)
    return column_names


def build_submission_output_batch_df(
    batch_output_config: BatchOutputConfig,
    batch_records: list[dict],
    predictions: list[str],
    confidence_values: list[float] | None,
) -> pl.DataFrame:
    data = {
        batch_output_config.output_id_column: [
            record[batch_output_config.output_id_column] for record in batch_records
        ],
        batch_output_config.prediction_output_column: predictions,
    }
    if batch_output_config.add_transliteration and batch_output_config.source_output_column == "transliteration":
        data["transliteration"] = [record[batch_output_config.source_column] for record in batch_records]
    if batch_output_config.add_translation and batch_output_config.source_output_column == "translation":
        data["translation"] = [record[batch_output_config.source_column] for record in batch_records]
    if batch_output_config.compute_confidence:
        if confidence_values is None:
            raise ValueError("confidence_values must be provided when compute_confidence=True.")
        data[batch_output_config.output_confidence_column] = confidence_values
    return pl.DataFrame(data)


def append_csv_rows(path: Path, df: pl.DataFrame) -> None:
    include_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fp:
        df.write_csv(fp, include_header=include_header)


def write_submission_batch(
    batch_output_config: BatchOutputConfig,
    records_by_row_id: dict[int, dict],
    row_ids: list[int],
    predictions: list[str],
    confidence_values: list[float] | None = None,
) -> None:
    batch_records = [records_by_row_id[row_id] for row_id in row_ids]
    batch_df = build_submission_output_batch_df(
        batch_output_config=batch_output_config,
        batch_records=batch_records,
        predictions=predictions,
        confidence_values=confidence_values,
    )
    output_path = Path(batch_output_config.output_path)
    if _BATCH_WRITE_LOCK is None:
        append_csv_rows(output_path, batch_df)
        return

    with _BATCH_WRITE_LOCK:
        append_csv_rows(output_path, batch_df)


def run_inference_shard(
    args: Args,
    records: list[dict],
    device: str,
    *,
    batch_output_config: BatchOutputConfig | None = None,
    collect_outputs: bool = True,
    show_progress: bool,
    progress_desc: str | None = None,
    progress_position: int = 0,
) -> tuple[list[int], list[str], list[float]]:
    if not records:
        return [], [], []

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8)

    infer_df = pl.DataFrame(records).select("__row_id", "input_text")

    def tokenize_fn(example):
        row_id = example["__row_id"]
        tokenized = tokenizer(
            example["input_text"],
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        )

        input_token_len = len(tokenized["input_ids"])
        sample_max_new_tokens = max(1, math.ceil(input_token_len * args.generation_max_new_tokens_input_ratio))
        tokenized["__row_id"] = row_id
        tokenized["__input_token_len"] = input_token_len
        tokenized["__sample_max_new_tokens"] = sample_max_new_tokens
        return tokenized

    infer_dataset = Dataset.from_polars(infer_df)
    map_kwargs = {"batched": False, "remove_columns": infer_df.columns}
    if args.preprocess_num_proc > 1:
        map_kwargs["num_proc"] = args.preprocess_num_proc
    infer_dataset = infer_dataset.map(tokenize_fn, **map_kwargs)
    infer_dataset = infer_dataset.sort("__input_token_len", reverse=True)

    generation_parameters = {
        "num_beams": args.generation_num_beams,
        "num_return_sequences": args.generation_num_return_sequences,
        "do_sample": args.generation_do_sample,
        "length_penalty": args.generation_length_penalty,
        "early_stopping": args.generation_early_stopping,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collator,
    )

    records_by_row_id = None
    if batch_output_config is not None:
        records_by_row_id = {record["__row_id"]: record for record in records}

    decoded_preds_list = []
    decoded_row_ids = []
    confidence_list = []
    with torch.inference_mode():
        for batch in tqdm(
            infer_loader,
            disable=not show_progress,
            desc=progress_desc,
            position=progress_position,
            dynamic_ncols=True,
        ):
            row_ids = torch.as_tensor(batch.pop("__row_id")).tolist()
            batch_max_new_tokens = int(torch.as_tensor(batch.pop("__sample_max_new_tokens")).max().item())
            batch.pop("__input_token_len")
            batch_confidence: list[float] | None = None

            batch = batch.to(device)
            preds = model.generate(
                **batch,
                **generation_parameters,
                max_new_tokens=min(args.generation_max_new_tokens, batch_max_new_tokens),
            )

            if args.compute_confidence:
                decoder_input_ids = preds[:, :-1]
                labels = preds[:, 1:]
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                ).logits

                token_log_probs = torch.log_softmax(logits, dim=-1)
                chosen_token_log_probs = torch.gather(token_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    valid_mask = torch.ones_like(labels, dtype=torch.bool)
                else:
                    valid_mask = labels.ne(pad_token_id)

                token_counts = valid_mask.sum(dim=-1).clamp_min(1)
                avg_log_prob = (chosen_token_log_probs * valid_mask).sum(dim=-1) / token_counts
                batch_confidence = torch.exp(avg_log_prob).to("cpu").tolist()

            preds = preds.to("cpu")
            decoded_preds = tokenizer.batch_decode(
                preds,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            num_return_sequences = args.generation_num_return_sequences
            if num_return_sequences == 1:
                batch_selected_row_ids = row_ids
                batch_selected_preds = decoded_preds
            else:
                if len(decoded_preds) != len(row_ids) * num_return_sequences:
                    raise RuntimeError(
                        "Unexpected generated sequence shape: "
                        f"decoded={len(decoded_preds)}, batch={len(row_ids)}, "
                        f"num_return_sequences={num_return_sequences}"
                    )

                batch_selected_row_ids = []
                batch_selected_preds = []
                for i, row_id in enumerate(row_ids):
                    start = i * num_return_sequences
                    end = start + num_return_sequences
                    candidates = decoded_preds[start:end]

                    if args.rerank_strategy == "first":
                        best_pred = candidates[0]
                    elif args.rerank_strategy == "mbr_chrf":
                        utilities = []
                        for cand in candidates:
                            utility = sum(
                                sacrebleu.sentence_chrf(cand, [other], word_order=2).score for other in candidates
                            ) / len(candidates)
                            utilities.append(utility)
                        best_pred = candidates[int(max(range(len(utilities)), key=lambda k: utilities[k]))]
                    else:
                        raise ValueError(f"Unknown rerank_strategy: {args.rerank_strategy}")

                    batch_selected_row_ids.append(row_id)
                    batch_selected_preds.append(best_pred)

            if batch_output_config is not None:
                write_submission_batch(
                    batch_output_config=batch_output_config,
                    records_by_row_id=records_by_row_id,
                    row_ids=batch_selected_row_ids,
                    predictions=batch_selected_preds,
                    confidence_values=batch_confidence,
                )

            if collect_outputs:
                decoded_row_ids.extend(batch_selected_row_ids)
                decoded_preds_list.extend(batch_selected_preds)
                if batch_confidence is not None:
                    confidence_list.extend(batch_confidence)

    return decoded_row_ids, decoded_preds_list, confidence_list


def initialize_worker_state(tqdm_lock, batch_write_lock=None) -> None:
    tqdm.set_lock(tqdm_lock)
    global _BATCH_WRITE_LOCK
    _BATCH_WRITE_LOCK = batch_write_lock


def run_inference_shard_on_gpu(
    args: Args,
    records: list[dict],
    gpu_id: int,
    progress_position: int,
    batch_output_config: BatchOutputConfig | None = None,
    collect_outputs: bool = True,
) -> tuple[list[int], list[str], list[float]]:
    return run_inference_shard(
        args=args,
        records=records,
        device=f"cuda:{gpu_id}",
        batch_output_config=batch_output_config,
        collect_outputs=collect_outputs,
        show_progress=True,
        progress_desc=f"gpu {gpu_id}",
        progress_position=progress_position,
    )


def main(args: Args):
    seed_everything(42)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.generation_num_return_sequences > args.generation_num_beams:
        raise ValueError(
            "`generation_num_return_sequences` must be <= `generation_num_beams` "
            f"(got {args.generation_num_return_sequences} > {args.generation_num_beams})."
        )
    if args.compute_confidence and args.generation_num_return_sequences > 1:
        raise ValueError("`compute_confidence=True` currently supports only `generation_num_return_sequences=1`.")

    parsed_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if parsed_gpu_ids is None:
        devices: list[str] = ["cuda" if torch.cuda.is_available() else "cpu"]
        validated_gpu_ids: list[int] = []
    else:
        validated_gpu_ids = validate_gpu_ids(parsed_gpu_ids)
        devices = [f"cuda:{gpu_id}" for gpu_id in validated_gpu_ids]

    raw_df = pl.read_csv(args.input_path)
    if args.fold is not None:
        if "fold" not in raw_df.columns:
            raise ValueError("`--fold` was specified, but `fold` column is missing in input_path.")
        raw_df = raw_df.filter(pl.col("fold") == args.fold)

    (
        source_column,
        target_column,
        prediction_output_column,
        source_output_column,
        apply_source_normalization,
    ) = resolve_direction_config(args)

    if source_column not in raw_df.columns:
        raise ValueError(f"Input column '{source_column}' not found. columns={raw_df.columns}")

    input_text_expr = pl.coalesce([pl.col(source_column).cast(pl.Utf8), pl.lit("")])

    if apply_source_normalization:
        input_text_expr = input_text_expr.map_elements(normalize_transliteration, return_dtype=pl.String)

    processed = raw_df.with_columns(input_text_expr.alias("input_text")).with_row_index("__row_id")

    if args.dryrun:
        processed = processed.sample(128, seed=42)

    has_target = target_column in processed.columns
    if has_target:
        processed = processed.with_columns(pl.col(target_column).cast(pl.Utf8).fill_null("").alias("target_text"))
    if args.write_output_per_batch and has_target:
        raise ValueError("`write_output_per_batch=True` supports only inputs without a target column.")

    output_id_column = None
    batch_output_config = None
    collect_outputs = True
    record_columns = ["__row_id", "input_text"]
    if not has_target:
        output_id_column = resolve_output_id_column(processed, args.output_id_column)
        if args.write_output_per_batch:
            batch_output_config = BatchOutputConfig(
                output_path=args.output_path,
                output_id_column=output_id_column,
                prediction_output_column=prediction_output_column,
                source_column=source_column,
                source_output_column=source_output_column,
                add_transliteration=args.add_transliteration,
                add_translation=args.add_translation,
                compute_confidence=args.compute_confidence,
                output_confidence_column=args.output_confidence_column,
            )
            collect_outputs = False
            extra_record_columns = [output_id_column]
            if args.add_transliteration and source_output_column == "transliteration":
                extra_record_columns.append(source_column)
            if args.add_translation and source_output_column == "translation":
                extra_record_columns.append(source_column)
            for column in extra_record_columns:
                if column not in record_columns:
                    record_columns.append(column)
            if output_path.exists():
                output_path.unlink()

    records = processed.select(*record_columns).to_dicts()
    decoded_row_ids = []
    decoded_preds_list = []
    confidence_list = []

    if len(devices) == 1:
        shard_row_ids, shard_preds, shard_confidences = run_inference_shard(
            args=args,
            records=records,
            device=devices[0],
            batch_output_config=batch_output_config,
            collect_outputs=collect_outputs,
            show_progress=True,
        )
        decoded_row_ids.extend(shard_row_ids)
        decoded_preds_list.extend(shard_preds)
        confidence_list.extend(shard_confidences)
    else:
        shards = split_records(records, len(validated_gpu_ids))
        non_empty_jobs = [(gpu_id, shard) for gpu_id, shard in zip(validated_gpu_ids, shards) if shard]
        if non_empty_jobs:
            mp_context = mp.get_context("spawn")
            tqdm_lock = mp_context.RLock()
            batch_write_lock = mp_context.RLock() if batch_output_config is not None else None
            with ProcessPoolExecutor(
                max_workers=len(non_empty_jobs),
                mp_context=mp_context,
                initializer=initialize_worker_state,
                initargs=(tqdm_lock, batch_write_lock),
            ) as executor:
                futures = [
                    executor.submit(
                        run_inference_shard_on_gpu,
                        args,
                        shard,
                        gpu_id,
                        progress_position,
                        batch_output_config,
                        collect_outputs,
                    )
                    for progress_position, (gpu_id, shard) in enumerate(non_empty_jobs)
                ]
                for future in futures:
                    shard_row_ids, shard_preds, shard_confidences = future.result()
                    decoded_row_ids.extend(shard_row_ids)
                    decoded_preds_list.extend(shard_preds)
                    confidence_list.extend(shard_confidences)

    output_df = None
    if collect_outputs:
        pred_df = pl.DataFrame({"__row_id": decoded_row_ids, "prediction": decoded_preds_list})
        if args.compute_confidence:
            pred_df = pred_df.with_columns(pl.Series(args.output_confidence_column, confidence_list))
        output_df = processed.join(pred_df, on="__row_id", how="left").sort("__row_id").drop("__row_id")

    if has_target:
        if output_df is None:
            raise RuntimeError("output_df must be available when target_text exists.")
        ordered_preds = output_df.get_column("prediction").to_list()
        refs = output_df.get_column("target_text").to_list()
        raw_bleu = sacrebleu.corpus_bleu(ordered_preds, [refs])
        raw_chrf = sacrebleu.corpus_chrf(ordered_preds, [refs], word_order=2)
        raw_geo_mean = math.sqrt(raw_bleu.score * raw_chrf.score)
        print(f"BLEU: {raw_bleu.score:.2f}, CHRF: {raw_chrf.score:.2f}, GeoMean: {raw_geo_mean:.2f}")

        raw_sample_bleu_scores = [sacrebleu.sentence_bleu(pred, [label]).score for pred, label in zip(ordered_preds, refs)]
        raw_sample_chrf_scores = [
            sacrebleu.sentence_chrf(pred, [label], word_order=2).score for pred, label in zip(ordered_preds, refs)
        ]
        raw_sample_geo_means = [
            math.sqrt(bleu_score * chrf_score) for bleu_score, chrf_score in zip(raw_sample_bleu_scores, raw_sample_chrf_scores)
        ]

        sample_score_columns = ["input_text", "target_text", "prediction"]
        if args.add_transliteration and source_output_column == "transliteration":
            sample_score_columns.append(pl.col(source_column).alias("transliteration"))
        if args.add_translation and source_output_column == "translation":
            sample_score_columns.append(pl.col(source_column).alias("translation"))
        if args.compute_confidence:
            sample_score_columns.append(args.output_confidence_column)

        output_df.select(*sample_score_columns).with_columns(
            pl.Series("sample_bleu", raw_sample_bleu_scores),
            pl.Series("sample_chrf", raw_sample_chrf_scores),
            pl.Series("sample_geo_mean", raw_sample_geo_means),
        ).write_csv(args.sample_scores_path)
    else:
        if output_id_column is None:
            raise RuntimeError("output_id_column must be available for submission output.")
        if args.write_output_per_batch:
            if batch_output_config is None:
                raise RuntimeError("batch_output_config must be available when write_output_per_batch=True.")
            if not output_path.exists():
                pl.DataFrame({column: [] for column in get_submission_output_column_names(batch_output_config)}).write_csv(
                    output_path
                )
        else:
            if output_df is None:
                raise RuntimeError("output_df must be available when write_output_per_batch=False.")
            output_columns = build_submission_output_columns(
                args=args,
                output_id_column=output_id_column,
                prediction_output_column=prediction_output_column,
                source_column=source_column,
                source_output_column=source_output_column,
            )
            output_df.select(*output_columns).write_csv(args.output_path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
