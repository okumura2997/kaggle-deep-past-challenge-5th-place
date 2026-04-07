import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import sacrebleu
import tyro
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

import wandb
from preprocess import cut_if_too_long, was_truncated
from utils import print_truncation_stats, seed_everything


@dataclass
class Args:
    model: str = "./outputs/pretrain/large-span-corruption/best_model"
    train_path: str = "./data/train_processed.csv"
    run_name: str | None = None
    fold: int | None = None
    translation_direction: Literal["akkadian_to_english", "english_to_akkadian"] = "akkadian_to_english"
    seed: int = 42
    full_fit: bool = False
    add_extracted_data: bool = False
    extracted_data_path: str = "./data/link_pairs_with_transliteration/matched_pairs_with_transliteration_processed_en_qwen.csv"
    add_hecker: bool = False
    hecker_path: str = "./data/extract_excavation_translation_pairs_from_locations_hecker/pdf_only_excavation_numbers_with_locations/translations_by_record_processed_en.csv"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 4

    train_tokenize_max_length: int = 512
    eval_tokenize_max_length: int = 8196
    learning_rate: float = 1e-4
    num_train_epochs: int = 30
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    bf16: bool = True
    gradient_checkpointing: bool = False

    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    predict_with_generate: bool = False
    generation_num_beams: int = 8
    generation_max_length: int = 512

    report_to: str = "none"

    save_strategy: str = "best"
    save_steps: int | None = None
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    save_total_limit: int = 1

    apply_cut_if_too_long: bool = False
    transliteration_to_translation_max_ratio: float = 1.7
    translation_to_transliteration_max_ratio: float = 1.7


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        if hasattr(preds, "ndim") and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        preds = preds.astype(np.int64)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, spaces_between_special_tokens=False)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, spaces_between_special_tokens=False)

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
        geo_mean = math.sqrt(bleu.score * chrf.score)

        return {"chrf": chrf.score, "bleu": bleu.score, "geo_mean": geo_mean}

    return compute_metrics


def _normalized_non_empty_string_expr(column_name: str) -> pl.Expr:
    col = pl.col(column_name).cast(pl.String, strict=False)
    return pl.when(col.is_null() | (col.str.strip_chars() == "")).then(None).otherwise(col)


def _build_extracted_text_expr(df: pl.DataFrame, canonical_col: str) -> pl.Expr:
    column_candidates = {
        "transliteration": ["transliteration"],
        "translation": ["translation_en", "translation"],
    }
    candidates = column_candidates.get(canonical_col, [canonical_col])
    available = [col for col in candidates if col in df.columns]
    if not available:
        raise ValueError(
            f"Extracted data is missing required column for `{canonical_col}`. "
            f"Tried columns: {candidates}"
        )

    return pl.coalesce([_normalized_non_empty_string_expr(col) for col in available]).alias(canonical_col)


def prepare_extracted_training_data(df: pl.DataFrame, source_col: str, target_col: str, dataset_name: str) -> pl.DataFrame:
    if "oare_id_in_train_processed" in df.columns:
        marker_dtype = df.schema["oare_id_in_train_processed"]
        if marker_dtype == pl.Boolean:
            filter_expr = pl.col("oare_id_in_train_processed") == False
        else:
            filter_expr = (
                pl.col("oare_id_in_train_processed")
                .cast(pl.String, strict=False)
                .str.strip_chars()
                .str.to_lowercase()
                == "false"
            )
        before_count = df.height
        df = df.filter(filter_expr)
        print(
            "Filtered extracted data by oare_id_in_train_processed: "
            f"dataset={dataset_name}, kept={df.height}, removed={before_count - df.height}"
        )

    return df.with_columns(
        _build_extracted_text_expr(df, source_col),
        _build_extracted_text_expr(df, target_col),
    )


def main(args: Args):
    if args.eval_steps is not None and args.eval_steps <= 0:
        raise ValueError("`--eval-steps` must be a positive integer.")
    if args.save_steps is not None and args.save_steps <= 0:
        raise ValueError("`--save-steps` must be a positive integer.")

    if args.translation_direction == "akkadian_to_english":
        source_col = "transliteration"
        target_col = "translation"
        source_to_target_ratio = args.transliteration_to_translation_max_ratio
        target_to_source_ratio = args.translation_to_transliteration_max_ratio
    else:
        source_col = "translation"
        target_col = "transliteration"
        source_to_target_ratio = args.translation_to_transliteration_max_ratio
        target_to_source_ratio = args.transliteration_to_translation_max_ratio

    run_group_name = args.run_name.split("/", 1)[0] if args.run_name is not None else None

    if args.report_to == "wandb":
        wandb_init_kwargs = {
            "project": "deep-past-initiative-machine-translation",
            "job_type": "finetune",
            "config": {"cli_args": vars(args)},
        }
        if args.run_name is not None:
            wandb_init_kwargs["name"] = args.run_name
            if args.fold is not None or args.full_fit:
                wandb_init_kwargs["group"] = run_group_name
        wandb.init(**wandb_init_kwargs)
        run_name = wandb.run.name
    else:
        run_name = "dryrun"

    seed_everything(args.seed)

    output_dir = "./outputs/finetune/" + run_name

    processed = pl.read_csv(args.train_path)
    required_processed_cols = {source_col, target_col}
    missing_processed_cols = required_processed_cols - set(processed.columns)
    if missing_processed_cols:
        raise ValueError(
            f"train_path is missing required columns for current settings: {sorted(missing_processed_cols)}. "
            "Run preprocess.py first."
        )

    processed = processed.with_columns(
        pl.col(source_col).alias("input_text"),
        pl.col(target_col).alias("target_text"),
    )
    train_columns = ["input_text", "target_text"]
    eval_columns = ["input_text", "target_text"]
    if args.full_fit:
        train_base = processed
        valid = None
    else:
        if args.fold is None:
            raise ValueError("`--fold` is required when `full_fit` is False.")
        if "fold" not in processed.columns:
            raise ValueError("`--fold` was specified, but `fold` column is missing in train_path.")
        train_base = processed.filter(pl.col("fold") != args.fold)
        valid_source = processed.filter(pl.col("fold") == args.fold)

        valid = valid_source.select(eval_columns)

    if args.add_extracted_data:
        extracted_path = args.extracted_data_path
        extracted = pl.read_csv(extracted_path)

        extracted = prepare_extracted_training_data(extracted, source_col, target_col, extracted_path)
        train_base = pl.concat([train_base, extracted], how="diagonal_relaxed")
        print(f"Added extracted data: {extracted.height} rows from {extracted_path}")

    if args.add_hecker:
        hecker_path = args.hecker_path
        hecker = pl.read_csv(hecker_path)

        # Reuse the extracted-data normalization path so `translation_en` is used as the English column.
        hecker = prepare_extracted_training_data(hecker, source_col, target_col, hecker_path)
        train_base = pl.concat([train_base, hecker], how="diagonal_relaxed")
        print(f"Added Hecker data: {hecker.height} rows from {hecker_path}")

    if args.apply_cut_if_too_long:
        train_base = train_base.with_columns(
            pl.struct([source_col, target_col])
            .map_elements(
                lambda row: cut_if_too_long(
                    row[source_col],
                    row[target_col],
                    max_ratio=source_to_target_ratio,
                ),
                return_dtype=pl.String,
            )
            .alias("__cut_source"),
            pl.struct([target_col, source_col])
            .map_elements(
                lambda row: cut_if_too_long(
                    row[target_col],
                    row[source_col],
                    max_ratio=target_to_source_ratio,
                ),
                return_dtype=pl.String,
            )
            .alias("__cut_target"),
        ).with_columns(
            pl.struct([source_col, "__cut_source"])
            .map_elements(
                lambda row: was_truncated(row[source_col], row["__cut_source"]),
                return_dtype=pl.Boolean,
            )
            .alias("__source_cut_applied"),
            pl.struct([target_col, "__cut_target"])
            .map_elements(
                lambda row: was_truncated(row[target_col], row["__cut_target"]),
                return_dtype=pl.Boolean,
            )
            .alias("__target_cut_applied"),
        )

        cut_stats = train_base.select(
            pl.len().alias("total"),
            pl.col("__source_cut_applied").sum().alias("source_cut"),
            pl.col("__target_cut_applied").sum().alias("target_cut"),
            (pl.col("__source_cut_applied") | pl.col("__target_cut_applied")).sum().alias("either_cut"),
        ).row(0, named=True)
        total = int(cut_stats["total"])
        source_cut = int(cut_stats["source_cut"])
        target_cut = int(cut_stats["target_cut"])
        either_cut = int(cut_stats["either_cut"])
        denom = total if total > 0 else 1
        print(
            f"[cut_if_too_long][train] total={total} "
            f"source_cut={source_cut} ({source_cut / denom:.2%}) "
            f"target_cut={target_cut} ({target_cut / denom:.2%}) "
            f"either_cut={either_cut} ({either_cut / denom:.2%})"
        )

        train_base = train_base.with_columns(
            pl.col("__cut_source").alias(source_col),
            pl.col("__cut_target").alias(target_col),
        ).drop(
            "__cut_source",
            "__cut_target",
            "__source_cut_applied",
            "__target_cut_applied",
        )

    # Keep training columns in sync with any source/target edits (e.g. cut_if_too_long).
    train_base = train_base.with_columns(
        pl.col(source_col).alias("input_text"),
        pl.col(target_col).alias("target_text"),
    )
    train = train_base.select(train_columns)
    train_input_texts = train.get_column("input_text").to_list()
    train_target_texts = train.get_column("target_text").to_list()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print_truncation_stats(tokenizer, "train", train_input_texts, train_target_texts, args.train_tokenize_max_length)
    if valid is not None:
        valid_input_texts = valid.get_column("input_text").to_list()
        valid_target_texts = valid.get_column("target_text").to_list()
        print_truncation_stats(tokenizer, "eval", valid_input_texts, valid_target_texts, args.eval_tokenize_max_length)

    train_dataset = Dataset.from_polars(train)
    if valid is not None:
        eval_dataset = Dataset.from_polars(valid)
    else:
        eval_dataset = None

    train_dataset = train_dataset.map(
        lambda batch: tokenizer(
            batch["input_text"],
            text_target=batch["target_text"],
            truncation=True,
            max_length=args.train_tokenize_max_length,
            add_special_tokens=True,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda batch: tokenizer(
                batch["input_text"],
                text_target=batch["target_text"],
                truncation=True,
                max_length=args.eval_tokenize_max_length,
                add_special_tokens=True,
            ),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8)

    eval_strategy = "no" if args.full_fit else args.eval_strategy
    save_strategy = "epoch" if args.full_fit and args.save_strategy == "best" else args.save_strategy
    load_best_model_at_end = False if args.full_fit else args.load_best_model_at_end
    if eval_strategy == "steps" and args.eval_steps is None:
        raise ValueError("`--eval-steps` is required when `--eval-strategy steps`.")
    if save_strategy == "steps" and args.save_steps is None:
        raise ValueError("`--save-steps` is required when `--save-strategy steps`.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.generation_max_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer) if (eval_dataset is not None and args.predict_with_generate) else None,
    )

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(Path(output_dir) / "best_model")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
