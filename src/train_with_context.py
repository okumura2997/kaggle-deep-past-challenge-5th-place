import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import sacrebleu
import torch
import torch.nn as nn
import tyro
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

import wandb
from preprocess import (
    assign_group_folds,
    compute_context_prefixes,
)
from utils import (
    _tokenize_and_truncate_for_map,
    compute_truncation_stats,
    print_truncation_stats,
    seed_everything,
    tokenize_and_truncate,
)


@dataclass
class Args:
    model: str = "google/byt5-base"
    train_processed_path: str = "data/extract_unified/all_pairs_final.csv"
    run_name: str | None = None
    fold: int | None = None
    n_folds: int = 5
    translation_direction: Literal["akkadian_to_english", "english_to_akkadian"] = (
        "akkadian_to_english"
    )
    seed: int = 42
    full_fit: bool = False
    map_num_proc: int | None = 8

    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 4

    train_tokenize_max_length: int = 2048
    eval_tokenize_max_length: int = 8196
    learning_rate: float = 1e-4
    num_train_epochs: int = 30
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 25
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    decoder_only: bool = False

    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    """When eval_strategy='steps', evaluate every this many steps."""
    predict_with_generate: bool = False
    generation_num_beams: int = 8
    generation_max_length: int = 512

    report_to: str = "none"

    save_strategy: str = "best"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    group_by_length: bool = False

    # Augmentation
    sentence_concat_prob: float = 0.0

    # Context: prepend previous sentence as context to input
    prepend_context: bool = True
    context_num_prev: int = 2
    context_max_bytes: int = (
        1024  # 0 = train_tokenize_max_length // 2; truncate oldest first
    )

    # Quantization-aware training (CTranslate2 int8 fake quantize)
    qat_int8: bool = False


import re

_MULTI_GAP = re.compile(r"(<gap>\s*){2,}")


def _line_range_sort_key(lr: str) -> int:
    """Extract the first number from a line_range string for sorting (e.g. '4-13' -> 4)."""
    m = re.match(r"(\d+)", str(lr))
    return int(m.group(1)) if m else 0


def _sort_by_excavation(df: pl.DataFrame) -> pl.DataFrame:
    """Sort DataFrame by excavation_number then line_range."""
    return df.with_columns(
        pl.col("line_range")
        .map_elements(_line_range_sort_key, return_dtype=pl.Int64)
        .alias("__lr_key")
    ).sort(["excavation_number", "__lr_key"])


def augment_sentence_concat(
    df: pl.DataFrame,
    prob: float,
    seed: int,
    context_prefixes: list[str] | None = None,
) -> pl.DataFrame:
    """Randomly concatenate adjacent row pairs within the same excavation_number.

    ``df`` must already be sorted by excavation_number then line_range.
    Consecutive ``<gap>`` tokens produced by joining are collapsed into one.
    If *context_prefixes* is given, each aug row gets the prefix of its first sentence.
    Returns **only** the augmented rows (not originals).
    """
    if prob <= 0 or df.height < 2:
        return pl.DataFrame({"input_text": [], "target_text": []})

    rng = np.random.RandomState(seed)
    inputs = df.get_column("input_text").to_list()
    targets = df.get_column("target_text").to_list()
    exc_nums = df.get_column("excavation_number").to_list()

    aug_inputs: list[str] = []
    aug_targets: list[str] = []
    for i in range(len(inputs) - 1):
        if exc_nums[i] != exc_nums[i + 1]:
            continue
        if rng.random() >= prob:
            continue
        src = f"{inputs[i]} {inputs[i + 1]}"
        tgt = f"{targets[i]} {targets[i + 1]}"
        src = _MULTI_GAP.sub("<gap> ", src).strip()
        tgt = _MULTI_GAP.sub("<gap> ", tgt).strip()
        if context_prefixes is not None:
            src = context_prefixes[i] + src
        aug_inputs.append(src)
        aug_targets.append(tgt)

    if aug_inputs:
        print(f"Sentence concat augmentation: +{len(aug_inputs)} rows")
    return pl.DataFrame({"input_text": aug_inputs, "target_text": aug_targets})


def prepend_prev_context(
    df: pl.DataFrame,
    num_prev: int = 1,
    max_bytes: int = 0,
) -> pl.DataFrame:
    """Prepend previous sentences' transliteration as context to input_text,
    within the same excavation_number sorted by line_range.

    Format (num_prev=1):
      ``[context] prev_translit [/context] current_input``

    Format (num_prev=2):
      ``[context] prev2_translit [sep] prev1_translit [/context] current_input``

    Only sentences from the same excavation_number are used as context.
    Returns DataFrame with columns (excavation_number, line_range, input_text, target_text).
    """
    if num_prev <= 0:
        return df.select(
            ["excavation_number", "line_range", "input_text", "target_text"]
        )

    df = _sort_by_excavation(df)
    exc_nums = df.get_column("excavation_number").to_list()
    inputs = df.get_column("input_text").to_list()
    targets = df.get_column("target_text").to_list()

    prefixes = compute_context_prefixes(exc_nums, inputs, num_prev, max_bytes)
    new_inputs = [prefixes[i] + inputs[i] for i in range(len(inputs))]

    return df.select(["excavation_number", "line_range"]).with_columns(
        pl.Series("input_text", new_inputs),
        pl.Series("target_text", targets),
    )


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

        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
        geo_mean = math.sqrt(bleu.score * chrf.score)

        return {"chrf": chrf.score, "bleu": bleu.score, "geo_mean": geo_mean}

    return compute_metrics


class _FakeQuantizeInt8(torch.autograd.Function):
    """Per-row symmetric int8 fake quantize matching CTranslate2's scheme.

    Forward: quantize → clamp → dequantize (simulates int8 rounding error).
    Backward: straight-through estimator (gradient passes through unchanged).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        scale = 127.0 / weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        w_q = (weight * scale).round().clamp(-128, 127)
        return w_q / scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _fake_quantize_int8(weight: torch.Tensor) -> torch.Tensor:
    return _FakeQuantizeInt8.apply(weight)


class FakeQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear that applies CTranslate2-compatible
    int8 fake quantization on every forward pass."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.weight = original.weight  # keeps original Parameter (trainable)
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            x, _fake_quantize_int8(self.weight), self.bias
        )


class FakeQuantEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding that applies CTranslate2-compatible
    int8 fake quantization on every forward pass."""

    def __init__(self, original: nn.Embedding):
        super().__init__()
        self.weight = original.weight
        self.padding_idx = original.padding_idx
        self.max_norm = original.max_norm
        self.norm_type = original.norm_type
        self.scale_grad_by_freq = original.scale_grad_by_freq
        self.sparse = original.sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(
            x,
            _fake_quantize_int8(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


# Embedding layers that CTranslate2 does NOT quantize (kept as-is).
_QAT_SKIP_EMBEDDINGS = {"relative_attention_bias"}


def apply_qat_int8(model: nn.Module) -> tuple[int, int]:
    """Replace nn.Linear and nn.Embedding with fake-quantized versions.
    Skips embeddings in _QAT_SKIP_EMBEDDINGS (e.g. relative_attention_bias).
    Returns (num_linear_replaced, num_embedding_replaced)."""
    n_linear = 0
    n_embed = 0
    for name, module in list(model.named_modules()):
        attr = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear):
            replacement = FakeQuantLinear(module)
            n_linear += 1
        elif isinstance(module, nn.Embedding) and attr not in _QAT_SKIP_EMBEDDINGS:
            replacement = FakeQuantEmbedding(module)
            n_embed += 1
        else:
            continue
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(parent, attr, replacement)
    return n_linear, n_embed


def revert_qat_int8(model: nn.Module) -> int:
    """Revert FakeQuantLinear/FakeQuantEmbedding back to originals for clean save_pretrained."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, FakeQuantLinear):
            restored = nn.Linear(
                module.weight.shape[1],
                module.weight.shape[0],
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            restored.weight = module.weight
            restored.bias = module.bias
        elif isinstance(module, FakeQuantEmbedding):
            restored = nn.Embedding(
                module.weight.shape[0],
                module.weight.shape[1],
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            restored.weight = module.weight
        else:
            continue
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        attr = parts[-1]
        setattr(parent, attr, restored)
        count += 1
    return count


class SaveLastModelCallback(TrainerCallback):
    """Overwrites `last_model/` at every epoch end; the final write is the last-epoch snapshot."""

    def __init__(self, save_dir, tokenizer, *, qat_int8: bool = False):
        self.save_dir = Path(save_dir)
        self.tokenizer = tokenizer
        self.qat_int8 = qat_int8

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if self.qat_int8:
            revert_qat_int8(model)
        save_path = self.save_dir / "last_model"
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        if self.qat_int8:
            apply_qat_int8(model)


def main(args: Args):
    if args.translation_direction == "akkadian_to_english":
        source_col = "transliteration"
        target_col = "translation"
    else:
        source_col = "translation"
        target_col = "transliteration"

    run_group_name = (
        args.run_name.split("/", 1)[0] if args.run_name is not None else None
    )

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

    # 1. Load data and assign folds
    all_data = pl.read_csv(args.train_processed_path)

    # GroupKFold by excavation_number to prevent data leakage
    groups = all_data.get_column("excavation_number").to_list()
    folds = assign_group_folds(groups, n_splits=args.n_folds, seed=args.seed)
    all_data = all_data.with_columns(pl.Series("fold", folds).cast(pl.Int8))

    # Helper columns used by augmentation & context
    _select_cols = [
        pl.col("excavation_number"),
        pl.col("line_range"),
        pl.col(source_col).alias("input_text"),
        pl.col(target_col).alias("target_text"),
    ]

    # 2. Split train / eval
    if not args.full_fit:
        if args.fold is None:
            raise ValueError("`--fold` is required when `full_fit` is False.")
        train_base = all_data.filter(pl.col("fold") != args.fold).select(_select_cols)
        valid = all_data.filter(pl.col("fold") == args.fold).select(_select_cols)
        print(f"Eval data: {valid.height} rows (fold {args.fold})")
    else:
        train_base = all_data.select(_select_cols)
        valid = None

    # 3. Sort train_base (shared order for context + concat aug)
    train_base = _sort_by_excavation(train_base)

    # Compute context prefixes from raw inputs
    if args.prepend_context and args.context_num_prev > 0:
        ctx_max = (
            args.context_max_bytes
            if args.context_max_bytes > 0
            else args.train_tokenize_max_length // 2
        )
        _exc = train_base.get_column("excavation_number").to_list()
        _inp = train_base.get_column("input_text").to_list()
        ctx_prefixes = compute_context_prefixes(
            _exc, _inp, args.context_num_prev, ctx_max
        )
    else:
        ctx_prefixes = None

    # 4. Sentence concat augmentation (on raw inputs; context prefix applied to aug rows)
    aug_rows = augment_sentence_concat(
        train_base,
        args.sentence_concat_prob,
        args.seed,
        context_prefixes=ctx_prefixes,
    )

    # 5. Apply context to original rows
    if ctx_prefixes is not None:
        orig_inputs = train_base.get_column("input_text").to_list()
        orig_targets = train_base.get_column("target_text").to_list()
        ctx_inputs = [ctx_prefixes[i] + orig_inputs[i] for i in range(len(orig_inputs))]
        train = pl.DataFrame({"input_text": ctx_inputs, "target_text": orig_targets})
    else:
        train = train_base.select(["input_text", "target_text"])

    # Merge originals + aug rows
    if aug_rows.height > 0:
        train = pl.concat([train, aug_rows], how="vertical")

    # Apply context to eval (same max as train for consistency)
    if args.prepend_context and valid is not None:
        valid = prepend_prev_context(valid, args.context_num_prev, ctx_max)

    # Drop metadata columns for eval
    if valid is not None:
        valid = valid.select(["input_text", "target_text"])

    print(f"Training data: {train.height} rows from {args.train_processed_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = Dataset.from_polars(train)
    if valid is not None:
        eval_dataset = Dataset.from_polars(valid)
    else:
        eval_dataset = None

    num_proc = args.map_num_proc
    if num_proc is not None and num_proc > 1:
        from functools import partial

        _map_fn_train = partial(
            _tokenize_and_truncate_for_map,
            tokenizer_name_or_path=args.model,
            max_length=args.train_tokenize_max_length,
        )
        train_dataset = train_dataset.map(
            _map_fn_train,
            batched=True,
            num_proc=num_proc,
            remove_columns=train_dataset.column_names,
        )
        train_trunc_stats = compute_truncation_stats(
            train_dataset, args.train_tokenize_max_length
        )
        train_dataset = train_dataset.remove_columns(["__src_len", "__tgt_len"])
    else:
        train_trunc_stats: dict[str, int] = {
            "source": 0,
            "target": 0,
            "either": 0,
            "total": 0,
        }
        train_dataset = train_dataset.map(
            lambda batch: tokenize_and_truncate(
                batch, tokenizer, args.train_tokenize_max_length, train_trunc_stats
            ),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        train_dataset = train_dataset.remove_columns(["__src_len", "__tgt_len"])
    print_truncation_stats("train", train_trunc_stats, args.train_tokenize_max_length)

    if eval_dataset is not None:
        if num_proc is not None and num_proc > 1:
            _map_fn_eval = partial(
                _tokenize_and_truncate_for_map,
                tokenizer_name_or_path=args.model,
                max_length=args.eval_tokenize_max_length,
            )
            eval_dataset = eval_dataset.map(
                _map_fn_eval,
                batched=True,
                num_proc=num_proc,
                remove_columns=eval_dataset.column_names,
            )
            eval_trunc_stats = compute_truncation_stats(
                eval_dataset, args.eval_tokenize_max_length
            )
            eval_dataset = eval_dataset.remove_columns(["__src_len", "__tgt_len"])
        else:
            eval_trunc_stats: dict[str, int] = {
                "source": 0,
                "target": 0,
                "either": 0,
                "total": 0,
            }
            eval_dataset = eval_dataset.map(
                lambda batch: tokenize_and_truncate(
                    batch, tokenizer, args.eval_tokenize_max_length, eval_trunc_stats
                ),
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
            eval_dataset = eval_dataset.remove_columns(["__src_len", "__tgt_len"])
        print_truncation_stats("eval", eval_trunc_stats, args.eval_tokenize_max_length)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.qat_int8:
        n_linear, n_embed = apply_qat_int8(model)
        print(f"QAT int8: replaced {n_linear} Linear + {n_embed} Embedding layers")
    if args.decoder_only:
        if not hasattr(model, "get_encoder"):
            raise ValueError(
                "`--decoder-only` is enabled, but model does not expose an encoder."
            )
        encoder = model.get_encoder()
        for param in encoder.parameters():
            param.requires_grad = False
        print("Enabled decoder-only training: encoder parameters are frozen.")
    if args.gradient_checkpointing:
        model.config.use_cache = False
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", pad_to_multiple_of=8
    )

    eval_strategy = "no" if args.full_fit else args.eval_strategy
    save_strategy = (
        "epoch"
        if args.full_fit and args.save_strategy == "best"
        else args.save_strategy
    )
    load_best_model_at_end = False if args.full_fit else args.load_best_model_at_end

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        fp16=args.fp16,
        save_strategy=save_strategy,
        save_steps=args.eval_steps,
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
        dataloader_num_workers=4,
        torch_compile=False,
        group_by_length=args.group_by_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer)
        if (eval_dataset is not None and args.predict_with_generate)
        else None,
    )

    trainer.add_callback(
        SaveLastModelCallback(output_dir, tokenizer, qat_int8=args.qat_int8)
    )
    trainer.train()
    if args.qat_int8:
        revert_qat_int8(model)
    trainer.save_model(Path(output_dir) / "best_model")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)