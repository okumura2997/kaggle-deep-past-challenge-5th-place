from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import tyro
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

import wandb
from preprocess import normalize_transliteration
from utils import seed_everything


@dataclass
class Args:
    model: str = "google/byt5-base"
    run_name: Optional[str] = None
    seed: int = 42

    noise_density: float = 0.15
    mean_noise_span_length: float = 20.0
    input_length: int = 1024

    add_evacun: bool = False

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 100
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    bf16: bool = True
    gradient_checkpointing: bool = False

    eval_size: float = 0.1
    eval_strategy: str = "epoch"

    save_strategy: str = "best"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    save_total_limit: int = 1

    report_to: str = "wandb"


def compute_input_and_target_lengths(desired_input_length, noise_density, mean_noise_span_length):
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)

        input_length = num_nonnoise_tokens + num_noise_spans + 1
        target_length = num_noise_tokens + num_noise_spans + 1

        return input_length, target_length

    tokens_length = desired_input_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= desired_input_length:
        tokens_length += 1

    input_length, target_length = _tokens_length_to_inputs_length_targets_length(tokens_length)
    return tokens_length, target_length


def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    input_ids: shape (batch, seq_len)
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


@dataclass
class DataCollatorForT5SpanCorruption:
    tokenizer: any
    noise_density: float = 0.15
    mean_noise_span_length: float = 20.0
    input_length: int = 512
    target_length: int = None
    pad_token_id: int = None
    decoder_start_token_id: int = None

    def random_spans_noise_mask(self, length):
        num_noise_tokens = int(round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

        num_noise_spans = int(round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)

        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2])
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:length]

    def create_sentinel_ids(self, mask_indices):
        mask_indices = mask_indices.astype(np.int32)
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0].reshape((batch_size, -1))
        input_ids = np.concatenate([input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1)
        return input_ids

    def __call__(self, features):
        batch_input_ids = np.stack([np.array(f["input_ids"], dtype=np.int32) for f in features], axis=0)

        batch_size, seq_len = batch_input_ids.shape

        noise_masks = np.stack([self.random_spans_noise_mask(seq_len) for _ in range(batch_size)], axis=0)

        input_sentinel = self.create_sentinel_ids(noise_masks)
        label_sentinel = self.create_sentinel_ids(~noise_masks)

        inputs = self.filter_input_ids(batch_input_ids, input_sentinel)
        labels = self.filter_input_ids(batch_input_ids, label_sentinel)

        attention_mask = (inputs != self.tokenizer.pad_token_id).astype(np.int32)
        labels[labels == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)
        assert inputs.shape[-1] == self.input_length, (
            f"`inputs` are incorrectly preprocessed. `inputs` length is {inputs.shape[-1]}, but should be {self.input_length}."
        )
        assert labels.shape[-1] == self.target_length, (
            f"`labels` are incorrectly preprocessed. `labels` length is {labels.shape[-1]}, but should be {self.target_length}."
        )

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        }


def main(args: Args):
    if args.report_to == "wandb":
        wandb_init_kwargs = {
            "project": "deep-past-initiative-machine-translation",
            "job_type": "pretrain",
            "config": {"cli_args": vars(args)},
        }
        if args.run_name is not None:
            wandb_init_kwargs["name"] = args.run_name
        wandb.init(**wandb_init_kwargs)
        run_name = args.run_name or wandb.run.name
    else:
        run_name = args.run_name or "dryrun"

    seed_everything(args.seed)

    output_dir = "./outputs/pretrain/" + run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    published_texts = pl.read_csv("./input/deep-past-initiative-machine-translation/published_texts.csv", ignore_errors=True)
    transliteration = published_texts.get_column("transliteration").to_list()
    transliteration = [normalize_transliteration(text) for text in transliteration]
    if args.add_evacun:
        evacun = pl.read_csv("./data/evacun/train_processed_all.csv")
        evacun_transliteration = evacun.get_column("transliteration").drop_nulls().to_list()
        transliteration.extend(evacun_transliteration)
    transliteration = list(dict.fromkeys(transliteration))
    transliteration = np.random.default_rng(args.seed).permutation(transliteration).tolist()

    ds = Dataset.from_list([{"text": text} for text in transliteration])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=True,
            return_attention_mask=False,
        )

    tokenized_datasets = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    expanded_inputs_length, target_length = compute_input_and_target_lengths(
        args.input_length, args.noise_density, args.mean_noise_span_length
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        assert total_length >= expanded_inputs_length, (
            f"group_texts produced too-short batch: total_length={total_length}, "
            f"expanded_inputs_length={expanded_inputs_length}. "
            "Increase map batch_size or filter short batches."
        )
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=100000)

    collator = DataCollatorForT5SpanCorruption(
        tokenizer=tokenizer,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length,
        input_length=args.input_length,
        target_length=target_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    eval_dataset = None
    train_dataset = tokenized_datasets
    if args.eval_size > 0:
        split = tokenized_datasets.train_test_split(test_size=args.eval_size, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        args.eval_strategy = "no"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(Path(output_dir) / "best_model")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
