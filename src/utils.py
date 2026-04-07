import json
from typing import Any

import numpy as np
import torch


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def tokenize_and_truncate(
    batch: dict,
    tokenizer: Any,
    max_length: int,
    stats: dict[str, int] | None = None,
) -> dict:
    """Tokenize without truncation, collect stats, then truncate to max_length."""
    result = tokenizer(
        batch["input_text"],
        text_target=batch["target_text"],
        truncation=False,
        add_special_tokens=True,
    )
    if stats is not None:
        for src_ids, tgt_ids in zip(result["input_ids"], result["labels"]):
            stats["total"] += 1
            src_over = len(src_ids) > max_length
            tgt_over = len(tgt_ids) > max_length
            stats["source"] += int(src_over)
            stats["target"] += int(tgt_over)
            stats["either"] += int(src_over or tgt_over)

    result["__src_len"] = [len(ids) for ids in result["input_ids"]]
    result["__tgt_len"] = [len(ids) for ids in result["labels"]]
    result["input_ids"] = [ids[:max_length] for ids in result["input_ids"]]
    result["attention_mask"] = [mask[:max_length] for mask in result["attention_mask"]]
    result["labels"] = [ids[:max_length] for ids in result["labels"]]
    return result


def _tokenize_and_truncate_for_map(
    batch: dict,
    tokenizer_name_or_path: str,
    max_length: int,
    _tokenizer_cache: dict[str, Any] = {},
) -> dict:
    """Pickle-safe wrapper for use with Dataset.map(num_proc>1)."""
    if tokenizer_name_or_path not in _tokenizer_cache:
        from transformers import AutoTokenizer

        _tokenizer_cache[tokenizer_name_or_path] = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path
        )

    tokenizer = _tokenizer_cache[tokenizer_name_or_path]
    return tokenize_and_truncate(batch, tokenizer, max_length, stats=None)


def compute_truncation_stats(dataset: Any, max_length: int) -> dict[str, int]:
    """Compute truncation stats from __src_len / __tgt_len columns."""
    src_lens = dataset["__src_len"]
    tgt_lens = dataset["__tgt_len"]
    total = len(src_lens)
    source = sum(1 for length in src_lens if length > max_length)
    target = sum(1 for length in tgt_lens if length > max_length)
    either = sum(
        1
        for src_len, tgt_len in zip(src_lens, tgt_lens)
        if src_len > max_length or tgt_len > max_length
    )
    return {"source": source, "target": target, "either": either, "total": total}


def print_truncation_stats(*args) -> None:
    """Support both stats-based and tokenizer-based truncation reporting."""
    if len(args) == 3:
        split_name, stats, max_length = args
    elif len(args) == 5:
        tokenizer, split_name, input_texts, target_texts, max_length = args
        source_input_ids = tokenizer(
            input_texts,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        target_input_ids = tokenizer(
            text_target=target_texts,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        stats = {
            "source": sum(len(ids) > max_length for ids in source_input_ids),
            "target": sum(len(ids) > max_length for ids in target_input_ids),
            "either": sum(
                (len(src_ids) > max_length) or (len(tgt_ids) > max_length)
                for src_ids, tgt_ids in zip(
                    source_input_ids, target_input_ids, strict=True
                )
            ),
            "total": len(source_input_ids),
        }
    else:
        raise TypeError(
            "print_truncation_stats expects either "
            "(split_name, stats, max_length) or "
            "(tokenizer, split_name, input_texts, target_texts, max_length)."
        )

    total = stats["total"]
    if total == 0:
        print(f"[truncation][{split_name}] total=0 max_length={max_length}")
        return

    print(
        f"[truncation][{split_name}] total={total} max_length={max_length} "
        f"source={stats['source']} ({stats['source'] / total:.2%}) "
        f"target={stats['target']} ({stats['target'] / total:.2%}) "
        f"either={stats['either']} ({stats['either'] / total:.2%})"
    )
