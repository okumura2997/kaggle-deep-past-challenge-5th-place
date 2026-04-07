#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro

from preprocess import build_published_texts_without_oare_ids


@dataclass
class Args:
    train_csv_path: Path = Path("./input/deep-past-initiative-machine-translation/train.csv")
    published_csv_path: Path = Path("./input/deep-past-initiative-machine-translation/published_texts.csv")
    extracted_translation_pairs_csv_path: Path = Path(
        "./data/extract_excavation_translation_pairs_from_locations/"
        "translations_by_record_merged_processed_en.csv"
    )
    output_without_train_csv: Path = Path("./data/published_texts_without_train.csv")
    output_without_train_or_extracted_csv: Path = Path("./data/published_texts_without_train_or_extracted.csv")
    overwrite_output: bool = False


def ensure_safe_output(output_csv: Path, overwrite_output: bool) -> None:
    if output_csv.exists() and not overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_csv}. "
            "Pass --overwrite-output to replace it."
        )


def load_unique_non_empty_oare_ids(csv_path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(csv_path)
        .select("oare_id")
        .filter(
            pl.col("oare_id").is_not_null()
            & (pl.col("oare_id").str.strip_chars() != "")
        )
        .unique(subset=["oare_id"], keep="first", maintain_order=True)
    )


def main(args: Args) -> None:
    ensure_safe_output(args.output_without_train_csv, args.overwrite_output)
    ensure_safe_output(args.output_without_train_or_extracted_csv, args.overwrite_output)

    args.output_without_train_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_without_train_or_extracted_csv.parent.mkdir(parents=True, exist_ok=True)

    published_texts = pl.read_csv(args.published_csv_path, ignore_errors=True)
    train_oare_ids = load_unique_non_empty_oare_ids(args.train_csv_path)
    extracted_translation_pairs_oare_ids = load_unique_non_empty_oare_ids(
        args.extracted_translation_pairs_csv_path
    )

    published_without_train = build_published_texts_without_oare_ids(
        published_texts,
        train_oare_ids,
    )
    published_without_train.write_csv(args.output_without_train_csv)

    published_without_train_or_extracted = build_published_texts_without_oare_ids(
        published_texts,
        pl.concat([train_oare_ids, extracted_translation_pairs_oare_ids]),
    )
    published_without_train_or_extracted.write_csv(
        args.output_without_train_or_extracted_csv
    )

    print(
        "Wrote"
        f" {len(published_without_train):,} rows to {args.output_without_train_csv}"
    )
    print(
        "Wrote"
        " "
        f"{len(published_without_train_or_extracted):,} rows to "
        f"{args.output_without_train_or_extracted_csv}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
