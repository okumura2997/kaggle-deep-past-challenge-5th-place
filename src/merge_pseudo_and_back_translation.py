#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import tyro


OUTPUT_FIELDNAMES = [
    "oare_id",
    "synthetic_id",
    "translation",
    "transliteration",
    "fold",
    "source_dataset",
]


@dataclass
class Args:
    pseudo_labels_path: Path = Path("./data/pseudo_labels/pseudo_labels_large.csv")
    back_translation_path: Path = Path(
        "./data/generated_translation_like_english_10k_fold0/"
        "generated_translation_like_english_back_translated.csv"
    )
    output_path: Path = Path("./data/pseudo_synthetic_merged/train_large.csv")
    pseudo_source_dataset: str = "pseudo_labels/ecru-shot-61"
    synthetic_source_dataset: str = (
        "generated_translation_like_english_10k_add_pseudo_fold0/"
        "generated_translation_like_english_back_translated"
    )
    synthetic_fold: int = 1
    overwrite_output: bool = False


def ensure_safe_output(output_path: Path, overwrite_output: bool) -> None:
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}. "
            "Pass --overwrite-output to replace it."
        )


def require_columns(fieldnames: list[str] | None, required: set[str], csv_path: Path) -> None:
    available = set(fieldnames or [])
    missing = required - available
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )


def first_non_empty_value(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip() != "":
            return text
    return ""


def load_pseudo_rows(pseudo_labels_path: Path, pseudo_source_dataset: str) -> list[dict[str, str]]:
    with pseudo_labels_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_columns(
            reader.fieldnames,
            {"oare_id", "translation", "transliteration", "fold"},
            pseudo_labels_path,
        )

        rows: list[dict[str, str]] = []
        for row in reader:
            translation = first_non_empty_value(row.get("translation"))
            transliteration = first_non_empty_value(row.get("transliteration"))
            fold = first_non_empty_value(row.get("fold"))
            if not translation or not transliteration or not fold:
                continue

            rows.append(
                {
                    "oare_id": str(row.get("oare_id", "") or ""),
                    "synthetic_id": "",
                    "translation": translation,
                    "transliteration": transliteration,
                    "fold": fold,
                    "source_dataset": pseudo_source_dataset,
                }
            )

    return rows


def load_synthetic_rows(
    back_translation_path: Path,
    synthetic_source_dataset: str,
    synthetic_fold: int,
) -> list[dict[str, str]]:
    with back_translation_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_columns(
            reader.fieldnames,
            {"synthetic_id", "translation", "transliteration"},
            back_translation_path,
        )

        rows: list[dict[str, str]] = []
        for row in reader:
            synthetic_id = first_non_empty_value(row.get("synthetic_id"))
            translation = first_non_empty_value(row.get("translation"))
            transliteration = first_non_empty_value(row.get("transliteration"))
            if not synthetic_id or not translation or not transliteration:
                continue

            rows.append(
                {
                    "oare_id": "",
                    "synthetic_id": synthetic_id,
                    "translation": translation,
                    "transliteration": transliteration,
                    "fold": str(synthetic_fold),
                    "source_dataset": synthetic_source_dataset,
                }
            )

    return rows


def main(args: Args) -> None:
    ensure_safe_output(args.output_path, args.overwrite_output)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    pseudo_rows = load_pseudo_rows(
        args.pseudo_labels_path,
        args.pseudo_source_dataset,
    )
    synthetic_rows = load_synthetic_rows(
        args.back_translation_path,
        args.synthetic_source_dataset,
        args.synthetic_fold,
    )

    with args.output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(pseudo_rows)
        writer.writerows(synthetic_rows)

    print(f"Wrote {len(pseudo_rows) + len(synthetic_rows):,} rows to {args.output_path}")
    print(f"Included {len(pseudo_rows):,} rows from {args.pseudo_labels_path}")
    print(f"Appended {len(synthetic_rows):,} rows from {args.back_translation_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
