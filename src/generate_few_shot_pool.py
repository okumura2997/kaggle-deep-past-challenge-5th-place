#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import tyro


OUTPUT_FIELDNAMES = ["oare_id", "transliteration", "translation", "fold"]


@dataclass
class Args:
    train_processed_path: Path = Path("./data/train_processed.csv")
    extracted_translation_pairs_path: Path = Path(
        "./data/extract_excavation_translation_pairs_from_locations/"
        "translations_by_record_merged_processed_en.csv"
    )
    output_path: Path = Path("./data/few_shot_pool.csv")
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


def is_false_like(value: object) -> bool:
    if isinstance(value, bool):
        return value is False
    if value is None:
        return False
    return str(value).strip().lower() == "false"


def has_non_empty_text(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def first_non_empty_value(*values: object) -> str:
    for value in values:
        if has_non_empty_text(value):
            return str(value)
    return ""


def load_train_rows(train_processed_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with train_processed_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_columns(
            reader.fieldnames,
            set(OUTPUT_FIELDNAMES),
            train_processed_path,
        )

        rows: list[dict[str, str]] = []
        train_oare_ids: set[str] = set()
        for row in reader:
            output_row = {field: str(row.get(field, "") or "") for field in OUTPUT_FIELDNAMES}
            rows.append(output_row)

            oare_id = output_row["oare_id"].strip()
            if oare_id:
                train_oare_ids.add(oare_id)

    return rows, train_oare_ids


def load_extracted_rows(
    extracted_translation_pairs_path: Path,
    train_oare_ids: set[str],
) -> list[dict[str, str]]:
    with extracted_translation_pairs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require_columns(
            reader.fieldnames,
            {"oare_id", "transliteration"},
            extracted_translation_pairs_path,
        )

        available_columns = set(reader.fieldnames or [])
        if "translation_en" not in available_columns and "translation" not in available_columns:
            raise ValueError(
                f"{extracted_translation_pairs_path} must contain either `translation_en` or `translation`."
            )

        has_train_marker = "oare_id_in_train_processed" in available_columns
        extracted_rows: list[dict[str, str]] = []
        for row in reader:
            oare_id = str(row.get("oare_id", "") or "")
            transliteration = first_non_empty_value(row.get("transliteration"))
            translation = first_non_empty_value(
                row.get("translation_en"),
                row.get("translation"),
            )
            if not transliteration or not translation:
                continue

            if has_train_marker:
                if not is_false_like(row.get("oare_id_in_train_processed")):
                    continue
            elif oare_id.strip() and oare_id.strip() in train_oare_ids:
                continue

            extracted_rows.append(
                {
                    "oare_id": oare_id,
                    "transliteration": transliteration,
                    "translation": translation,
                    "fold": "-1",
                }
            )

    return extracted_rows


def main(args: Args) -> None:
    ensure_safe_output(args.output_path, args.overwrite_output)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows, train_oare_ids = load_train_rows(args.train_processed_path)
    extracted_rows = load_extracted_rows(
        args.extracted_translation_pairs_path,
        train_oare_ids,
    )

    with args.output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(train_rows)
        writer.writerows(extracted_rows)

    print(f"Wrote {len(train_rows) + len(extracted_rows):,} rows to {args.output_path}")
    print(f"Included {len(train_rows):,} rows from {args.train_processed_path}")
    print(
        "Appended "
        f"{len(extracted_rows):,} extracted rows from "
        f"{args.extracted_translation_pairs_path}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
