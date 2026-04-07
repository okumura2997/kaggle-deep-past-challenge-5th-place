#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro
from preprocess import normalize_transliteration


@dataclass
class Args:
    published_csv_path: Path = Path("./input/deep-past-initiative-machine-translation/published_texts.csv")
    output_csv: Path = Path("./data/published_texts_akt_subset.csv")
    expected_rows: int = 2800
    overwrite_output: bool = False


EXCAVATION_NO_OVERRIDES = {
    # The source excavation_no is truncated, but aliases/label retain the full identifier.
    "017f25c5-0f92-4631-957a-fc98000f078b": "Kt 92/k 321",
}

EXCAVATION_NO_TEXT_OVERRIDES = {
    "Kt 91/k 518 envelope": "Kt 91/k 518A",
    "Kt 91/k 518 tablet": "Kt 91/k 518B",
}

AKT1_IDENTIFIER_RE = re.compile(
    r"(?i)\bAKT\s+1\s+(?P<number>\d+[a-z]?)(?:\s+(?P<suffix>env(?:elope)?|tab(?:let)?|zarf))?\b"
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_akt1_identifier(*texts: str) -> str:
    for text in texts:
        match = AKT1_IDENTIFIER_RE.search(text or "")
        if match is None:
            continue

        number = normalize_space(match.group("number"))
        if not number:
            continue
        return f"No.{number}"

    return ""


def resolve_excavation_no(row: pd.Series) -> str:
    excavation_no = normalize_space(row["excavation_no"])
    akt1_identifier = extract_akt1_identifier(
        row["aliases"],
        row["label"],
        row["publication_catalog"],
    )
    if akt1_identifier:
        return akt1_identifier

    searchable_text = " | ".join(
        normalize_space(row[column]) for column in ("excavation_no", "aliases", "label", "publication_catalog")
    ).lower()
    for source_text, target in EXCAVATION_NO_TEXT_OVERRIDES.items():
        if source_text.lower() in searchable_text:
            return target
    if excavation_no:
        return excavation_no
    return ""


def extract_akt_volumes(*texts: str) -> list[str]:
    akt_volume_re = re.compile(r"(?i)\bAKT\s+(\d+[a-z]?)(?:\s+(19\d{2}|20\d{2}))?")
    volumes: list[str] = []
    for text in texts:
        for match in akt_volume_re.finditer(text or ""):
            base = match.group(1).lower()
            year = match.group(2)
            volume = f"AKT {base}" + (f" {year}" if year else "")
            if volume not in volumes:
                volumes.append(volume)
    return volumes


def join_unique(values: list[str]) -> str:
    ordered: list[str] = []
    for value in values:
        value = normalize_space(value)
        if not value or value in ordered:
            continue
        ordered.append(value)
    return "|".join(ordered)


def ensure_safe_output(output_csv: Path, overwrite_output: bool) -> None:
    if output_csv.exists() and not overwrite_output:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_csv}. Pass --overwrite-output to replace it.")


def main(args: Args) -> None:
    ensure_safe_output(args.output_csv, args.overwrite_output)

    published = pd.read_csv(
        args.published_csv_path,
        dtype=str,
        usecols=[
            "oare_id",
            "transliteration",
            "publication_catalog",
            "aliases",
            "label",
            "excavation_no",
        ],
    ).fillna("")

    published["akt_volume"] = published.apply(
        lambda row: join_unique(
            extract_akt_volumes(
                row["publication_catalog"],
                row["aliases"],
                row["label"],
            )
        ),
        axis=1,
    )
    published["transliteration"] = published["transliteration"].map(normalize_transliteration)
    published["excavation_no"] = published.apply(resolve_excavation_no, axis=1)
    override_mask = published["oare_id"].isin(EXCAVATION_NO_OVERRIDES)
    published.loc[override_mask, "excavation_no"] = published.loc[override_mask, "oare_id"].map(EXCAVATION_NO_OVERRIDES)

    akt_only = published.loc[published["akt_volume"] != ""].copy()
    raw_rows = len(akt_only)
    akt_only = (
        akt_only.loc[:, ["oare_id", "transliteration", "akt_volume", "excavation_no"]]
        .drop_duplicates(subset=["oare_id"], keep="first")
        .sort_values("oare_id")
        .reset_index(drop=True)
    )

    if args.expected_rows > 0 and len(akt_only) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows after oare_id dedupe, got {len(akt_only)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    akt_only.to_csv(args.output_csv, index=False)

    print(f"raw_akt_rows={raw_rows}")
    print(f"dropped_duplicate_oare_ids={raw_rows - len(akt_only)}")
    print(f"rows={len(akt_only)}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
