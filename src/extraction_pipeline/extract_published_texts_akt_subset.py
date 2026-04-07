"""Extract AKT-related rows from published_texts.csv."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro

from extraction_pipeline.utils.preprocess import normalize_transliteration


@dataclass
class Args:
    published_csv_path: str = "./input/deep-past-initiative-machine-translation/published_texts.csv"
    output_csv: str = "./data/published_texts_akt_subset.csv"
    expected_rows: int = 2800
    overwrite_output: bool = False


EXCAVATION_NO_OVERRIDES: dict[str, str] = {
    "017f25c5-0f92-4631-957a-fc98000f078b": "Kt 92/k 321",
}

EXCAVATION_NO_TEXT_OVERRIDES: dict[str, str] = {
    "Kt 91/k 518 envelope": "Kt 91/k 518A",
    "Kt 91/k 518 tablet": "Kt 91/k 518B",
}

AKT1_IDENTIFIER_RE = re.compile(
    r"(?i)\bAKT\s+1\s+(?P<number>\d+[a-z]?)(?:\s+(?P<suffix>env(?:elope)?|tab(?:let)?|zarf))?\b"
)

AKT_VOLUME_RE = re.compile(
    r"(?i)\bAKT\s+(\d+[a-z]?)(?:\s+(19\d{2}|20\d{2}))?"
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_akt1_identifier(*texts: str) -> str:
    for text in texts:
        match = AKT1_IDENTIFIER_RE.search(text or "")
        if match is None:
            continue
        number = _normalize_space(match.group("number"))
        if number:
            return f"No.{number}"
    return ""


def _resolve_excavation_no(
    excavation_no: str,
    aliases: str,
    label: str,
    publication_catalog: str,
) -> str:
    excavation_no = _normalize_space(excavation_no)
    akt1_id = _extract_akt1_identifier(aliases, label, publication_catalog)
    if akt1_id:
        return akt1_id

    searchable = " | ".join(
        _normalize_space(t) for t in (excavation_no, aliases, label, publication_catalog)
    ).lower()
    for source_text, target in EXCAVATION_NO_TEXT_OVERRIDES.items():
        if source_text.lower() in searchable:
            return target
    return excavation_no


def _extract_akt_volumes(*texts: str) -> str:
    volumes: list[str] = []
    for text in texts:
        for match in AKT_VOLUME_RE.finditer(text or ""):
            base = match.group(1).lower()
            year = match.group(2)
            volume = f"AKT {base}" + (f" {year}" if year else "")
            if volume not in volumes:
                volumes.append(volume)
    return "|".join(volumes)


def main(args: Args) -> None:
    output_csv = Path(args.output_csv)
    if output_csv.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_csv}. "
            "Pass --overwrite-output to replace it."
        )

    df = pl.read_csv(
        args.published_csv_path,
        columns=[
            "oare_id",
            "transliteration",
            "publication_catalog",
            "aliases",
            "label",
            "excavation_no",
        ],
        schema_overrides={col: pl.Utf8 for col in [
            "oare_id", "transliteration", "publication_catalog",
            "aliases", "label", "excavation_no",
        ]},
        null_values=[""],
    ).with_columns(pl.all().fill_null(""))

    # Extract AKT volumes
    df = df.with_columns(
        pl.struct(["publication_catalog", "aliases", "label"])
        .map_elements(
            lambda row: _extract_akt_volumes(
                row["publication_catalog"], row["aliases"], row["label"]
            ),
            return_dtype=pl.Utf8,
        )
        .alias("akt_volume")
    )

    # Normalize transliteration
    df = df.with_columns(
        pl.col("transliteration")
        .map_elements(normalize_transliteration, return_dtype=pl.Utf8)
    )

    # Resolve excavation_no
    df = df.with_columns(
        pl.struct(["excavation_no", "aliases", "label", "publication_catalog"])
        .map_elements(
            lambda row: _resolve_excavation_no(
                row["excavation_no"],
                row["aliases"],
                row["label"],
                row["publication_catalog"],
            ),
            return_dtype=pl.Utf8,
        )
        .alias("excavation_no")
    )

    # Apply overrides by oare_id
    override_map = pl.DataFrame({
        "oare_id": list(EXCAVATION_NO_OVERRIDES.keys()),
        "excavation_no_override": list(EXCAVATION_NO_OVERRIDES.values()),
    })
    df = (
        df.join(override_map, on="oare_id", how="left")
        .with_columns(
            pl.when(pl.col("excavation_no_override").is_not_null())
            .then(pl.col("excavation_no_override"))
            .otherwise(pl.col("excavation_no"))
            .alias("excavation_no")
        )
        .drop("excavation_no_override")
    )

    # Filter to AKT rows only
    akt_only = (
        df.filter(pl.col("akt_volume") != "")
        .select(["oare_id", "transliteration", "akt_volume", "excavation_no"])
        .unique(subset=["oare_id"], keep="first")
        .sort("oare_id")
    )

    if args.expected_rows > 0 and len(akt_only) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} rows after oare_id dedupe, got {len(akt_only)}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    akt_only.write_csv(output_csv)

    print(f"rows={len(akt_only)}")
    print(f"output_csv={output_csv}")


if __name__ == "__main__":
    main(tyro.cli(Args))
