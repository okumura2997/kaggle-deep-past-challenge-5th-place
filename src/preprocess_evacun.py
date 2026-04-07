from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import tyro

DIACRITIC_MAP = {
    "a₂": "á",
    "a₃": "à",
    "e₂": "é",
    "e₃": "è",
    "i₂": "í",
    "i₃": "ì",
    "u₂": "ú",
    "u₃": "ù",
    "A₂": "Á",
    "A₃": "À",
    "E₂": "É",
    "E₃": "È",
    "I₂": "Í",
    "I₃": "Ì",
    "U₂": "Ú",
    "U₃": "Ù",
    "a2": "á",
    "a3": "à",
    "e2": "é",
    "e3": "è",
    "i2": "í",
    "i3": "ì",
    "u2": "ú",
    "u3": "ù",
    "A2": "Á",
    "A3": "À",
    "E2": "É",
    "E3": "È",
    "I2": "Í",
    "I3": "Ì",
    "U2": "Ú",
    "U3": "Ù",
}

SUBSCRIPT_DIGIT_TO_ASCII = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)

SUBSCRIPT_CHAR_DELETE = str.maketrans(
    "",
    "",
    "₀₁₂₃₄₅₆₇₈₉ₓ",
)

DIACRITIC_H_TO_ASCII = str.maketrans(
    {
        "ḫ": "h",
        "Ḫ": "H",
        "ḥ": "h",
        "Ḥ": "H",
    }
)
GLOTTAL_STOP_TO_APOSTROPHE = str.maketrans({"ʾ": "'"})
DASH_TO_ASCII_HYPHEN = str.maketrans({"—": "-", "–": "-"})

PROCESSED_HEADER = ["transliteration", "translation", "fold"]
SPLIT_TO_FILENAMES = {
    "train": ("transcription_train.txt", "english_train.txt"),
    "valid": ("transcription_validation.txt", "english_validation.txt"),
}
SPLIT_TO_FOLD = {"train": 1, "valid": 0}


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def convert_subscript_vowels_to_diacritics(text: str) -> str:
    for src, dst in DIACRITIC_MAP.items():
        text = text.replace(src, dst)
    return text


LEADING_ZERO_NUMBER_RE = re.compile(r"(?<!\d)0+(\d+)")
GRAMMATICAL_MARKER_RE = re.compile(r"(?i)(?<!\w)(?:fem\.|sing\.|pl\.|plur\.|plural)(?![\w.])")
NUM_PLUS_X_RE = re.compile(r"(?<!\w)(\d+)\s*\+\s*x(?!\w)")
STANDALONE_X_RE = re.compile(r"(?<!\w)x(?!\w)")
STANDALONE_PLACEHOLDER_RE = re.compile(r"(?<!\w)(?:PN|GN|NN)(?!\w)")
DOUBLE_ANGLE_BRACKET_RE = re.compile(r"<<\s*([^<>]+?)\s*>>")
ROMAN_NUMERAL_RE = re.compile(r"\b(XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)(?:th)?\b")
MIXED_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s+(\d+)\s*([/:])\s*(\d+)(?!\d)")
SIMPLE_FRACTION_RE = re.compile(r"(?<!\d)(\d+)\s*([/:])\s*(\d+)(?!\d)")
ROMAN_TO_ARABIC = {
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
    "XI": "11",
    "XII": "12",
    "XIII": "13",
    "XIV": "14",
    "XV": "15",
    "XVI": "16",
    "XVII": "17",
    "XVIII": "18",
    "XIX": "19",
    "XX": "20",
}
FRACTION_TO_GLYPH = {
    (1, 10): "⅒",
    (1, 8): "⅛",
    (1, 6): "⅙",
    (1, 5): "⅕",
    (1, 4): "¼",
    (1, 3): "⅓",
    (1, 2): "½",
    (2, 3): "⅔",
    (2, 5): "⅖",
    (3, 8): "⅜",
    (3, 5): "⅗",
    (3, 4): "¾",
    (4, 5): "⅘",
    (5, 8): "⅝",
    (5, 6): "⅚",
    (7, 8): "⅞",
}
FRACTION_GLYPHS = "".join(FRACTION_TO_GLYPH.values())
FRACTION_GLYPH_RE = re.compile(f"([{re.escape(FRACTION_GLYPHS)}])")
DECIMAL_NUMBER_RE = re.compile(r"\b\d+\.\d+\b")


def normalize_leading_zero_numbers(text: str) -> str:
    return LEADING_ZERO_NUMBER_RE.sub(lambda m: str(int(m.group(1))), text)


def remove_subscript_x(text: str) -> str:
    return text.replace("ₓ", "")


def normalize_remaining_subscript_digits(text: str) -> str:
    return text.translate(SUBSCRIPT_DIGIT_TO_ASCII)


def remove_subscript_chars(text: str) -> str:
    return text.translate(SUBSCRIPT_CHAR_DELETE)


def normalize_diacritic_h(text: str) -> str:
    return text.translate(DIACRITIC_H_TO_ASCII)


def normalize_glottal_stop(text: str) -> str:
    return text.translate(GLOTTAL_STOP_TO_APOSTROPHE)


def remove_slash_chars(text: str) -> str:
    return text.replace("/", "")


def remove_pipe_chars(text: str) -> str:
    return text.replace("|", "")


def remove_colon_chars(text: str) -> str:
    return text.replace(":", "")


def remove_parentheses_transliteration(text: str) -> str:
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_dashes(text: str) -> str:
    return text.translate(DASH_TO_ASCII_HYPHEN)


def normalize_gap_token(text: str) -> str:
    return text.replace("...", "<gap>")


def remove_soft_hyphen(text: str) -> str:
    return text.replace("\u00ad", "")


def remove_consecutive_duplicate_words(text: str) -> str:
    words = text.split()
    if not words:
        return text

    deduped = [words[0]]
    for word in words[1:]:
        if word != deduped[-1]:
            deduped.append(word)
    return " ".join(deduped)


def normalize_transliteration_parenthesized_tokens(text: str) -> str:
    text = re.sub(r"\(\s*d\s*\)", "{d}", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*ki\s*\)", "{ki}", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*TÚG\s*\)", "TÚG", text)
    return text


def normalize_translation_x_gaps(text: str) -> str:
    text = NUM_PLUS_X_RE.sub(r"\1+<gap>", text)
    return STANDALONE_X_RE.sub("<gap>", text)


def normalize_translation_placeholders(text: str) -> str:
    return STANDALONE_PLACEHOLDER_RE.sub("<gap>", text)


def remove_double_angle_brackets(text: str) -> str:
    return DOUBLE_ANGLE_BRACKET_RE.sub(r"\1", text)


def normalize_translation_roman_numerals(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        return ROMAN_TO_ARABIC.get(token, token)

    return ROMAN_NUMERAL_RE.sub(repl, text)


def normalize_unicode_fractions(text: str) -> str:
    def replace_mixed(match: re.Match[str]) -> str:
        whole = match.group(1)
        numerator = int(match.group(2))
        denominator = int(match.group(4))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return f"{whole}{glyph}"

    def replace_simple(match: re.Match[str]) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(3))
        glyph = FRACTION_TO_GLYPH.get((numerator, denominator))
        if glyph is None:
            return match.group(0)
        return glyph

    text = MIXED_FRACTION_RE.sub(replace_mixed, text)
    return SIMPLE_FRACTION_RE.sub(replace_simple, text)


def normalize_decimal_fractions(text: str) -> str:
    frac_map = {
        1 / 4: "¼",
        1 / 2: "½",
        1 / 3: "⅓",
        2 / 3: "⅔",
        1 / 6: "⅙",
        3 / 4: "¾",
        5 / 6: "⅚",
        5 / 8: "⅝",
    }
    tol = 1e-3

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            value = float(token)
        except ValueError:
            return token

        whole = math.floor(value)
        frac = value - whole

        frac_1dp = round(frac, 1)
        if frac_1dp == 0.3:
            glyph = frac_map[1 / 3]
            return glyph if whole == 0 else f"{whole} {glyph}"
        if frac_1dp == 0.6:
            glyph = frac_map[2 / 3]
            return glyph if whole == 0 else f"{whole} {glyph}"

        for target, glyph in frac_map.items():
            if abs(frac - target) <= tol:
                return glyph if whole == 0 else f"{whole} {glyph}"
        return token

    return DECIMAL_NUMBER_RE.sub(repl, text)


def add_spaces_around_fraction_glyphs(text: str) -> str:
    text = FRACTION_GLYPH_RE.sub(r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def remove_grammatical_markers(text: str) -> str:
    text = GRAMMATICAL_MARKER_RE.sub("", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def remove_parentheses_chars(text: str) -> str:
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def remove_curly_braces_chars(text: str) -> str:
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def normalize_curly_double_quotes(text: str) -> str:
    return text.replace("“", '"').replace("”", '"')


def remove_uncertain_parenthetical_marker(text: str) -> str:
    text = re.sub(r"\(\s*\?\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def remove_parenthetical_numbers_1_to_12(text: str) -> str:
    text = re.sub(r"\(\s*(?:[1-9]|1[0-2]|I)\s*\)", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def remove_trailing_double_quotes(text: str) -> str:
    return re.sub(r'"(?:\s*"+)*\s*$', "", text).rstrip()


def preprocess_transliteration(text: str) -> str:
    text = normalize_dashes(text)
    text = normalize_gap_token(text)
    text = normalize_transliteration_parenthesized_tokens(text)
    text = normalize_unicode_fractions(text)
    text = normalize_decimal_fractions(text)
    text = add_spaces_around_fraction_glyphs(text)
    text = convert_subscript_vowels_to_diacritics(text)
    text = remove_subscript_x(text)
    text = normalize_remaining_subscript_digits(text)
    text = normalize_diacritic_h(text)
    text = normalize_glottal_stop(text)
    text = remove_slash_chars(text)
    text = remove_pipe_chars(text)
    text = remove_colon_chars(text)
    text = remove_parentheses_transliteration(text)
    text = remove_consecutive_duplicate_words(text)
    text = normalize_leading_zero_numbers(text)
    return text


def preprocess_translation(text: str) -> str:
    text = normalize_dashes(text)
    text = remove_soft_hyphen(text)
    text = normalize_gap_token(text)
    text = remove_double_angle_brackets(text)
    text = normalize_unicode_fractions(text)
    text = normalize_decimal_fractions(text)
    text = remove_subscript_chars(text)
    text = normalize_translation_x_gaps(text)
    text = normalize_translation_placeholders(text)
    text = normalize_translation_roman_numerals(text)
    text = normalize_diacritic_h(text)
    text = normalize_glottal_stop(text)
    text = remove_grammatical_markers(text)
    text = remove_parenthetical_numbers_1_to_12(text)
    text = normalize_curly_double_quotes(text)
    text = remove_uncertain_parenthetical_marker(text)
    text = remove_parentheses_chars(text)
    text = remove_curly_braces_chars(text)
    text = remove_trailing_double_quotes(text)
    return add_spaces_around_fraction_glyphs(text)


def build_rows(
    transcription_path: Path,
    translation_path: Path,
) -> list[tuple[str, str]]:
    transliterations = read_lines(transcription_path)
    translations = read_lines(translation_path)

    if len(transliterations) != len(translations):
        raise ValueError(
            f"Line count mismatch: {transcription_path}={len(transliterations)}, {translation_path}={len(translations)}"
        )

    processed_transliterations = [preprocess_transliteration(line) for line in transliterations]
    processed_translations = [preprocess_translation(line) for line in translations]
    return list(zip(processed_transliterations, processed_translations))


def write_train_processed_csv(output_path: Path, split_rows: dict[str, list[tuple[str, str]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PROCESSED_HEADER)
        for split in ("train", "valid"):
            fold = SPLIT_TO_FOLD[split]
            rows = split_rows.get(split, [])
            for transliteration, translation in rows:
                writer.writerow([transliteration, translation, fold])


def deduplicate_pair_rows(
    split_rows: dict[str, list[tuple[str, str]]],
    split_order: tuple[str, ...] = ("train", "valid"),
) -> tuple[dict[str, list[tuple[str, str]]], int]:
    seen_pairs: set[tuple[str, str]] = set()
    deduped: dict[str, list[tuple[str, str]]] = {}
    removed = 0

    for split in split_order:
        rows = split_rows.get(split, [])
        kept_rows: list[tuple[str, str]] = []
        for row in rows:
            if row in seen_pairs:
                removed += 1
                continue
            seen_pairs.add(row)
            kept_rows.append(row)
        deduped[split] = kept_rows

    for split, rows in split_rows.items():
        if split not in deduped:
            deduped[split] = rows

    return deduped, removed


@dataclass
class Args:
    input_dir: Path = Path("input/evacun")
    output_dir: Path = Path("data/evacun")


def get_split_paths(input_dir: Path) -> dict[str, tuple[Path, Path]]:
    return {
        split: (input_dir / transcription_file, input_dir / translation_file)
        for split, (transcription_file, translation_file) in SPLIT_TO_FILENAMES.items()
    }


def main(args: Args) -> None:
    split_to_paths = get_split_paths(args.input_dir)
    split_rows: dict[str, list[tuple[str, str]]] = {}

    for split, (transcription_path, translation_path) in split_to_paths.items():
        rows = build_rows(
            transcription_path=transcription_path,
            translation_path=translation_path,
        )
        split_rows[split] = rows

    split_rows_all, removed_count_all = deduplicate_pair_rows(split_rows)
    if removed_count_all > 0:
        print(f"Removed duplicate (transliteration, translation) pairs (all): {removed_count_all}")

    train_processed_path = args.output_dir / "train_processed.csv"
    write_train_processed_csv(output_path=train_processed_path, split_rows=split_rows_all)
    print(f"Wrote: {train_processed_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
