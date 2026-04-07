import math
import re
from typing import Literal

import polars as pl


ERROR_OARE_IDS = [
    "e3aecf83-f197-4c23-ae53-56162c679468",
    "8376cbda-b423-42d4-abb5-188d04896392",
    "8eb60135-fe29-4a3b-bad6-8f629e15e0b7",
]
OLD_TRAIN_TRANSLITERATION_SWAP_OARE_IDS = [
    "0960dbf7-bab1-4613-bb18-f81927fa3313",
    "efc3ed8c-e0a0-4902-8a5d-226001b68d96", 
    "2df72c22-1bb1-4032-8f17-68a82a56db01", 
    "0d14c5a8-c152-4417-90cc-36b6dff3226d", 
    "4ac8fc85-8b60-478e-a4ee-855368073efb", 
    "474a0cdb-9910-466c-8431-3c0a20298013", 
    "b7f7be46-076f-42d3-ba4a-524ebaa39ca5", 
    "3a0e700f-5caf-4660-b948-6124e26e550e", 
    "5105200e-a276-48f7-af73-73eb0c1fd657", 
    "5bf906ac-870f-4a27-951e-e54cd64dd9f2", 
    "7fce41d6-9525-46c8-a50d-27fc1af651d7", 
    "f1ff7ea8-3705-4f1c-86ff-e0d2be46edbd", 
    "433be08c-c218-4bbe-b0e7-d6c229dc37bd", 
    "988f713f-9400-4559-9953-8a8e8e350ce9", 
    "d00bd926-f58b-4de4-90cf-2233c6a8f54f", 
    "06a9a456-58e2-4fed-9687-913d824cadc4", 
    "92803d8c-9e53-4d6f-96a2-bda0c4ae4051", 
    "c0b6557c-3836-4f43-b344-db63f8b99eae", 
    "c227c990-44a9-41d6-81f3-243b1ef86e2b", 
    "19f5f61c-f07c-41c7-8f37-211a529863d4", 
    "7c1b5a7f-e203-4c50-afc8-f3dbd3f37cb1", 
    "0309d067-0bb7-40f6-9d27-1dfb0c60e507", 
    "fcf5a0e5-2a2f-4651-8b46-5e860167dc10", 
    "dd9930f0-816c-4db8-8764-076d725f252f", 
    "df4155ff-db56-41c6-92f2-b2f6f3c1518b", 
    "a7e83126-95c9-44b2-9cc7-a3771a009138", 
    "0b94fbc1-0422-4849-9bca-53197403585b", 
    "0a086ecd-dfc9-4828-89c8-ea0b55b23ce5", 
    "3d4d49b5-b646-42db-ae15-c7c017f3a98a", 
    "6e47ef61-8f1b-4a48-b4a6-a759d5be37b8", 
    "89b3ddad-7210-442d-b4e5-192b36f840d5", 
    "95c82565-b89e-48f0-b35a-fcece1abe0ab", 
    "8dad8a7c-a628-432a-9058-e66805874554", 
    "ddaebc28-4dc8-4511-a010-1f0e7cd97b8d", 
    "57949d3d-cab0-4007-b0b2-4ddcf053bcb6", 
    "82155dda-2e35-4568-a6ab-99b0a8158586", 
    "f09d22e9-efb1-4145-b5df-bc85afeb29cb", 
    "659e367b-9c69-4945-90ff-c0fae71c41cc", 
    "6131d12a-0906-4306-95a3-e0ea57efaac4", 
    "c7483adc-e85b-403f-9c1c-c9482d71fb2d", 
    "b24f8913-70f0-4422-847f-12e4c8d5bb36", 
    "821b6253-72c8-43a5-93e1-0b40b5f9a7fb", 
    "84183318-0d21-4f37-96d0-a30811e06873", 
    "fc15e06d-312d-4f77-87f2-7004dd734d6a", 
    "305f83e9-ec38-41d3-b38c-cd73931d01bb", 
    "c5d8a8b7-2b71-4a3a-abf2-4ac84910e219", 
    "3d8d933c-85b8-41d6-af30-76743babaa1a", 
    "107b536a-b713-474f-a2b3-cb3852ef839e", 
    "9c7e32b3-fc22-41cb-a43a-60cd04266022", 
]

OLD_TRAIN_TRANSLATION_SWAP_OARE_IDS = [
    "2e1ab3e9-90ef-48f9-9517-fc9b7329f311", 
    "aa963c56-2765-4559-aa63-873e490df0e1", 
    "5962bdb0-7f70-48ab-872b-d54c47393987", 
    "629f1e04-a93d-429b-bdb1-aa6e02278659", 
    "6bba0a9c-51fc-4fc0-87d3-c65def1790ad", 
    "bcad41ab-0c33-43f2-b9ff-d2b30699f181", 
    "315acabf-90ed-4d43-9e90-250df372a187", 
    "ea075e58-893d-487d-978f-2682802ecb70", 
    "529fcbd8-9122-4d49-9cb9-021545dd9fb1", 
    "19050ec7-5bef-442a-a85a-54a3c4a9674c", 
    "c5670aba-105b-40b5-bac1-12bc43a51091", 
    "e5cab8ae-3b39-4280-a22f-a143682ad448", 
    "e76705fe-094c-4fa3-8506-2bca4d4e7b7c", 
    "054fdba4-0cff-4153-969d-c77e42413e1c", 
    "7c71b551-4b3e-42f3-91e8-6eae52048f27", 
    "d38ae78c-c321-4e83-8c0f-264ef6e0ef78", 
    "4eb23b20-e6ce-4c45-ac07-07d62a2ef0d0", 
    "e05f20dc-7a27-4b71-8d0f-628a8757091d", 
    "f14f3479-31a5-4f13-a721-3b0eec97a895", 
    "27a814ba-f860-4468-8986-16e7ba2faaaa", 
    "1d0c7278-13ef-42e9-814f-0412eecadf20", 
    "d604d277-4d17-492e-9f1b-0357108bb99b", 
    "7b05ef05-e0c2-4123-905a-da6d2c3301eb", 
    "a9f48c23-483a-4f68-8930-25eaf746029d", 
    "99552f43-f500-4605-b650-5e18dd30a4c4", 
    "103e6d83-27e6-4dcf-ac7f-747f9674e37c", 
    "81f1a9f0-544f-4939-8510-2128c98764b7", 
    "8c737557-7e47-4193-a739-13e261a9ea99", 
    "7ba84024-458e-4452-93b5-073052e72b24",  # gap issue
    "697fdae3-abd6-4de6-89ac-03796381be51", 
    "821b6253-72c8-43a5-93e1-0b40b5f9a7fb", 
]

TRANSLITERATION_TO_TRANSLATION_MAX_RATIO = 1.7
TRANSLATION_TO_TRANSLITERATION_MAX_RATIO = 1.7
PERSON_NAME_PLACEHOLDER_RE = re.compile(
    r"\bPN(?:\s*[0-9₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]+)?\b",
    flags=re.IGNORECASE,
)

TRANSLATION_SHEKEL_FRACTION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"5\s+11\s*/\s*12\s*shekels?\b"),
        "6 shekels less 15 grains",
    ),
    (
        re.compile(r"1\s*/\s*12\s*shekels?\b"),
        "15 grains",
    ),
    (
        re.compile(r"7\s*/\s*12\s*shekels?\b"),
        "½ shekel 15 grains",
    ),
    (
        re.compile(r"5\s*/\s*12\s*shekels?\b"),
        "⅓ shekel 15 grains",
    ),
    ( 
        re.compile(r"1\s*/\s*12(?!\s*shekels?\b)"),
        "15 grains",
    ),
]


def normalize_transliteration(text: str) -> str:
    text = text.replace("ₓ", "")
    text = normalize_transliteration_parenthesized_tokens(text)
    text = re.sub(r"(xx+|\s+x\s+)", "<gap>", text)
    text = PERSON_NAME_PLACEHOLDER_RE.sub("<gap>", text)
    text = text.replace("ı", "i")
    text = text.replace("!", " ")
    text = text.replace(",", " ")
    text = text.replace(":", "")
    text = text.replace("=", " ")
    text = text.replace("⁇", "")
    text = re.sub(r"\(\s*[1-9]\s+broken\s+lines\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*broken\s+line\s*\)", "", text, flags=re.IGNORECASE)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s{2,}", " ", text).strip()
    subscript_digit_map = str.maketrans(
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
    special_char_map = str.maketrans(
        {
            "Ḫ": "H",
            "ḫ": "h",
            "ā": "á",
            "Ā": "Á",
            "â": "á",
            "Â": "Á",
            "ē": "é",
            "Ē": "É",
            "ê": "é",
            "Ê": "É",
            "ī": "í",
            "Ī": "Í",
            "î": "í",
            "Î": "Í",
            "ū": "ú",
            "Ū": "Ú",
            "û": "ú",
            "Û": "Ú",
            "/": "",
            "ʾ": "",
            "ʿ": "",
            "ʔ": "",
            "ʽ": "",
            "ʹ": "",
            "′": "",
            "ˈ": "",
            "´": "",
            "|": "",
            "⌈": "",
            "⌉": "",
            "ț": "ṭ",
            "Ț": "Ṭ",
            "ţ": "ṭ",
            "Ţ": "Ṭ",
            "ș": "š",
            "Ș": "Š",
            "ş": "š",
            "Ş": "Š",
            "ğ": "g",
            "Ğ": "G",
            "Ĝ": "G",
            "Į": "I",
            "Ỉ": "I",
            "ỉ": "i",
            "ʓ": "z",
            "ž": "z",
            "Ž": "Z",
            "ź": "z",
            "Ź": "Z",
            "ś": "s",
            "Ś": "S",
            "ť": "t",
            "Ť": "T",
            "ᵦ": "b",
            "ṗ": "p",
            "Ṗ": "P",
            "ă": "a",
            "ä": "a",
            "Ä": "A",
            "ü": "u",
            "Ü": "U",
            "ß": "ss",
            "Ă": "A",
            "ḳ": "q",
            "Ḳ": "Q",
            "ṇ": "n",
            "Ṇ": "N",
            "ṃ": "m",
            "Ṃ": "M",
            "ṁ": "m",
            "Ṁ": "M",
            "ř": "r",
            "Ř": "R",
            "ł": "l",
            "Ł": "L",
            "ḵ": "k",
            "Ḵ": "K",
            "ḏ": "d",
            "Ḏ": "D",
            "ṯ": "t",
            "Ṯ": "T",
            "ṫ": "t",
            "Ṫ": "T",
            "ẕ": "z",
            "Ẕ": "Z",
            "ʃ": "š",
            "ï": "i",
            "Ï": "I",
            "Ŋ": "N",
            "˘": "",
            "˝": "",
            "°": "",
            "̌": "",
            "ˉ": "",
            "˂": "",
            "˃": "",
            "˰": "",
            "⁁": "",
            "𒋀": "",
            "f": "",
            "́": "",
            "̇": "",
        }
    )
    text = text.translate(subscript_digit_map)
    text = PERSON_NAME_PLACEHOLDER_RE.sub("<gap>", text)
    text = text.translate(special_char_map)
    text = normalize_decimals(text)
    text = dedup_consecutive_words(text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def normalize_translation(text: str) -> str:
    char_map = {
        "}": "", 
        "ד": " ", 
        "ʾ": "'",
        "ʔ": "'",
        "ʽ": "'",
        "ʹ": "'",
        "′": "'",
        "ˈ": "'",
        "ˀ": "'",
        "´": "'",
        "ḫ": "h",
        "Ḫ": "H",
        "ḥ": "h",
        "Ḥ": "H",
        "â": "ā",
        "ê": "ē",
        "î": "ī",
        "û": "ū",
        "Â": "A",
        "Ê": "E",
        "Î": "I",
        "Û": "U",
        "Ā": "A",
        "Ē": "E",
        "Ī": "I",
        "Ū": "U",
        "á": "a",
        "Á": "A",
        "à": "a",
        "À": "A",
        "é": "e",
        "É": "E",
        "è": "e",
        "È": "E",
        "í": "i",
        "Í": "I",
        "ì": "i",
        "Ì": "I",
        "ú": "u",
        "Ú": "U",
        "ù": "u",
        "Ù": "U",
        "ä": "a",
        "Ä": "A",
        "ï": "i",
        "Ï": "I",
        "ë": "e",
        "Ë": "E",
        "ß": "ss",
        "{": "",
        "V": "v",
        "е": "e",
        "д": "d",
        "р": "r",
        "і": "i",
        "и": "i",
        "н": "n",
        "а": "a",
        "с": "s",
        "đ": "d",
        "Đ": "D",
        "ž": "z",
        "Ž": "Z",
        "ż": "z",
        "Ż": "Z",
        "ț": "t",
        "Ț": "T",
        "ș": "s",
        "Ș": "S",
        "ţ": "t",
        "Ţ": "T",
        "ụ": "u",
        "ᵉ": "e",
        "†": "",
        "ə": "e",
        "ĝ": "g",
        "Ĝ": "G",
        "č": "c",
        "Č": "C",
        "ň": "n",
        "Ň": "N",
        "ñ": "n",
        "Ñ": "N",
        "ł": "l",
        "Ł": "L",
        "ǧ": "g",
        "Ǧ": "G",
        "ŝ": "s",
        "Ŝ": "S",
        "Ŭ": "U",
        "ŭ": "u",
        "ɓ": "b",
        "Į": "I",
        "į": "i",
        "ą": "a",
        "Ą": "A",
        "Ĥ": "H",
        "ĥ": "h",
        "ľ": "l",
        "Ľ": "L",
        "Ã": "A",
        "ã": "a",
        "ᶜ": "c",
        "ᵛ": "v",
        "ᶦ": "i",
        "ᵇ": "b",
        "ǵ": "g",
        "Ǵ": "G",
        "ᵐ": "m",
        "%": "",
        "°": "",
        "=": " ",
        "ḍ": "d",
        "Ḍ": "D",
        "ḏ": "d",
        "Ḏ": "D",
        "ḳ": "q",
        "Ḳ": "Q",
        "ṯ": "t",
        "Ṯ": "T",
        "ǩ": "k",
        "Ǩ": "K",
        "ṛ": "r",
        "Ṛ": "R",
        "ṟ": "r",
        "Ṟ": "R",
        "ṝ": "r",
        "Ṝ": "R",
        "ṙ": "r",
        "Ṙ": "R",
        "ṇ": "n",
        "Ṇ": "N",
        "ṉ": "n",
        "Ṉ": "N",
        "ṃ": "m",
        "Ṃ": "M",
        "ṁ": "m",
        "Ṁ": "M",
        "ḷ": "l",
        "Ḷ": "L",
        "ḵ": "k",
        "Ḵ": "K",
        "ẞ": "ss",
        "β": "ss",
        "̩": "",
        "̱": "",
        "̇": "",
        "̈": "",
        "̥": "",
        "̓": "",
        "̃": "",
        "̂": "",
    }
    
    text = text.replace("huburhe", "hubur he")
    text = text.replace("IVth month", "month IV")
    text = re.sub(r"\byouplur\.?(?=\W|$)", "you", text, flags=re.IGNORECASE)
    # text = re.sub(r"(?<![A-Za-z])-gold\b", "pašallum gold", text)
    # text = re.sub(r"(?<![A-Za-z])-tax\b", "šadduātum tax", text)
    text = re.sub(r"(?<![A-Za-z])-textile\b", "kutānum-textile", text)
    text = re.sub(r"(?<!\S)-textiles\b", "kutānu-textiles", text)
    text = re.sub(r"<<\s*([^<>]+?)\s*>>", r"\1", text)
    text = PERSON_NAME_PLACEHOLDER_RE.sub("<gap>", text)

    text = re.sub(r"\(\s*\?\s*\)", "", text)
    text = re.sub(r'""\s*$', "", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(
        r"\(?\s*(?<![A-Za-z])(fem|sing|pl|plur|plural)\.?\s*\)?\.?(?![A-Za-z])",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    text = normalize_decimals(text)
    text = normalize_roman_numerals(text)
    text = replace_translation_shekel_fractions(text)
    text = normalize_fraction(text)
    char_trans = str.maketrans(char_map)
    text = text.translate(char_trans)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def replace_translation_shekel_fractions(text: str) -> str:
    for pattern, replacement in TRANSLATION_SHEKEL_FRACTION_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def normalize_transliteration_parenthesized_tokens(text: str) -> str:
    text = re.sub(r"\(\s*d\s*\)", "{d}", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*ki\s*\)", "{ki}", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*TÚG\s*\)", " TÚG ", text)
    text = re.sub(r"\{\s*TÚG\s*\}", " TÚG ", text)
    return text


def dedup_consecutive_words(text: str) -> str:
    words = text.split()
    if not words:
        return text

    deduped = [words[0]]
    for word in words[1:]:
        if word != deduped[-1]:
            deduped.append(word)
    return " ".join(deduped)


def normalize_decimals(text: str) -> str:
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

    return re.sub(r"\b\d+\.\d+\b", repl, text)


def normalize_roman_numerals(text: str) -> str:
    roman_map = {
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
    }
    pattern = re.compile(r"\b(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)(?:th)?\b")

    def repl(match: re.Match[str]) -> str:
        return roman_map.get(match.group(1), match.group(0))

    return pattern.sub(repl, text)


def normalize_fraction(text: str) -> str:
    fraction_map = {
        (1, 2): "½",
        (1, 3): "⅓",
        (2, 3): "⅔",
        (1, 4): "¼",
        (3, 4): "¾",
        (1, 5): "⅕",
        (2, 5): "⅖",
        (3, 5): "⅗",
        (4, 5): "⅘",
        (1, 6): "⅙",
        (5, 6): "⅚",
        (1, 8): "⅛",
        (3, 8): "⅜",
        (5, 8): "⅝",
        (7, 8): "⅞",
    }

    # 両側にスペースがある分数だけマッチ（1/4 や 1 /4 はマッチしない）
    pat = re.compile(r"(?<!\d)(\d+)\s+/\s+(\d+)(?!\d)")

    def replace_frac(m):
        # Keep expressions like "Month 2 / 3" as-is.
        prefix = text[: m.start()]
        if re.search(r"month\s+$", prefix, flags=re.IGNORECASE):
            return m.group(0)
        a, b = int(m.group(1)), int(m.group(2))
        return fraction_map.get((a, b), m.group(0))  # 未対応はそのまま

    return pat.sub(replace_frac, text)


def choose_slash_word(
    text: str,
    pick: Literal["first", "second"] = "first",
    debug: bool = True,
) -> str:

    SLASH_WORD_RE = re.compile(r"(?P<left>[^\s/]+)\s*/\s*(?P<right>[^\s/]+)")

    def _repl(m: re.Match) -> str:
        left = m.group("left")
        right = m.group("right")

        # 除外: {数値} / 12 は変換しない
        if left.isdigit() and right == "12":
            return m.group(0)

        replacement = left if pick == "first" else right

        if debug and replacement != m.group(0):
            return replacement

    replaced = SLASH_WORD_RE.sub(_repl, text)

    if text != replaced:
        print(f"before: {text}\nafter: {replaced}")

    return replaced


def cut_by_word_boundary(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    words = normalized.split(" ")
    if not words:
        return ""

    kept: list[str] = []
    current_len = 0
    for word in words:
        next_len = current_len + (1 if kept else 0) + len(word)
        if kept and next_len > max_chars:
            break
        kept.append(word)
        current_len = next_len
        if current_len >= max_chars:
            break
    return " ".join(kept)


def cut_if_too_long(
    text: str,
    reference_text: str,
    max_ratio: float = 1.8,
    fallback_chars: int = 16,
) -> str:
    normalized_text = " ".join(text.split())
    normalized_ref = " ".join(reference_text.split())

    if not normalized_text:
        return text
    if not normalized_ref:
        return cut_by_word_boundary(normalized_text, fallback_chars)

    if len(normalized_text) <= len(normalized_ref) * max_ratio:
        return text

    target_chars = max(int(len(normalized_ref) * max_ratio), 1)
    return cut_by_word_boundary(normalized_text, target_chars)


def was_truncated(original_text: str, cut_text: str) -> bool:
    original_normalized = " ".join((original_text or "").split())
    cut_normalized = " ".join((cut_text or "").split())
    return len(cut_normalized) < len(original_normalized)


def assign_group_folds(
    groups: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[int]:
    """Assign fold indices so that each group stays in a single fold."""
    import numpy as np

    rng = np.random.default_rng(seed)

    group_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        group_to_rows.setdefault(g, []).append(i)

    unique_groups = list(group_to_rows.keys())
    rng.shuffle(unique_groups)

    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    folds = np.full(len(groups), -1, dtype=np.int64)

    for g in unique_groups:
        rows = group_to_rows[g]
        target_fold = int(fold_sizes.argmin())
        for r in rows:
            folds[r] = target_fold
        fold_sizes[target_fold] += len(rows)

    return folds.tolist()


def _truncate_context_piece_from_left(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    truncated = encoded[-max_bytes:]
    while truncated and (truncated[0] & 0b1100_0000) == 0b1000_0000:
        truncated = truncated[1:]
    return truncated.decode("utf-8", errors="ignore").lstrip()



def build_context_prefix(ctx_parts: list[str], max_bytes: int) -> str:
    """Build ``[context]...[/context] `` from ctx_parts (oldest-first order).

    If *max_bytes* > 0 and the prefix exceeds it, drop oldest sentences first.
    Returns empty string if no context remains.
    """
    if not ctx_parts:
        return ""
    _TAG_OVERHEAD = len("[context]  [/context] ".encode())
    _SEP = " [sep] "
    _SEP_BYTES = len(_SEP.encode())

    if max_bytes > 0:
        while ctx_parts:
            content_bytes = sum(len(p.encode()) for p in ctx_parts)
            sep_bytes = _SEP_BYTES * (len(ctx_parts) - 1) if len(ctx_parts) > 1 else 0
            total = _TAG_OVERHEAD + content_bytes + sep_bytes
            if total <= max_bytes:
                break
            ctx_parts.pop(0)

    if not ctx_parts:
        return ""
    ctx = _SEP.join(ctx_parts)
    return f"[context] {ctx} [/context] "


def compute_context_prefixes(
    group_keys: list,
    inputs: list[str],
    num_prev: int,
    max_bytes: int = 0,
) -> list[str]:
    """Compute ``[context]...[/context] `` prefix for each position in sorted order.

    *group_keys* identifies the document group (e.g. excavation_number or text_id).
    Only previous sentences within the same group are used as context.
    If *max_bytes* > 0, truncate oldest context sentences first to fit within the limit.
    Returns a list of prefix strings (empty string for rows with no context).
    """
    prefixes: list[str] = []
    for i in range(len(inputs)):
        ctx_parts: list[str] = []
        for j in range(1, num_prev + 1):
            prev_idx = i - j
            if prev_idx < 0 or group_keys[prev_idx] != group_keys[i]:
                break
            ctx_parts.append(inputs[prev_idx])
        if ctx_parts:
            ctx_parts.reverse()  # oldest first
        prefixes.append(build_context_prefix(ctx_parts, max_bytes))
    return prefixes



def assign_stratified_folds_from_continuous(
    values: list[float],
    n_splits: int = 4,
    n_bins: int = 20,
    seed: int = 42,
) -> list[int]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    folds = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    valid_mask = np.isfinite(arr)
    valid_idx = np.where(valid_mask)[0]
    invalid_idx = np.where(~valid_mask)[0]

    if valid_idx.size > 0:
        sorted_pos = np.argsort(arr[valid_mask], kind="mergesort")
        ranks = np.empty(valid_idx.size, dtype=np.int64)
        ranks[sorted_pos] = np.arange(valid_idx.size, dtype=np.int64)
        # 連続値を分位ベースで離散化して疑似層化する。
        bins = (ranks * n_bins) // valid_idx.size
        bins = np.clip(bins, 0, n_bins - 1)

        for b in range(n_bins):
            local = valid_idx[bins == b]
            if local.size == 0:
                continue
            shuffled = local[rng.permutation(local.size)]
            folds[shuffled] = np.arange(local.size) % n_splits

    if invalid_idx.size > 0:
        shuffled = invalid_idx[rng.permutation(invalid_idx.size)]
        folds[shuffled] = np.arange(invalid_idx.size) % n_splits

    return folds.tolist()


def build_published_texts_without_oare_ids(published_texts, excluded_oare_ids):
    return (
        published_texts
        .join(
            excluded_oare_ids
            .select("oare_id")
            .filter(
                pl.col("oare_id").is_not_null()
                & (pl.col("oare_id").str.strip_chars() != "")
            )
            .unique(subset=["oare_id"], keep="first", maintain_order=True),
            on="oare_id",
            how="anti",
        )
        .select(["oare_id", "transliteration"])
        .with_columns(
            pl.col("transliteration").map_elements(
                normalize_transliteration, return_dtype=pl.String
            )
        )
        .unique(subset=["transliteration"], keep="first", maintain_order=True)
    )


if __name__ == "__main__":
    import polars as pl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    train = pl.read_csv("./input/deep-past-initiative-machine-translation/train.csv")
    train_old = (
        pl.read_csv("./input/deep-past-initiative-machine-translation_old/train.csv")
        .select(["oare_id", "transliteration", "translation"])
        .unique(subset=["oare_id"], keep="first", maintain_order=True)
        .rename(
            {
                "transliteration": "old_train_transliteration",
                "translation": "old_train_translation",
            }
        )
    )
    published_texts = pl.read_csv("./input/deep-past-initiative-machine-translation/published_texts.csv", ignore_errors=True)
    train = train.join(train_old, on="oare_id", how="left").with_columns(
        pl.when(
            pl.col("oare_id").is_in(OLD_TRAIN_TRANSLITERATION_SWAP_OARE_IDS)
            & pl.col("old_train_transliteration").is_not_null()
        )
        .then(pl.col("old_train_transliteration"))
        .otherwise(pl.col("transliteration"))
        .alias("transliteration"),
        pl.when(
            pl.col("oare_id").is_in(OLD_TRAIN_TRANSLATION_SWAP_OARE_IDS)
            & pl.col("old_train_translation").is_not_null()
        )
        .then(pl.col("old_train_translation"))
        .otherwise(pl.col("translation"))
        .alias("translation"),
    )
    published_texts_for_join = (
        published_texts
        .select(["oare_id", "transliteration"])
        .unique(subset=["oare_id"], keep="first", maintain_order=True)
        .rename({"transliteration": "published_texts"})
    )
    train = train.join(published_texts_for_join, on="oare_id", how="left")
    train_oare_ids = train.select("oare_id").unique(subset=["oare_id"], keep="first", maintain_order=True)
    
    train = train.filter(~pl.col("oare_id").is_in(ERROR_OARE_IDS))

    # formatting
    train = train.with_columns(
        pl.col("transliteration").map_elements(
            normalize_transliteration, return_dtype=pl.String
        ).alias("transliteration_orig"),
        pl.col("translation").map_elements(normalize_translation, return_dtype=pl.String),
        pl.when(pl.col("published_texts").is_not_null())
        .then(
            pl.col("published_texts").map_elements(
                normalize_transliteration, return_dtype=pl.String
            )
        )
        .otherwise(None)
        .alias("published_texts"),
    ).with_columns(
        (
            pl.col("published_texts").is_not_null()
            & (pl.col("transliteration_orig") == pl.col("published_texts"))
        ).alias("transliteration_published_texts_match")
    ).with_columns(
        pl.when(pl.col("oare_id").is_in(OLD_TRAIN_TRANSLITERATION_SWAP_OARE_IDS))
        .then(pl.col("transliteration_orig"))
        .otherwise(pl.coalesce([pl.col("published_texts"), pl.col("transliteration_orig")]))
        .alias("transliteration")
    )

    train = train.with_columns(
        pl.col("transliteration").str.len_chars().alias("translit_len"),
        pl.col("translation").str.len_chars().alias("translation_len"),
    ).with_columns(
        pl.when(pl.col("translit_len") > 0)
        .then(pl.col("translation_len") / pl.col("translit_len"))
        .otherwise(None)
        .alias("length_ratio")
    )

    translit_tokenized = tokenizer(train.get_column("transliteration").to_list(), return_attention_mask=False, add_special_tokens=False)
    translation_tokenized = tokenizer(train.get_column("translation").to_list(), return_attention_mask=False, add_special_tokens=False)
    train = train.with_columns(
        pl.Series([len(x) for x in translit_tokenized["input_ids"]]).alias("translit_tokenized_len"),
        pl.Series([len(x) for x in translation_tokenized["input_ids"]]).alias("translation_tokenized_len"),
    ).with_columns(
        pl.when(pl.col("translit_tokenized_len") > 0)
        .then(pl.col("translation_tokenized_len") / pl.col("translit_tokenized_len"))
        .otherwise(None)
        .alias("tokenized_length_ratio")
    )

    folds = assign_stratified_folds_from_continuous(train.get_column("length_ratio").to_list(), n_splits=4, n_bins=20, seed=42)
    cols = [
        "oare_id",
        "transliteration",
        "translation",
        "fold"
    ]
    train = train.with_columns(pl.Series("fold", folds).cast(pl.Int8))
    train.select(cols).write_csv("./data/train_processed.csv")
