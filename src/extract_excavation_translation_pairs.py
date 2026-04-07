import csv
import gc
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
import tyro
from sacrebleu.metrics import CHRF
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


@dataclass(frozen=True)
class TranslationPromptProfile:
    name: str
    system_prompt: str
    user_prompt_template: str


TRANSLATION_SYSTEM_PROMPT = """You extract Turkish translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) Turkish summary prose (must be excluded)
  3) Akkadian transliteration block
  4) Turkish translation block (extract this only)
- Extract only Turkish translation text for the specified excavation number.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Transliteration often appears with section cues like "öy.", "ak.", "ay."; do not include it.
- Translation often starts with line-range markers like "1-2)", "3-9)"; these are part of translation block.
- Do not include transliteration, English text, headings, commentary, or section titles.
- If only summary prose is visible and translation block is not visible, return empty translation.
- If no Turkish translation is readable, return an empty translation string.
"""

TRANSLATION_USER_PROMPT_TEMPLATE = """Extract the Turkish translation for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}
Reference transliteration:
{transliteration}

Important:
- Use the transliteration above to identify the matching translation passage for this excavation.
- If nearby entries are visible, extract only the translation corresponding to this transliteration.
- Exclude the Turkish summary text that appears before transliteration.
- Extract only Turkish translation that comes after transliteration.

Return JSON with schema:
{{
  "excavation_number": "string",
  "translation": "string"
}}
"""

AKT2_TRANSLATION_SYSTEM_PROMPT = """You extract Turkish translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) Turkish summary prose (must be excluded)
  3) Akkadian transliteration block
  4) Turkish translation block (extract this only)
- Extract only Turkish translation text for the specified excavation number.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Transliteration often appears with section cues like "öy.", "ak.", "ay."; do not include it.
- Translation often starts with line-range markers like "1-2)", "3-9)"; these are part of translation block.
- After the translation block, pages may include commentary or bibliography notes. Stop before those.
- Exclude note/reference lines such as "13: ... AHw ...", "GKT ...", "CAD ...", etc.
- Do not include transliteration, English text, headings, commentary, or section titles.
- If only summary prose is visible and translation block is not visible, return empty translation.
- If no Turkish translation is readable, return an empty translation string.
"""

AKT2_TRANSLATION_USER_PROMPT_TEMPLATE = """Extract the Turkish translation for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}
Reference transliteration:
{transliteration}

Important:
- Use the transliteration above to identify the matching translation passage for this excavation.
- If nearby entries are visible, extract only the translation corresponding to this transliteration.
- Exclude the Turkish summary text that appears before transliteration.
- Extract only Turkish translation that comes after transliteration.
- Exclude trailing commentary/reference notes that appear after the translation.

Return JSON with schema:
{{
  "excavation_number": "string",
  "translation": "string"
}}
"""

AKT3_TRANSLATION_SYSTEM_PROMPT = """You extract German translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) prose before transliteration (must be excluded)
  3) Akkadian transliteration block
  4) German translation block (extract this only)
- Extract only German translation text for the specified excavation number.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract prose between heading and transliteration.
- Transliteration often appears with section cues like "öy.", "ak.", "ay."; do not include it.
- Translation often starts with line-range markers like "1-2)", "3-9)"; these are part of translation block.
- Do not include transliteration, English text, headings, commentary, or section titles.
- If only summary prose is visible and translation block is not visible, return empty translation.
- If no German translation is readable, return an empty translation string.
"""

AKT3_TRANSLATION_USER_PROMPT_TEMPLATE = """Extract the German translation for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}
Reference transliteration:
{transliteration}

Important:
- Use the transliteration above to identify the matching translation passage for this excavation.
- If nearby entries are visible, extract only the translation corresponding to this transliteration.
- Exclude any summary prose that appears before transliteration.
- Extract only German translation that comes after transliteration.

Return JSON with schema:
{{
  "excavation_number": "string",
  "translation": "string"
}}
"""

SIDE_BY_SIDE_TRANSLATION_SYSTEM_PROMPT = """You extract translation text for a single excavation number from page images.
Return JSON only.

Rules:
- These pages use a side-by-side layout:
  1) left side = Akkadian transliteration
  2) right side = translation
- Extract only the translation for the specified excavation number.
- Read only the RIGHT-HAND translation column/area.
- Ignore the LEFT-HAND transliteration column entirely, even if it has line numbers, side labels, or clearer text.
- Do not interleave left-column transliteration with right-column translation.
- The first image is the TARGET page where the excavation number was detected.
- Additional images are NEXT context pages and may contain continuation text.
- Merge continuation text in reading order within the translation column, page by page.
- Header lines at the top of the entry may contain metadata, references, or edition notes. Exclude those unless they are clearly part of the translation itself.
- Translation may not repeat line numbers because numbering is often shown only beside the transliteration column.
- Do not include transliteration, metadata, references, bibliography, headings, commentary, or section titles.
- If no readable translation is visible on the right side, return an empty translation string.
"""

SIDE_BY_SIDE_TRANSLATION_USER_PROMPT_TEMPLATE = """Extract the translation for this excavation number from the side-by-side page layout.
Excavation number: {excavation_number}
Target page number: {page_number}
Reference transliteration:
{transliteration}

Important:
- Use the transliteration above to identify the matching translation passage for this excavation.
- If nearby entries are visible, extract only the translation corresponding to this transliteration.
- Read only the RIGHT-SIDE translation column.
- Ignore the LEFT-SIDE transliteration column completely.
- Exclude top metadata/reference lines before the translation column begins.

Return JSON with schema:
{{
  "excavation_number": "string",
  "translation": "string"
}}
"""

TRANSLATION_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_number": {"type": "string"},
        "translation": {"type": "string"},
    },
    "required": ["excavation_number", "translation"],
    "additionalProperties": False,
}


PAGE_IMAGE_RE = re.compile(r"^page_(\d{3,})\.(?:png|jpg|jpeg|webp)$", flags=re.IGNORECASE)
INTRO_SUMMARY_CUE_RE = re.compile(
    r"(?i)\b(?:mektuptur|yazdığı|tarafından|maktubun sahibi|yayınlanmış|belgesidir)\b"
)
LINE_MARKER_RE = re.compile(r"\b\d{1,3}(?:\s*['′]?\s*[-)\.]|['′])")
TRANSLATION_START_CUE_RE = re.compile(
    r"(?i)\b(?:şöyle\s*\(söyler\)\s*:|buzutaya['’]?ya\s+söyle!?|um-ma)\b"
)
TRANSLITERATION_SECTION_CUE_RE = re.compile(r"(?i)\b(?:öy|oy|ak|ay)\.\s*\d{1,3}\b")
TRANSLATION_LINE_RANGE_RE = re.compile(r"(?m)^\s*(\d{1,3}\s*-\s*\d{1,3}\s*\))")
PAGE_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
CHRF_PLUS_PLUS = CHRF(word_order=2)

DEFAULT_TRANSLATION_PROMPT_PROFILE = TranslationPromptProfile(
    name="default_turkish_v3",
    system_prompt=TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=TRANSLATION_USER_PROMPT_TEMPLATE,
)
AKT2_TRANSLATION_PROMPT_PROFILE = TranslationPromptProfile(
    name="akt2_turkish_v4",
    system_prompt=AKT2_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=AKT2_TRANSLATION_USER_PROMPT_TEMPLATE,
)
AKT3_TRANSLATION_PROMPT_PROFILE = TranslationPromptProfile(
    name="akt3_german_v1",
    system_prompt=AKT3_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=AKT3_TRANSLATION_USER_PROMPT_TEMPLATE,
)
SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE = TranslationPromptProfile(
    name="side_by_side_translation_v1",
    system_prompt=SIDE_BY_SIDE_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=SIDE_BY_SIDE_TRANSLATION_USER_PROMPT_TEMPLATE,
)


HECKER_ICK4_PDF_NAME = (
    "Hecker Kryszat Matous - Kappadokische Keilschrifttafeln aus den "
    "Sammlungen der Karlsuniversitat Prag. ICK 4 1998.pdf"
)
HECKER_ICK4_VOLUME_NAME = (
    "HECKER KRYSZAT MATOUS - KAPPADOKISCHE KEILSCHRIFTTAFELN AUS DEN "
    "SAMMLUNGEN DER KARLSUNIVERSITAT PRAG. ICK 4 1998"
)

TRANSLATION_PROMPT_PROFILES_BY_PDF_KEY: dict[str, TranslationPromptProfile] = {
    "AKT21995FIXEDPDF": AKT2_TRANSLATION_PROMPT_PROFILE,
    "AKT31995FIXEDPDF": AKT3_TRANSLATION_PROMPT_PROFILE,
    "AKT12PDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT52008PDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6APDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6BPDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6CPDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6DPDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6EPDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT82015PDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT42006PDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT7APDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT7BPDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT9APDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT10PDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT11APDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT11BPDF": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    re.sub(r"[^A-Z0-9]+", "", HECKER_ICK4_PDF_NAME.upper()): AKT3_TRANSLATION_PROMPT_PROFILE,
    "LARSEN2002THEASSURNADAARCHIVEPIHANS962002PDF": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
}
TRANSLATION_PROMPT_PROFILES_BY_VOLUME_KEY: dict[str, TranslationPromptProfile] = {
    "AKT12": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT5": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6A": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6B": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6C": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6D": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT6E": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT8": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
    "AKT4": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT7A": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT7B": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT9A": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT10": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT11A": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    "AKT11B": DEFAULT_TRANSLATION_PROMPT_PROFILE,
    re.sub(r"[^A-Z0-9]+", "", HECKER_ICK4_VOLUME_NAME.upper()): AKT3_TRANSLATION_PROMPT_PROFILE,
    "LARSEN2002THEASSURNADAARCHIVEPIHANS962002": SIDE_BY_SIDE_TRANSLATION_PROMPT_PROFILE,
}

TRANSLATION_CACHE_VERSION = "translation_prompt_profiles_v5"


@dataclass
class Args:
    input_csv: str = (
        "./data/find_excavation_number_pages/"
        "published_texts_akt_subset_with_locations.csv"
    )
    transliteration_pairs_csv: str = ""
    pdf_root: str = "./input/pdfs"
    output_root: str = "./data/extract_excavation_translation_pairs_from_locations"
    run_name: str = ""
    dpi: int = 140
    render_workers: int = 32
    context_next_pages: int = 2
    require_found: bool = True
    target_volumes: list[str] = field(default_factory=list)
    target_pdf_names: list[str] = field(default_factory=list)
    input_row_indices: list[int] = field(default_factory=list)
    limit_rows: int = -1
    dryrun: bool = False

    model: str = "Qwen/Qwen3.5-27B-FP8"
    translate_batch_size: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    translate_max_tokens: int = 4096
    translate_num_samples: int = 5
    use_structured_outputs: bool = True

    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 16384
    seed: int = 42
    trust_remote_code: bool = False
    enforce_eager: bool = False


def parse_first_json_object(text: str) -> dict:
    content = (text or "").strip()
    if not content:
        return {}

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def extract_final_answer(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_bool(text: str) -> bool:
    return normalize_ws(text).lower() in {"1", "true", "t", "yes", "y"}


def normalize_filter_key(text: str) -> str:
    return normalize_ws(text).upper()


def normalize_prompt_lookup_key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_ws(text).upper())


def normalize_excavation_lookup_key(text: str) -> str:
    return normalize_ws(text).casefold()


def select_translation_prompt_profile(pdf_name: str, akt_volume: str) -> TranslationPromptProfile:
    pdf_key = normalize_prompt_lookup_key(pdf_name)
    prompt_profile = TRANSLATION_PROMPT_PROFILES_BY_PDF_KEY.get(pdf_key)
    if prompt_profile is not None:
        return prompt_profile

    volume_key = normalize_prompt_lookup_key(akt_volume)
    prompt_profile = TRANSLATION_PROMPT_PROFILES_BY_VOLUME_KEY.get(volume_key)
    if prompt_profile is not None:
        return prompt_profile

    return DEFAULT_TRANSLATION_PROMPT_PROFILE


def format_transliteration_for_prompt(text: str) -> str:
    transliteration = (text or "").strip()
    if transliteration:
        return transliteration
    return "(unavailable)"


def infer_transliteration_pairs_csv_path(input_csv: Path) -> Path | None:
    candidate = (
        Path("./data/extract_excavation_transliteration_pairs_from_locations")
        / input_csv.stem
        / "pairs.csv"
    )
    if candidate.exists():
        return candidate
    return None


def resolve_transliteration_pairs_csv_path(args: Args, input_csv: Path) -> Path | None:
    configured_path = normalize_ws(args.transliteration_pairs_csv)
    if configured_path:
        path = Path(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"Transliteration pairs CSV not found: {path}")
        return path
    return infer_transliteration_pairs_csv_path(input_csv)


def load_transliteration_lookup(pairs_csv_path: Path) -> dict[str, str]:
    with pairs_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {pairs_csv_path}")
        if "excavation_number" not in reader.fieldnames or "transliteration" not in reader.fieldnames:
            raise ValueError(
                "Transliteration pairs CSV must contain "
                f"'excavation_number' and 'transliteration': {pairs_csv_path}"
            )

        lookup: dict[str, str] = {}
        for raw_row in reader:
            excavation_number = normalize_ws(raw_row.get("excavation_number", ""))
            transliteration = normalize_ws(raw_row.get("transliteration", ""))
            if not excavation_number or not transliteration:
                continue

            key = normalize_excavation_lookup_key(excavation_number)
            existing = lookup.get(key, "")
            if not existing or len(transliteration) > len(existing):
                lookup[key] = transliteration
    return lookup


def build_excavation_lookup_keys(row: dict[str, str]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for value in (
        row.get("canonical_excavation_number", ""),
        row.get("excavation_no", ""),
        row.get("oare_id", ""),
    ):
        key = normalize_excavation_lookup_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def safe_int(text: str, default: int) -> int:
    value = normalize_ws(text)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_page_list(text: str) -> list[int]:
    raw = normalize_ws(text)
    if not raw:
        return []

    values: list[int] = []
    for part in re.split(r"[;,|]", raw):
        token = normalize_ws(part)
        if not token:
            continue
        if token.isdigit():
            values.append(int(token))
            continue
        range_match = PAGE_RANGE_RE.match(token)
        if range_match is not None:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                start, end = end, start
            values.extend(range(start, end + 1))
            continue
        return []

    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def chunk_page_tasks(
    page_tasks: list[tuple[int, Path]],
    chunk_count: int,
) -> list[list[tuple[int, Path]]]:
    if not page_tasks:
        return []
    chunk_size = max(1, (len(page_tasks) + chunk_count - 1) // chunk_count)
    return [page_tasks[i : i + chunk_size] for i in range(0, len(page_tasks), chunk_size)]


def render_pdf_page_chunk(
    pdf_path_str: str,
    page_tasks: list[tuple[int, str]],
    scale: float,
) -> tuple[int, list[tuple[int, str]]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path_str)
    try:
        rendered_count = 0
        rendered_paths: list[tuple[int, str]] = []
        for page_num, image_path_str in page_tasks:
            image_path = Path(image_path_str)
            if not image_path.exists():
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                rendered_count += 1
            rendered_paths.append((page_num, image_path_str))
        return rendered_count, rendered_paths
    finally:
        doc.close()


def render_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    pages: list[int],
    dpi: int,
    render_workers: int,
) -> tuple[dict[int, Path], list[int]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    requested_pages = sorted(set(page for page in pages if page >= 1))
    if not requested_pages:
        return {}, []

    try:
        import pypdfium2 as pdfium
    except ImportError:
        rendered: dict[int, Path] = {}
        for page_num in requested_pages:
            for path in sorted(images_dir.glob(f"page_{page_num:03d}.*")):
                if PAGE_IMAGE_RE.match(path.name) is None:
                    continue
                rendered[page_num] = path
                break
        missing_pages = [page_num for page_num in requested_pages if page_num not in rendered]
        if missing_pages:
            raise ImportError(
                "pypdfium2 is required to render missing pages. "
                "Install with: pip install pypdfium2"
            )
        print(
            "[warn] pypdfium2 is not installed. "
            f"Using existing cached images only: pdf={pdf_path.name}, pages={len(rendered)}"
        )
        return rendered, []

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        valid_pages = [page_num for page_num in requested_pages if page_num <= total_pages]
        invalid_pages = [page_num for page_num in requested_pages if page_num > total_pages]

        scale = dpi / 72.0
        rendered: dict[int, Path] = {}
        num_skipped = 0
        pending_page_tasks: list[tuple[int, Path]] = []
        for page_num in valid_pages:
            image_path = images_dir / f"page_{page_num:03d}.png"
            if image_path.exists():
                num_skipped += 1
                rendered[page_num] = image_path
                continue
            pending_page_tasks.append((page_num, image_path))

        num_rendered = 0
        worker_count = max(1, min(render_workers, len(pending_page_tasks))) if pending_page_tasks else 0
        if worker_count <= 1:
            for page_num, image_path in tqdm(
                pending_page_tasks,
                desc=f"render:{pdf_path.stem}",
                leave=False,
            ):
                page = doc[page_num - 1]
                pil_image = page.render(scale=scale).to_pil()
                pil_image.save(image_path, format="PNG")
                num_rendered += 1
                rendered[page_num] = image_path
        else:
            page_task_chunks = chunk_page_tasks(pending_page_tasks, worker_count)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        render_pdf_page_chunk,
                        str(pdf_path),
                        [(page_num, str(image_path)) for page_num, image_path in chunk],
                        scale,
                    )
                    for chunk in page_task_chunks
                ]
                with tqdm(total=len(pending_page_tasks), desc=f"render:{pdf_path.stem}", leave=False) as pbar:
                    for future in as_completed(futures):
                        rendered_count, rendered_paths = future.result()
                        num_rendered += rendered_count
                        for page_num, image_path_str in rendered_paths:
                            rendered[page_num] = Path(image_path_str)
                        pbar.update(len(rendered_paths))

        print(
            f"render_pdf done. pdf={pdf_path.name}, pages={len(valid_pages)}, "
            f"rendered={num_rendered}, skipped_existing={num_skipped}, invalid={len(invalid_pages)}"
        )
        return rendered, invalid_pages
    finally:
        doc.close()


def build_image_context(
    rendered_pages: dict[int, Path],
    target_page: int,
    next_pages: int,
) -> list[tuple[str, int, Path]]:
    image_context: list[tuple[str, int, Path]] = []
    target_image = rendered_pages.get(target_page)
    if target_image is None:
        return image_context

    image_context.append(("target", target_page, target_image))
    for offset in range(1, next_pages + 1):
        next_page = target_page + offset
        next_image = rendered_pages.get(next_page)
        if next_image is not None:
            image_context.append(("next", next_page, next_image))
    return image_context


def make_translation_messages(
    excavation_number: str,
    page_number: int,
    transliteration: str,
    image_context: list[tuple[str, int, Path]],
    prompt_profile: TranslationPromptProfile,
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": prompt_profile.user_prompt_template.format(
                excavation_number=excavation_number,
                page_number=page_number,
                transliteration=transliteration,
            ),
        }
    ]
    role_to_label = {"target": "TARGET page", "next": "NEXT context page"}
    for role, ctx_page, image_path in image_context:
        content.append({"type": "text", "text": f"{role_to_label[role]}: page {ctx_page}"})
        content.append({"type": "image_url", "image_url": {"url": f"file://{image_path.resolve()}"}})

    return [
        {"role": "system", "content": prompt_profile.system_prompt},
        {"role": "user", "content": content},
    ]


def build_sampling_params(args: Args, schema: dict, max_tokens: int, num_samples: int = 1) -> SamplingParams:
    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": max_tokens,
    }
    if num_samples > 1:
        sampling_kwargs["n"] = num_samples
    if args.use_structured_outputs:
        if StructuredOutputsParams is None:
            print("[warn] StructuredOutputsParams is not available. Continue without constrained decoding.")
        else:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=schema,
                disable_fallback=True,
            )
    return SamplingParams(**sampling_kwargs)


def is_likely_summary_only(text: str) -> bool:
    normalized = normalize_ws(text)
    if not normalized:
        return False
    if INTRO_SUMMARY_CUE_RE.search(normalized) and not LINE_MARKER_RE.search(normalized):
        return True
    return False


def strip_leading_summary(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    range_match = TRANSLATION_LINE_RANGE_RE.search(raw)
    if range_match:
        marker_start = range_match.start(1)
        prefix = raw[:marker_start]
        if INTRO_SUMMARY_CUE_RE.search(prefix) or TRANSLITERATION_SECTION_CUE_RE.search(prefix):
            return normalize_ws(raw[marker_start:])

    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) == 2:
        head = normalize_ws(parts[0])
        tail = normalize_ws(parts[1])
        if INTRO_SUMMARY_CUE_RE.search(head):
            return tail

    start_match = TRANSLATION_START_CUE_RE.search(raw)
    if start_match and INTRO_SUMMARY_CUE_RE.search(raw[: start_match.start()]):
        return normalize_ws(raw[start_match.start() :])

    return normalize_ws(raw)


def normalize_translation(parsed: dict, fallback_excavation_number: str) -> tuple[str, str]:
    excavation_number = normalize_ws(str(parsed.get("excavation_number", "")))
    translation = strip_leading_summary(str(parsed.get("translation", "")))
    if is_likely_summary_only(translation):
        translation = ""
    if not excavation_number:
        excavation_number = fallback_excavation_number
    return excavation_number, translation


def build_translation_candidates(out, fallback_excavation_number: str) -> list[dict]:
    candidates: list[dict] = []
    for candidate_index, output in enumerate(getattr(out, "outputs", []) or []):
        raw_text = output.text if output is not None else ""
        parsed = parse_first_json_object(extract_final_answer(raw_text))
        response_excavation_number, translation = normalize_translation(
            parsed=parsed,
            fallback_excavation_number=fallback_excavation_number,
        )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "raw_text": raw_text,
                "parsed": parsed,
                "response_excavation_number": response_excavation_number,
                "translation": translation,
            }
        )
    return candidates


def pairwise_chrfpp_score(left: str, right: str) -> float:
    left_norm = normalize_ws(left)
    right_norm = normalize_ws(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(CHRF_PLUS_PLUS.sentence_score(left_norm, [right_norm]).score)


def select_best_translation_candidate(
    candidates: list[dict],
    fallback_excavation_number: str,
) -> tuple[dict, dict]:
    if not candidates:
        empty_candidate = {
            "candidate_index": -1,
            "raw_text": "",
            "parsed": {},
            "response_excavation_number": fallback_excavation_number,
            "translation": "",
        }
        return empty_candidate, {
            "translation_candidate_count": 0,
            "translation_selected_candidate_index": -1,
            "translation_selected_candidate_mean_chrfpp": 0.0,
        }

    best_candidate: dict | None = None
    best_key: tuple[float, float, int, int] | None = None

    for candidate in candidates:
        translation = normalize_ws(str(candidate.get("translation", "")))
        peer_scores = [
            pairwise_chrfpp_score(translation, str(other.get("translation", "")))
            for other in candidates
            if other is not candidate
        ]
        mean_chrfpp = sum(peer_scores) / len(peer_scores) if peer_scores else 0.0
        candidate["mean_chrfpp"] = mean_chrfpp
        candidate_key = (
            1.0 if translation else 0.0,
            mean_chrfpp,
            len(translation),
            -int(candidate.get("candidate_index", 0)),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate, {
        "translation_candidate_count": len(candidates),
        "translation_selected_candidate_index": int(best_candidate.get("candidate_index", -1)),
        "translation_selected_candidate_mean_chrfpp": float(best_candidate.get("mean_chrfpp", 0.0)),
    }


def empty_translations_df(input_columns: list[str]) -> pl.DataFrame:
    schema = {column: pl.String for column in input_columns}
    schema.update(
        {
            "transliteration": pl.String,
            "page_candidates": pl.String,
            "selected_target_page": pl.Int64,
            "context_pages": pl.String,
            "image_path": pl.String,
            "pdf_path": pl.String,
            "translation_request_excavation_number": pl.String,
            "translation_prompt_profile": pl.String,
            "translation_candidate_count": pl.Int64,
            "translation_selected_candidate_index": pl.Int64,
            "translation_selected_candidate_mean_chrfpp": pl.Float64,
            "response_excavation_number": pl.String,
            "translation": pl.String,
            "translation_found": pl.Boolean,
            "cache_hit": pl.Boolean,
            "error": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)


def empty_skipped_df(input_columns: list[str]) -> pl.DataFrame:
    schema = {column: pl.String for column in input_columns}
    schema.update({"skip_reason": pl.String})
    return pl.DataFrame(schema=schema)


def merge_translations_by_excavation(records: list[dict]) -> list[dict]:
    ordered_rows = sorted(records, key=lambda row: safe_int(str(row.get("input_row_index", "")), 10**9))
    merged: OrderedDict[str, dict] = OrderedDict()

    for row in ordered_rows:
        translation = normalize_ws(str(row.get("translation", "")))
        if not translation:
            continue

        key = normalize_ws(str(row.get("canonical_excavation_number", "")))
        if not key:
            key = normalize_ws(str(row.get("response_excavation_number", "")))
        if not key:
            key = normalize_ws(str(row.get("excavation_no", "")))
        if not key:
            continue

        current = {
            "canonical_excavation_number": key,
            "translation": translation,
            "source_input_row_index": str(row.get("input_row_index", "")),
            "oare_id": str(row.get("oare_id", "")),
            "akt_volume": str(row.get("akt_volume", "")),
            "pdf_names": str(row.get("pdf_names", "")),
            "selected_target_page": str(row.get("selected_target_page", "")),
        }
        existing = merged.get(key)
        if existing is None or len(translation) > len(str(existing.get("translation", ""))):
            merged[key] = current

    return list(merged.values())


def shutdown_runtime(llm: LLM | None) -> None:
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", None)
            core_client = getattr(engine, "engine_core", None)
            shutdown = getattr(core_client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as e:
            print(f"[warn] LLM shutdown failed: {e}")

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_input_rows(args: Args) -> tuple[list[str], list[dict], list[dict]]:
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    transliteration_pairs_csv_path = resolve_transliteration_pairs_csv_path(args, input_csv)
    transliteration_lookup: dict[str, str] = {}
    if transliteration_pairs_csv_path is not None:
        transliteration_lookup = load_transliteration_lookup(transliteration_pairs_csv_path)
        print(
            "[info] Loaded transliteration lookup: "
            f"path={transliteration_pairs_csv_path}, pairs={len(transliteration_lookup)}"
        )

    volume_filter = {normalize_filter_key(value) for value in args.target_volumes if normalize_ws(value)}
    pdf_filter = {normalize_ws(value) for value in args.target_pdf_names if normalize_ws(value)}
    row_index_filter = set(args.input_row_indices)

    working_rows: list[dict] = []
    skipped_rows: list[dict] = []
    existing_transliteration_count = 0
    lookup_transliteration_count = 0
    unavailable_transliteration_count = 0

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        input_columns = list(reader.fieldnames)

        for raw_row in reader:
            row = {key: (value if value is not None else "") for key, value in raw_row.items()}
            input_row_index = safe_int(row.get("input_row_index", ""), 10**9)

            if row_index_filter and input_row_index not in row_index_filter:
                continue
            if volume_filter and normalize_filter_key(row.get("akt_volume", "")) not in volume_filter:
                continue
            if pdf_filter and normalize_ws(row.get("pdf_names", "")) not in pdf_filter:
                continue

            if args.require_found and not parse_bool(row.get("found", "")):
                skipped_rows.append({**row, "skip_reason": "found_is_false"})
                continue

            pdf_name = normalize_ws(row.get("pdf_names", ""))
            if not pdf_name:
                skipped_rows.append({**row, "skip_reason": "missing_pdf_name"})
                continue

            page_candidates = parse_page_list(row.get("pages", ""))
            if not page_candidates:
                skipped_rows.append({**row, "skip_reason": "invalid_pages"})
                continue

            pdf_path = Path(args.pdf_root) / pdf_name
            if not pdf_path.exists():
                skipped_rows.append({**row, "skip_reason": "missing_pdf_file"})
                continue

            excavation_hint = normalize_ws(row.get("canonical_excavation_number", ""))
            if not excavation_hint:
                excavation_hint = normalize_ws(row.get("excavation_no", ""))
            if not excavation_hint:
                excavation_hint = normalize_ws(row.get("oare_id", ""))
            translation_prompt_profile = select_translation_prompt_profile(
                pdf_name=pdf_name,
                akt_volume=row.get("akt_volume", ""),
            )
            transliteration = normalize_ws(row.get("transliteration", ""))
            if transliteration:
                existing_transliteration_count += 1
            elif transliteration_lookup:
                for lookup_key in build_excavation_lookup_keys(row):
                    transliteration = transliteration_lookup.get(lookup_key, "")
                    if transliteration:
                        lookup_transliteration_count += 1
                        break

            transliteration = format_transliteration_for_prompt(transliteration)
            if transliteration == "(unavailable)":
                unavailable_transliteration_count += 1

            source_row = dict(row)
            source_row["transliteration"] = transliteration

            working_rows.append(
                {
                    "source_row": source_row,
                    "input_row_index": input_row_index,
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_candidates": page_candidates,
                    "target_page": page_candidates[0],
                    "excavation_hint": excavation_hint,
                    "transliteration": transliteration,
                    "translation_prompt_profile": translation_prompt_profile,
                }
            )

    working_rows.sort(key=lambda row: row["input_row_index"])
    if args.limit_rows > 0:
        working_rows = working_rows[: args.limit_rows]
    print(
        "[info] Prepared translation rows: "
        f"rows={len(working_rows)}, input_transliterations={existing_transliteration_count}, "
        f"lookup_filled={lookup_transliteration_count}, unavailable={unavailable_transliteration_count}"
    )
    return input_columns, working_rows, skipped_rows


def build_translation_record(
    work_row: dict,
    image_context: list[tuple[str, int, Path]],
    response_excavation_number: str,
    translation: str,
    selection_metadata: dict | None,
    cache_hit: bool,
    error: str,
) -> dict:
    source_row = dict(work_row["source_row"])
    image_path = ""
    if image_context:
        image_path = str(image_context[0][2])
    selection_metadata = selection_metadata or {}

    return {
        **source_row,
        "transliteration": work_row["transliteration"],
        "page_candidates": ",".join(str(page) for page in work_row["page_candidates"]),
        "selected_target_page": int(work_row["target_page"]),
        "context_pages": ",".join(str(page) for _, page, _ in image_context),
        "image_path": image_path,
        "pdf_path": str(work_row["pdf_path"]),
        "translation_request_excavation_number": work_row["excavation_hint"],
        "translation_prompt_profile": work_row["translation_prompt_profile"].name,
        "translation_candidate_count": int(selection_metadata.get("translation_candidate_count", 0)),
        "translation_selected_candidate_index": int(
            selection_metadata.get("translation_selected_candidate_index", -1)
        ),
        "translation_selected_candidate_mean_chrfpp": float(
            selection_metadata.get("translation_selected_candidate_mean_chrfpp", 0.0)
        ),
        "response_excavation_number": response_excavation_number,
        "translation": translation,
        "translation_found": bool(normalize_ws(translation)),
        "cache_hit": cache_hit,
        "error": error,
    }


def main(args: Args):
    if args.translate_batch_size < 1:
        raise ValueError("translate_batch_size must be >= 1")
    if args.translate_num_samples < 1:
        raise ValueError("translate_num_samples must be >= 1")
    if args.render_workers < 1:
        raise ValueError("render_workers must be >= 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if args.context_next_pages < 0:
        raise ValueError("context_next_pages must be >= 0")

    input_columns, working_rows, skipped_rows = load_input_rows(args)
    if args.dryrun:
        dryrun_n = min(5, len(working_rows))
        working_rows = working_rows[:dryrun_n]
        print(f"[dryrun] Enabled. Using first {dryrun_n} rows from {args.input_csv}.")

    input_csv = Path(args.input_csv)
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        base_run_name = input_csv.stem or "run"
        run_name = f"{base_run_name}_dryrun" if args.dryrun else base_run_name

    run_dir = Path(args.output_root) / run_name
    images_root = run_dir / "images"
    translations_dir = run_dir / "translations_by_record"
    output_translations_path = run_dir / "translations_by_record.csv"
    output_pairs_path = run_dir / "translations_unique_by_excavation.csv"
    output_missing_path = run_dir / "missing_translations.csv"
    output_skipped_path = run_dir / "skipped_rows.csv"
    translate_raw_output_path = run_dir / "raw_translate_responses.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    translations_dir.mkdir(parents=True, exist_ok=True)

    pdf_to_pages: dict[str, set[int]] = defaultdict(set)
    for work_row in working_rows:
        for offset in range(0, args.context_next_pages + 1):
            pdf_to_pages[work_row["pdf_name"]].add(work_row["target_page"] + offset)

    rendered_by_pdf: dict[str, dict[int, Path]] = {}
    invalid_pages_by_pdf: dict[str, set[int]] = {}
    for pdf_name in tqdm(sorted(pdf_to_pages), desc="render_pdfs"):
        pdf_path = Path(args.pdf_root) / pdf_name
        images_dir = images_root / pdf_path.stem
        rendered, invalid_pages = render_pdf_pages(
            pdf_path=pdf_path,
            images_dir=images_dir,
            pages=sorted(pdf_to_pages[pdf_name]),
            dpi=args.dpi,
            render_workers=args.render_workers,
        )
        rendered_by_pdf[pdf_name] = rendered
        invalid_pages_by_pdf[pdf_name] = set(invalid_pages)

    if torch.cuda.is_available() and args.tensor_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"tensor_parallel_size={args.tensor_parallel_size} exceeds CUDA devices={torch.cuda.device_count()}"
        )

    quantization = (args.quantization or "").strip() or None
    if quantization is None and "fp8" in args.model.lower():
        quantization = "fp8"

    llm: LLM | None = None
    translation_records: list[dict] = []
    translate_raw_records: list[dict] = []
    try:
        if working_rows:
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                quantization=quantization,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                enforce_eager=args.enforce_eager,
                compilation_config={"mode": "none", "cudagraph_mode": "none"} if args.enforce_eager else None,
                allowed_local_media_path=str(images_root.resolve()),
                limit_mm_per_prompt={"image": 1 + args.context_next_pages},
            )

        translate_sampling = build_sampling_params(
            args=args,
            schema=TRANSLATION_OUTPUT_JSON_SCHEMA,
            max_tokens=args.translate_max_tokens,
            num_samples=args.translate_num_samples,
        )

        pending_translate: list[tuple[dict, Path, Path, list[tuple[str, int, Path]]]] = []
        for work_row in working_rows:
            row_idx = work_row["input_row_index"]
            translation_csv_path = translations_dir / f"row_{row_idx:05d}_translation.csv"
            raw_json_path = translations_dir / f"row_{row_idx:05d}_raw.json"

            if translation_csv_path.exists() and raw_json_path.exists():
                try:
                    raw_payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
                    cached_df = pl.read_csv(translation_csv_path)
                    cached_rows = cached_df.to_dicts()
                    if raw_payload.get("translation_cache_version") == TRANSLATION_CACHE_VERSION and len(
                        cached_rows
                    ) == 1:
                        cached_row = {**cached_rows[0], "cache_hit": True}
                        cached_target_page = safe_int(str(cached_row.get("selected_target_page", "")), -1)
                        cached_pdf_name = normalize_ws(str(cached_row.get("pdf_names", "")))
                        cached_request_excavation = normalize_ws(
                            str(cached_row.get("translation_request_excavation_number", ""))
                        )
                        cached_transliteration = normalize_ws(str(cached_row.get("transliteration", "")))
                        cached_prompt_profile = normalize_ws(
                            str(
                                cached_row.get(
                                    "translation_prompt_profile",
                                    raw_payload.get("translation_prompt_profile", ""),
                                )
                            )
                        )
                        cached_num_samples = safe_int(str(raw_payload.get("translate_num_samples", "")), -1)
                        if (
                            cached_target_page == int(work_row["target_page"])
                            and cached_pdf_name == work_row["pdf_name"]
                            and cached_request_excavation == work_row["excavation_hint"]
                            and cached_transliteration == normalize_ws(work_row["transliteration"])
                            and cached_prompt_profile == work_row["translation_prompt_profile"].name
                            and cached_num_samples == args.translate_num_samples
                        ):
                            translation_records.append(cached_row)
                            translate_raw_records.append(raw_payload)
                            continue
                    print(f"[info] Translation cache is outdated or mismatched for row={row_idx}. Re-running.")
                except Exception as e:
                    print(f"[warn] Failed to load cached translation for row={row_idx}: {e}. Re-running.")

            if work_row["target_page"] in invalid_pages_by_pdf.get(work_row["pdf_name"], set()):
                translation_record = build_translation_record(
                    work_row=work_row,
                    image_context=[],
                    response_excavation_number=work_row["excavation_hint"],
                    translation="",
                    selection_metadata=None,
                    cache_hit=False,
                    error="target_page_out_of_range",
                )
                pl.DataFrame([translation_record]).write_csv(translation_csv_path)
                translation_records.append(translation_record)
                continue

            rendered_pages = rendered_by_pdf.get(work_row["pdf_name"], {})
            image_context = build_image_context(
                rendered_pages=rendered_pages,
                target_page=work_row["target_page"],
                next_pages=args.context_next_pages,
            )
            if not image_context:
                translation_record = build_translation_record(
                    work_row=work_row,
                    image_context=[],
                    response_excavation_number=work_row["excavation_hint"],
                    translation="",
                    selection_metadata=None,
                    cache_hit=False,
                    error="target_page_image_missing",
                )
                pl.DataFrame([translation_record]).write_csv(translation_csv_path)
                translation_records.append(translation_record)
                continue

            pending_translate.append((work_row, translation_csv_path, raw_json_path, image_context))

        if pending_translate and llm is not None:
            for start in tqdm(
                range(0, len(pending_translate), args.translate_batch_size),
                desc="extract_translations",
            ):
                end = min(start + args.translate_batch_size, len(pending_translate))
                batch = pending_translate[start:end]
                batch_messages = [
                    make_translation_messages(
                        excavation_number=work_row["excavation_hint"],
                        page_number=work_row["target_page"],
                        transliteration=work_row["transliteration"],
                        image_context=image_context,
                        prompt_profile=work_row["translation_prompt_profile"],
                    )
                    for work_row, _translation_csv_path, _raw_json_path, image_context in batch
                ]
                outputs = llm.chat(batch_messages, sampling_params=translate_sampling, use_tqdm=False)

                for (work_row, translation_csv_path, raw_json_path, image_context), out in zip(batch, outputs):
                    candidates = build_translation_candidates(
                        out=out,
                        fallback_excavation_number=work_row["excavation_hint"],
                    )
                    selected_candidate, selection_metadata = select_best_translation_candidate(
                        candidates=candidates,
                        fallback_excavation_number=work_row["excavation_hint"],
                    )
                    response_excavation_number = str(selected_candidate.get("response_excavation_number", ""))
                    translation = str(selected_candidate.get("translation", ""))
                    translation_record = build_translation_record(
                        work_row=work_row,
                        image_context=image_context,
                        response_excavation_number=response_excavation_number,
                        translation=translation,
                        selection_metadata=selection_metadata,
                        cache_hit=False,
                        error="",
                    )
                    pl.DataFrame([translation_record]).write_csv(translation_csv_path)
                    translation_records.append(translation_record)

                    raw_payload = {
                        "input_row_index": work_row["input_row_index"],
                        "pdf_name": work_row["pdf_name"],
                        "pdf_path": str(work_row["pdf_path"]),
                        "target_page": work_row["target_page"],
                        "page_candidates": work_row["page_candidates"],
                        "excavation_hint": work_row["excavation_hint"],
                        "transliteration": work_row["transliteration"],
                        "translation_prompt_profile": work_row["translation_prompt_profile"].name,
                        "translate_num_samples": args.translate_num_samples,
                        "context_pages": [page for _, page, _ in image_context],
                        "candidate_outputs": candidates,
                        "selected_candidate_index": selection_metadata["translation_selected_candidate_index"],
                        "selected_candidate_mean_chrfpp": selection_metadata[
                            "translation_selected_candidate_mean_chrfpp"
                        ],
                        "translation_cache_version": TRANSLATION_CACHE_VERSION,
                    }
                    raw_json_path.write_text(
                        json.dumps(raw_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    translate_raw_records.append(raw_payload)

        if translation_records:
            translations_df = pl.DataFrame(translation_records).sort("input_row_index")
        else:
            translations_df = empty_translations_df(input_columns)
        translations_df.write_csv(output_translations_path)

        missing_df = (
            translations_df.filter(
                pl.col("translation").fill_null("").str.strip_chars() == ""
            ).sort("input_row_index")
            if translation_records
            else empty_translations_df(input_columns)
        )
        missing_df.write_csv(output_missing_path)

        skipped_df = (
            pl.DataFrame(skipped_rows).sort("input_row_index")
            if skipped_rows
            else empty_skipped_df(input_columns)
        )
        skipped_df.write_csv(output_skipped_path)

        unique_rows = merge_translations_by_excavation(translations_df.to_dicts())
        unique_df = (
            pl.DataFrame(unique_rows).sort("canonical_excavation_number")
            if unique_rows
            else pl.DataFrame(
                schema={
                    "canonical_excavation_number": pl.String,
                    "translation": pl.String,
                    "source_input_row_index": pl.String,
                    "oare_id": pl.String,
                    "akt_volume": pl.String,
                    "pdf_names": pl.String,
                    "selected_target_page": pl.String,
                }
            )
        )
        unique_df.write_csv(output_pairs_path)

        translate_raw_records.sort(key=lambda row: safe_int(str(row.get("input_row_index", "")), 10**9))
        with translate_raw_output_path.open("w", encoding="utf-8") as f:
            for rec in translate_raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"Done. input_rows={len(working_rows)}, translated_rows={len(translations_df)}, "
            f"missing_translations={len(missing_df)}, skipped_rows={len(skipped_df)}, "
            f"unique_excavations={len(unique_df)}"
        )
        print(f"Run directory: {run_dir}")
        print(f"Per-record translations: {output_translations_path}")
        print(f"Unique translations: {output_pairs_path}")
        print(f"Missing translations: {output_missing_path}")
        print(f"Skipped rows: {output_skipped_path}")
        print(f"Translate raw responses: {translate_raw_output_path}")
    finally:
        shutdown_runtime(llm)


if __name__ == "__main__":
    main(tyro.cli(Args))
