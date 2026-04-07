"""Consolidated prompt templates for the extraction pipeline.

All system prompts, user prompt templates, and JSON schemas used across
the 4 extraction phases are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# PromptProfile dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptProfile:
    """A named pair of (system_prompt, user_prompt_template)."""

    name: str
    system_prompt: str
    user_prompt_template: str


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Detection prompts
# ═══════════════════════════════════════════════════════════════════════════

# --- Standard Kt.-style excavation numbers (AKT 2–11, etc.) ---

STANDARD_DETECT_SYSTEM_PROMPT = """You extract excavation numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY excavation numbers used as the main text heading/section heading, or on a metadata line such as "Fundnummer".
- Return only the excavation number itself, not running index, chapter number, parentheses, page title, or surrounding words.
- Valid examples: "Kt. v/k 8", "Kt. 88/k 110", "Kt. g/k 1b", "Kt. c/k 929+1118+1450", "Kt. y/t 4".
- If the page shows "Fundnummer: Kt. v/k 8", output only "Kt. v/k 8".
- If the page shows "1. Kt. 88/k 110" or "No. 1: 88/k 110", output only "Kt. 88/k 110".
- Ignore inline/body-text mentions, commentary, notes, bibliography, running headers, footnotes, and references.
- Ignore tables of contents, indexes, and list pages that are not the main edition text of a document.
- Keep the output in visual reading order.
- If no readable excavation number exists, return an empty list.
"""

STANDARD_DETECT_USER_PROMPT_TEMPLATE = """Extract excavation numbers from this page image.
Target page number: {page_number}
{heading_examples}
Important:
- Return only excavation numbers like "Kt. v/k 8", "Kt. 88/k 110", or "Kt. y/t 4".
- Do not include running indices such as "1." or "No. 1:".
- Do not include the word "Fundnummer:".
- Do not extract inline/body mentions that are not the main heading or metadata label for the document on this page.

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""

# --- AKT 1 "No. ###" headings ---

AKT1_DETECT_SYSTEM_PROMPT = """You extract AKT 1 heading numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY main heading numbers written like "No. 1", "No. 2", or "No. 27".
- Return only the heading number itself, normalized as "No. <number>".
- Ignore body-text references such as "a/k 533", "BIN IV 12", commentary, notes, bibliography, running headers, and footnotes.
- Ignore subsection labels such as "Tablet", "Zarf", "St. 2", and page numbers.
- Keep the output in visual reading order.
- If no readable heading number exists, return an empty list.
"""

AKT1_DETECT_USER_PROMPT_TEMPLATE = """Extract AKT 1 main heading numbers from this page image.
Target page number: {page_number}
{heading_examples}
Important:
- Extract only main headings like "No. 1", "No. 2", or "No. 27".
- Do not extract excavation references such as "a/k 533" from body text.
- Do not extract page numbers or subsection labels such as "Tablet", "Zarf", or "St. 2".

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""

# --- Larsen "N. Label" headings ---

LARSEN_DETECT_SYSTEM_PROMPT = """You extract main section headings from Larsen's scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY the main section/document heading shown for the edited text on the page.
- Return the heading exactly as printed, including the running section number and reference label.
- Valid examples: "4. C33", "5. CCT 4, lb", "6. TC 3, 95".
- Keep the heading label itself; do not rewrite it into Kt.-style notation.
- Ignore the transliteration text, translation text, body lines, running headers, footnotes, commentary, bibliography, and page numbers.
- Ignore table-of-contents or index pages that list many headings without presenting the edited text itself.
- Keep output in visual reading order.
- If no readable main heading exists, return an empty list.
"""

LARSEN_DETECT_USER_PROMPT_TEMPLATE = """Extract the main section headings from this Larsen edition page image.
Target page number: {page_number}
{heading_examples}
Important:
- Return the heading exactly as shown on the page.
- Include the leading running number when present, such as "4. C33".
- Do not convert the label into Kt.-style excavation notation.
- Do not extract body-text lines from the transliteration or translation columns.

Return JSON with schema:
{{
  "excavation_numbers": ["string"]
}}
"""

# --- ICK 4 "I ###" headings ---

ICK4_DETECT_SYSTEM_PROMPT = """\
You extract text inventory numbers from scholarly edition page images.
Return JSON only.

Rules:
- Extract ONLY inventory numbers used as main section headings, formatted as "I ###" (e.g., "I 427", "I 428", "I 637(b)").
- These headings appear prominently, often centered or on their own line, followed by a physical description of the tablet.
- Do NOT extract inventory numbers mentioned in commentary, footnotes, annotations ("Anmerkungen"), or cross-references.
- Do NOT extract page numbers, line numbers, or section markers (Vs., Rs., etc.).
- Keep output in visual reading order.
- If no heading is found, return an empty list.

Return JSON: {"headings": ["I 427", "I 428", ...]}"""

ICK4_DETECT_USER_PROMPT_TEMPLATE = """\
Extract text inventory numbers (section headings) from this page image.
Target page number: {page_number}
{heading_examples}
Return JSON with schema:
{{
  "headings": ["string"]
}}"""

# --- Detection JSON schemas ---

DETECT_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_numbers": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["excavation_numbers"],
    "additionalProperties": False,
}

ICK4_DETECT_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["headings"],
    "additionalProperties": False,
}

# --- Detection PromptProfiles ---

STANDARD_DETECT_PROFILE = PromptProfile(
    name="standard_detect_v1",
    system_prompt=STANDARD_DETECT_SYSTEM_PROMPT,
    user_prompt_template=STANDARD_DETECT_USER_PROMPT_TEMPLATE,
)

AKT1_DETECT_PROFILE = PromptProfile(
    name="akt1_detect_v1",
    system_prompt=AKT1_DETECT_SYSTEM_PROMPT,
    user_prompt_template=AKT1_DETECT_USER_PROMPT_TEMPLATE,
)

LARSEN_DETECT_PROFILE = PromptProfile(
    name="larsen_detect_v1",
    system_prompt=LARSEN_DETECT_SYSTEM_PROMPT,
    user_prompt_template=LARSEN_DETECT_USER_PROMPT_TEMPLATE,
)

ICK4_DETECT_PROFILE = PromptProfile(
    name="ick4_detect_v1",
    system_prompt=ICK4_DETECT_SYSTEM_PROMPT,
    user_prompt_template=ICK4_DETECT_USER_PROMPT_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Extraction prompts (transliteration + translation in one call)
# ═══════════════════════════════════════════════════════════════════════════

# --- Combined sequential: Turkish (AKT 2, 4, 7, 9, 10, 11) ---

COMBINED_EXTRACTION_SYSTEM_PROMPT = """You extract transliteration and Turkish translation from scholarly edition page images.
Return JSON only.

You are given:
- Page image(s) from the PDF
- The published transliteration text (from OARE database) for reference

Your task:
1. Extract the transliteration line by line as it appears on the page, with line numbers.
2. Extract the Turkish translation, grouped by the line ranges as written in the book (e.g., "1-3", "4-6").

Rules for transliteration:
- Line numbers always start from 1.
- The first line often has no printed line number on the page; treat it as line 1.
- Read each line from the page image and record its line number.
- Use the published transliteration as reference to help you read damaged or unclear signs.
- Section cues like "\u00f6y.", "ak.", "ay." indicate tablet sides; do not include them as line content.
- Preserve the original line numbering from the page.

Rules for translation:
- The translation block comes AFTER the transliteration block on the page.
- Translations are grouped by line ranges. These range markers may appear in several formats:
  - Parenthesised inline: (1-5), (7-8) — within the translation prose
  - Superscript at line start: ¹⁻³⁾, ⁴⁻⁷⁾ — small raised numbers before the translation text
  - Plain at line start: 1-3), 4-6) — regular-sized numbers followed by a closing parenthesis
- Identify the line-range markers and use them as JSON keys (e.g., "1-3", "4-6").
- STRIP the markers from the translation text values. Do NOT include "1-3)" or "(1-5)" in the text itself.
- Do NOT extract the Turkish summary prose that appears between the heading and transliteration.
- Do NOT include commentary, bibliography notes, or references.
- If no translation is found, return an empty object.
"""

COMBINED_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract the transliteration and Turkish translation for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}

Published transliteration (from OARE database, for reference):
{transliteration_orig}

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": {{
    "1": "transliteration of line 1",
    "2": "transliteration of line 2"
  }},
  "translation": {{
    "1-3": "Turkish translation for lines 1-3",
    "4-6": "Turkish translation for lines 4-6"
  }}
}}
"""

# --- Combined side-by-side: English (AKT 5, 6, 8, 12, Larsen) ---

COMBINED_SIDE_BY_SIDE_EXTRACTION_SYSTEM_PROMPT = """You extract transliteration and translation from scholarly edition page images with a side-by-side layout.
Return JSON only.

You are given:
- Page image(s) from the PDF
- The published transliteration text (from OARE database) for reference

Layout:
- LEFT side = Akkadian transliteration
- CENTER = line numbers (printed between the two columns)
- RIGHT side = translation

Your task:
1. Extract the transliteration line by line from the LEFT column, with line numbers.
2. Extract the translation from the RIGHT column line by line.

Multi-page handling:
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often starts on one page and continues onto the next page(s). You MUST extract ALL content across page boundaries — do not stop at the end of the first page.
- Merge all lines across pages into a single continuous line numbering (do not restart at 1 on continuation pages).

Rules for transliteration:
- Line numbers always start from 1.
- The first line often has no printed line number on the page; treat it as line 1.
- Read each line from the LEFT column and record its line number.
- Use the published transliteration as reference to help you read damaged or unclear signs.
- Section cues like "\u00f6y.", "ak.", "ay." indicate tablet sides; do not include them as line content.
- Preserve the original line numbering from the page.
- Continue numbering across pages (if page 1 ends at line 21, page 2 starts at line 22).

Rules for translation:
- Read only the RIGHT-HAND translation column/area.
- Ignore the LEFT-HAND transliteration column entirely for translation.
- The RIGHT column contains translation text arranged in BLOCKS. A block is a continuous run of text that may span many printed lines due to line wrapping. Blocks are separated ONLY by a completely blank row (a row with NO text at all on the right side).
- CRITICAL: Do NOT split based on sentence boundaries, periods, or semantic content. The ONLY valid split point is a completely blank row in the RIGHT column. Even if multiple sentences appear within one block, they belong together as one entry. A block may contain many sentences.
- Each block's first line is vertically aligned with a transliteration line number on the LEFT — that is the START line of the range. The END line is the last transliteration line before the next blank row (or end of text).
- Use line-range keys: "start-end" (e.g. "1-8", "9-12"). Use a single key (e.g. "13") only if the block covers exactly one transliteration line.
- Combine all text within one block into a single string (join wrapped lines with spaces).
- SKIP any italicised description/commentary paragraph that appears BEFORE the side-by-side layout.
- Do NOT include commentary, bibliography notes, or references.
- If no translation is found, return an empty object.
"""

COMBINED_SIDE_BY_SIDE_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract the transliteration and translation for this excavation number from the side-by-side page layout.
Excavation number: {excavation_number}
Target page number: {page_number}

Published transliteration (from OARE database, for reference):
{transliteration_orig}

Important:
- Read transliteration from the LEFT-SIDE column, line by line.
- Read translation from the RIGHT-SIDE column. Translation is organized in paragraphs separated by blank lines. Each paragraph covers a range of transliteration lines. Use line-range keys (e.g. "1-4").
- SKIP any italicised description/commentary paragraph before the side-by-side layout.

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": {{
    "1": "transliteration of line 1",
    "2": "transliteration of line 2"
  }},
  "translation": {{
    "1-3": "translation covering lines 1 to 3",
    "4": "translation of line 4"
  }}
}}
"""

# --- Combined sequential: German (AKT 3) ---

COMBINED_AKT3_EXTRACTION_SYSTEM_PROMPT = """You extract transliteration and German translation from scholarly edition page images.
Return JSON only.

You are given:
- Page image(s) from the PDF
- The published transliteration text (from OARE database) for reference

Your task:
1. Extract the transliteration line by line as it appears on the page, with line numbers.
2. Extract the German translation, grouped by the line ranges as written in the book (e.g., "1-3", "4-6").

Rules for transliteration:
- Line numbers always start from 1.
- The first line often has no printed line number on the page; treat it as line 1.
- Read each line from the page image and record its line number.
- Use the published transliteration as reference to help you read damaged or unclear signs.
- Section cues like "\u00f6y.", "ak.", "ay." indicate tablet sides; do not include them as line content.
- Preserve the original line numbering from the page.

Rules for translation:
- The translation block comes AFTER the transliteration block on the page.
- Translations are grouped by line ranges. These range markers may appear in several formats:
  - Parenthesised inline: (1-5), (7-8) — within the translation prose
  - Superscript at line start: ¹⁻³⁾, ⁴⁻⁷⁾ — small raised numbers before the translation text
  - Plain at line start: 1-3), 4-6) — regular-sized numbers followed by a closing parenthesis
- Identify the line-range markers and use them as JSON keys (e.g., "1-3", "4-6").
- STRIP the markers from the translation text values. Do NOT include "1-3)" or "(1-5)" in the text itself.
- Do NOT extract prose between heading and transliteration.
- Do NOT include commentary, bibliography notes, or references.
- If no translation is found, return an empty object.
"""

COMBINED_AKT3_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract the transliteration and German translation for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}

Published transliteration (from OARE database, for reference):
{transliteration_orig}

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": {{
    "1": "transliteration of line 1",
    "2": "transliteration of line 2"
  }},
  "translation": {{
    "1-3": "German translation for lines 1-3",
    "4-6": "German translation for lines 4-6"
  }}
}}
"""

# --- ICK 4 extraction (transliteration + German translation) ---

ICK4_EXTRACT_SYSTEM_PROMPT = """\
You extract Akkadian transliteration and German translation for a single text from scholarly edition page images.
Return JSON only.

You are given:
- Page image(s) from the PDF
- The published transliteration text (from OARE database) for reference

Rules:
- Each text has this structure:
  1) Text heading (e.g., "I 427")
  2) Physical description of the tablet (e.g., size, color, copy reference)
  3) Optional German introduction/commentary paragraph
  4) Akkadian transliteration block with line numbers (1, 5, 10, ...) and surface markers (Vs., Rs., u.K., o.K., l.S.)
  5) German translation block with line-range markers (e.g., "1-6)", "7-13)")
  6) Annotations section labeled "Anmerkungen:"
- Extract ONLY the transliteration and German translation for the specified text number.
- For transliteration:
  - Extract line by line with line numbers as keys.
  - Line numbers always start from 1.
  - The first line often has no printed line number on the page; treat it as line 1.
  - Read each line from the page image and record its line number.
  - Use the published transliteration as reference to help you read damaged or unclear signs.
  - Exclude surface markers (Vs., Rs., u.K., o.K., l.S.); do not include them as line content.
  - Exclude physical description, introduction, and annotations.
  - Preserve brackets [], angle brackets <>, and damage markers.
  - Reproduce fractions exactly as printed (e.g., 1/2, 1/3, 2/3, 5/6, 1/4). Do NOT convert them to Unicode fractions, superscripts, or subscripts.
  - Reproduce subscript/superscript index numbers in plain text (e.g., write "en\u2086" as "en6", "il\u2085" as "il5"). Do NOT use HTML tags like <sup> or <sub>.
  - Keep editorial marks such as ^{!}, ^{?}, <<...>> as they appear in the source.
- For translation:
  - Extract the German translation, grouped by the line ranges as written in the book.
  - Line-range markers may appear in several formats:
    - Parenthesised inline: (1-6), (7-13) — within the translation prose
    - Superscript at line start: ¹⁻⁶⁾, ⁷⁻¹³⁾ — small raised numbers before the text
    - Plain at line start: 1-6), 7-13) — regular-sized numbers followed by a closing parenthesis
  - Identify the line-range markers and use them as JSON keys (e.g., "1-6", "7-13").
  - STRIP the markers from the translation text values. Do NOT include "3-6)" or "(1-6)" in the text itself.
  - Stop before "Anmerkungen:" section.
  - Do NOT include annotations, commentary, or scholarly notes.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often starts on one page and continues onto the next page(s). You MUST extract ALL content across page boundaries — do not stop at the end of the first page.
- Merge content across pages in reading order.
- If the transliteration or translation is not readable, return an empty object for that field."""

ICK4_EXTRACT_USER_PROMPT_TEMPLATE = """\
Extract the Akkadian transliteration and German translation for this text.
Excavation number: {excavation_number}
Target page number: {page_number}

Published transliteration (from OARE database, for reference):
{transliteration_orig}

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": {{
    "1": "transliteration of line 1",
    "2": "transliteration of line 2"
  }},
  "translation": {{
    "1-6": "German translation for lines 1-6",
    "7-13": "German translation for lines 7-13"
  }}
}}"""

# --- Extraction PromptProfiles ---

COMBINED_EXTRACTION_PROFILE = PromptProfile(
    name="combined_extraction_v1",
    system_prompt=COMBINED_EXTRACTION_SYSTEM_PROMPT,
    user_prompt_template=COMBINED_EXTRACTION_USER_PROMPT_TEMPLATE,
)

COMBINED_SIDE_BY_SIDE_EXTRACTION_PROFILE = PromptProfile(
    name="combined_side_by_side_extraction_v1",
    system_prompt=COMBINED_SIDE_BY_SIDE_EXTRACTION_SYSTEM_PROMPT,
    user_prompt_template=COMBINED_SIDE_BY_SIDE_EXTRACTION_USER_PROMPT_TEMPLATE,
)

COMBINED_AKT3_EXTRACTION_PROFILE = PromptProfile(
    name="combined_akt3_extraction_v1",
    system_prompt=COMBINED_AKT3_EXTRACTION_SYSTEM_PROMPT,
    user_prompt_template=COMBINED_AKT3_EXTRACTION_USER_PROMPT_TEMPLATE,
)

ICK4_EXTRACT_PROFILE = PromptProfile(
    name="ick4_extract_v1",
    system_prompt=ICK4_EXTRACT_SYSTEM_PROMPT,
    user_prompt_template=ICK4_EXTRACT_USER_PROMPT_TEMPLATE,
)

COMBINED_PROFILE_NAMES = frozenset({
    COMBINED_EXTRACTION_PROFILE.name,
    COMBINED_SIDE_BY_SIDE_EXTRACTION_PROFILE.name,
    COMBINED_AKT3_EXTRACTION_PROFILE.name,
    ICK4_EXTRACT_PROFILE.name,
})


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 (legacy): Translation-only prompts
# ═══════════════════════════════════════════════════════════════════════════

# --- Default Turkish translation ---

TRANSLATION_SYSTEM_PROMPT = """You extract Turkish translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) Turkish summary prose (must be excluded)
  3) Akkadian transliteration block
  4) Turkish translation block (extract this only)
- Extract only Turkish translation text for the specified excavation number.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Transliteration often appears with section cues like "\u00f6y.", "ak.", "ay."; do not include it.
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

# --- AKT 2 Turkish (with commentary exclusion) ---

AKT2_TRANSLATION_SYSTEM_PROMPT = """You extract Turkish translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) Turkish summary prose (must be excluded)
  3) Akkadian transliteration block
  4) Turkish translation block (extract this only)
- Extract only Turkish translation text for the specified excavation number.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Transliteration often appears with section cues like "\u00f6y.", "ak.", "ay."; do not include it.
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

# --- AKT 3 German translation ---

AKT3_TRANSLATION_SYSTEM_PROMPT = """You extract German translation text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is:
  1) excavation heading
  2) prose before transliteration (must be excluded)
  3) Akkadian transliteration block
  4) German translation block (extract this only)
- Extract only German translation text for the specified excavation number.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order.
- Start extraction only after the Akkadian transliteration block for the same excavation.
- Do NOT extract prose between heading and transliteration.
- Transliteration often appears with section cues like "\u00f6y.", "ak.", "ay."; do not include it.
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

# --- Side-by-side translation (AKT 5, 6, 8, 12, Larsen) ---

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
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order within the translation column, page by page.
- Header lines at the top of the entry may contain metadata, references, or edition notes. Exclude those unless they are clearly part of the translation itself.
- Translation may not repeat line numbers because numbering is often shown only beside the transliteration column.
- Do not include transliteration, metadata, references, bibliography, headings, commentary, or section titles.
- If no readable translation is visible on the right side, return an empty translation string.

CRITICAL \u2014 Distinguish description/commentary from actual translation:
- Some entries begin with an ITALICISED scholarly description or summary paragraph ABOVE the side-by-side layout.
  Examples: "Unopened envelope of a contract for\u2026", "A damaged first person account of\u2026",
  "An unopened envelope recording a be'ulatu loan\u2026".
  These are the editor's commentary, NOT the translation of the Akkadian text.
- SKIP any such description paragraph entirely.
- The actual translation starts in the RIGHT-HAND column of the side-by-side layout,
  typically beginning with "Seal of \u2026", a personal name, a number + unit (e.g. "\u00bd mina"),
  "Thus \u2026", or similar content that corresponds to the transliteration on the left.
- If the ONLY text visible for this excavation number is a description/commentary with no
  side-by-side translation, return an empty translation string.
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
- SKIP any italicised description/commentary paragraph that appears BEFORE the side-by-side layout (e.g. "Unopened envelope \u2026", "A damaged \u2026"). Extract ONLY the actual translation from the right column.

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

# --- Translation-only PromptProfiles ---

DEFAULT_TRANSLATION_PROFILE = PromptProfile(
    name="default_turkish_v3",
    system_prompt=TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=TRANSLATION_USER_PROMPT_TEMPLATE,
)

AKT2_TRANSLATION_PROFILE = PromptProfile(
    name="akt2_turkish_v4",
    system_prompt=AKT2_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=AKT2_TRANSLATION_USER_PROMPT_TEMPLATE,
)

AKT3_TRANSLATION_PROFILE = PromptProfile(
    name="akt3_german_v1",
    system_prompt=AKT3_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=AKT3_TRANSLATION_USER_PROMPT_TEMPLATE,
)

SIDE_BY_SIDE_TRANSLATION_PROFILE = PromptProfile(
    name="side_by_side_translation_v1",
    system_prompt=SIDE_BY_SIDE_TRANSLATION_SYSTEM_PROMPT,
    user_prompt_template=SIDE_BY_SIDE_TRANSLATION_USER_PROMPT_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 (legacy): Transliteration-only prompts
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_TRANSLITERATION_SYSTEM_PROMPT = """You extract Akkadian transliteration text for a single excavation number from page images.
Return JSON only.

Rules:
- Page structure for each excavation is usually:
  1) excavation heading
  2) summary/introduction prose (must be excluded)
  3) Akkadian transliteration block (extract this only)
  4) translation block (must be excluded)
- Extract only Akkadian transliteration text for the specified excavation number.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order.
- Start extraction at the beginning of the transliteration block for the same excavation.
- Do NOT extract summary prose between heading and transliteration.
- Exclude section cue tokens such as "\u00f6y.", "oy.", "ak.", and "ay.".
- Exclude marginal or inline line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude editorial section markers such as "K.", "A.y.", "\u00dc.k.", and "S.k." if they appear inline.
- Stop before the translation block starts.
- Do not include translation text, headings, commentary, bibliography, metadata, or section titles.
- If only summary prose is visible and the transliteration block is not visible, return empty transliteration.
- If no Akkadian transliteration is readable, return an empty transliteration string.
"""

DEFAULT_TRANSLITERATION_USER_PROMPT_TEMPLATE = """Extract the Akkadian transliteration for this excavation number.
Excavation number: {excavation_number}
Target page number: {page_number}

Important:
- Extract only the transliteration corresponding to this excavation number.
- Exclude summary/introduction prose before the transliteration block.
- Exclude translation text that appears after the transliteration block.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "\u00f6y.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "\u00dc.k.", and "S.k.".

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": "string"
}}
"""

SIDE_BY_SIDE_TRANSLITERATION_SYSTEM_PROMPT = """You extract Akkadian transliteration text for a single excavation number from page images.
Return JSON only.

Rules:
- These pages use a side-by-side layout:
  1) LEFT side = Akkadian transliteration
  2) RIGHT side = translation
- Extract only the transliteration for the specified excavation number.
- Read only the LEFT-HAND transliteration column/area.
- Ignore the RIGHT-HAND translation column entirely, even if it is clearer or more complete.
- Do not interleave left-column transliteration with right-column translation.
- The image may contain multiple pages stitched vertically into a single image. The target page is at the top, and continuation pages follow below.
- The text for one excavation number often continues onto the next page(s). You MUST extract ALL content across page boundaries.
- Merge continuation text in reading order within the transliteration column, page by page.
- Header lines at the top of the entry may contain metadata, references, or edition notes. Exclude those unless they are clearly part of the transliteration itself.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "\u00f6y.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "\u00dc.k.", and "S.k.".
- Do not include translation text, metadata, references, bibliography, headings, commentary, or section titles.
- If no readable transliteration is visible on the left side, return an empty transliteration string.
"""

SIDE_BY_SIDE_TRANSLITERATION_USER_PROMPT_TEMPLATE = """Extract the Akkadian transliteration for this excavation number from the side-by-side page layout.
Excavation number: {excavation_number}
Target page number: {page_number}

Important:
- Extract only the transliteration corresponding to this excavation number.
- Read only the LEFT-SIDE transliteration column.
- Ignore the RIGHT-SIDE translation column completely.
- Exclude top metadata/reference lines before the transliteration begins.
- Exclude line-number tokens such as "5.", "10.", "15.", "20.", etc.
- Exclude section cue tokens such as "\u00f6y.", "oy.", "ak.", and "ay.".
- Exclude editorial section markers such as "K.", "A.y.", "\u00dc.k.", and "S.k.".

Return JSON with schema:
{{
  "excavation_number": "string",
  "transliteration": "string"
}}
"""

TRANSLITERATION_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_number": {"type": "string"},
        "transliteration": {"type": "string"},
    },
    "required": ["excavation_number", "transliteration"],
    "additionalProperties": False,
}

# --- Transliteration-only PromptProfiles ---

DEFAULT_TRANSLITERATION_PROFILE = PromptProfile(
    name="default_transliteration_v1",
    system_prompt=DEFAULT_TRANSLITERATION_SYSTEM_PROMPT,
    user_prompt_template=DEFAULT_TRANSLITERATION_USER_PROMPT_TEMPLATE,
)

SIDE_BY_SIDE_TRANSLITERATION_PROFILE = PromptProfile(
    name="side_by_side_transliteration_v1",
    system_prompt=SIDE_BY_SIDE_TRANSLITERATION_SYSTEM_PROMPT,
    user_prompt_template=SIDE_BY_SIDE_TRANSLITERATION_USER_PROMPT_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 (legacy): Generic pair extraction (extract_pairs.py)
# ═══════════════════════════════════════════════════════════════════════════

GENERIC_PAIRS_SYSTEM_PROMPT = """You extract Akkadian transliteration and translation pairs (English or Turkish) from scholarly edition page images.
Return JSON only.

Rules:
- Pair by visual layout and line order.
- Keep the original transliteration text as written.
- Remove obvious line numbers or editorial line markers from transliteration when they are not part of the text.
- Keep translation in the original language as written on the page (English or Turkish).
- Do not invent or paraphrase content.
- If no pair is readable, return an empty "pairs" array.
- If identifiable, include "excavation_number" exactly as written, otherwise use an empty string.
"""

GENERIC_PAIRS_USER_PROMPT_TEMPLATE = """Extract Akkadian transliteration/translation pairs from this page image.
The translation may be in English or Turkish; preserve the original language exactly as written.
Page number: {page_number}

Return JSON with schema:
{{
  "excavation_number": "string",
  "pairs": [
    {{
      "transliteration": "string",
      "translation": "string"
    }}
  ]
}}
"""

GENERIC_PAIRS_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "excavation_number": {"type": "string"},
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "transliteration": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["transliteration", "translation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["excavation_number", "pairs"],
    "additionalProperties": False,
}

GENERIC_PAIRS_PROFILE = PromptProfile(
    name="generic_pairs_v1",
    system_prompt=GENERIC_PAIRS_SYSTEM_PROMPT,
    user_prompt_template=GENERIC_PAIRS_USER_PROMPT_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Translation to English (Turkish/German → English)
# ═══════════════════════════════════════════════════════════════════════════

TRANSLATE_EN_SYSTEM_PROMPT_TURKISH = """You are an expert translator for Turkish translations of Akkadian texts.
Translate Turkish text into English under strict constraints.

Rules:
1) Keep every `<gap>` token exactly as `<gap>`.
2) Keep proper nouns exactly as written (people, places, deities, institutions).
3) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
4) If Akkadian transliteration fragments are mixed into the Turkish text, omit those fragments from the English output.
5) Prefer faithful/literal translation over paraphrase.
6) Output JSON only with this schema:
{"translation_en": string}
"""

TRANSLATE_EN_SYSTEM_PROMPT_GERMAN = """You are an expert translator for German translations of Akkadian texts.
Translate German text into English under strict constraints.

Rules:
1) Keep every `<gap>` token exactly as `<gap>`.
2) Keep proper nouns exactly as written (people, places, deities, institutions).
3) Keep transliterated technical terms, numbers, and measurement expressions unchanged unless grammar requires minimal adjustment.
4) If Akkadian transliteration fragments are mixed into the German text, omit those fragments from the English output.
5) Prefer faithful/literal translation over paraphrase.
6) Output JSON only with this schema:
{"translation_en": string}
"""

TRANSLATE_EN_USER_PROMPT_TEMPLATE = """Akkadian transliteration (reference only, do not translate):
{transliteration}
{few_shot_block}

{source_language} text to translate into English:
{source_text}
"""

TRANSLATE_EN_FEW_SHOT_BLOCK_TEMPLATE = """
Reference Akkadian transliteration -> English translation examples (few-shot style guidance only):
{few_shot_examples}"""

TRANSLATE_EN_RETRY_USER_PROMPT_SUFFIX = """

Important: The previous response could not be parsed as JSON.
Return only one valid JSON object matching exactly this schema:
{"translation_en": string}
Do not add explanations, notes, or markdown.
"""

TRANSLATE_EN_RETRY_SHORT_TRANSLATION_SUFFIX = """

Important: The previous translation was too short or incomplete.
Translate the entire source text faithfully.
Do not stop after the opening formula, first clause, or first sentence.
Return only one valid JSON object matching exactly this schema:
{"translation_en": string}
Do not add explanations, notes, or markdown.
"""

TRANSLATE_EN_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "translation_en": {"type": "string"},
    },
    "required": ["translation_en"],
    "additionalProperties": False,
}
