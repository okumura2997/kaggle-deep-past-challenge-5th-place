"""ChrF++ consensus selection for multi-sample LLM extraction.

Extracted from the duplicated implementations in extract_translation_pairs.py,
extract_transliteration_pairs.py, and extract_ick4.py.
"""

from __future__ import annotations

from sacrebleu.metrics import CHRF

from extraction_pipeline._json_utils import normalize_ws

CHRF_PLUS_PLUS = CHRF(word_order=2)


def pairwise_chrfpp_score(left: str, right: str) -> float:
    """Compute ChrF++ score between two strings."""
    left_norm = normalize_ws(left)
    right_norm = normalize_ws(right)
    if not left_norm or not right_norm:
        return 0.0
    return float(CHRF_PLUS_PLUS.sentence_score(left_norm, [right_norm]).score)


def select_best_candidate(
    candidates: list[dict],
    text_key: str = "translation",
    fallback_fields: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """Select the candidate with highest mean ChrF++ agreement with peers.

    Args:
        candidates: List of candidate dicts, each with a ``text_key`` field
            and a ``candidate_index`` field.
        text_key: The field name to use for ChrF++ scoring.
            For combined scoring (e.g. ICK4), callers should pre-compute
            a combined field and pass its name here.
        fallback_fields: Default field values for the empty-candidate case.

    Returns:
        Tuple of (best_candidate, selection_metadata).
    """
    prefix = text_key
    if not candidates:
        empty_candidate: dict = {
            "candidate_index": -1,
            "raw_text": "",
            "parsed": {},
        }
        if fallback_fields:
            empty_candidate.update(fallback_fields)
        return empty_candidate, {
            f"{prefix}_candidate_count": 0,
            f"{prefix}_selected_candidate_index": -1,
            f"{prefix}_selected_candidate_mean_chrfpp": 0.0,
        }

    best_candidate: dict | None = None
    best_key: tuple[float, float, int, int] | None = None

    for candidate in candidates:
        text = normalize_ws(str(candidate.get(text_key, "")))
        peer_scores = [
            pairwise_chrfpp_score(text, str(other.get(text_key, "")))
            for other in candidates
            if other is not candidate
        ]
        mean_chrfpp = sum(peer_scores) / len(peer_scores) if peer_scores else 0.0
        candidate["mean_chrfpp"] = mean_chrfpp
        candidate_key = (
            1.0 if text else 0.0,
            mean_chrfpp,
            len(text),
            -int(candidate.get("candidate_index", 0)),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate, {
        f"{prefix}_candidate_count": len(candidates),
        f"{prefix}_selected_candidate_index": int(
            best_candidate.get("candidate_index", -1)
        ),
        f"{prefix}_selected_candidate_mean_chrfpp": float(
            best_candidate.get("mean_chrfpp", 0.0)
        ),
    }
