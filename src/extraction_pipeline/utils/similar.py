"""Retrieve similar transliterations using token-level Levenshtein distance (rapidfuzz)."""

from __future__ import annotations

import numpy as np


def _require_levenshtein():
    if Levenshtein is None:
        raise ModuleNotFoundError("rapidfuzz is required for similarity-based few-shot retrieval. Install rapidfuzz to use extraction_pipeline.utils.similar.")
    return Levenshtein

try:
    from rapidfuzz.distance import Levenshtein
except ModuleNotFoundError:
    Levenshtein = None


def token_levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance on whitespace-split token sequences.

    Each token is treated as an atomic unit, so one substitution/insertion/deletion
    corresponds to one token difference.
    """
    return _require_levenshtein().distance(a.split(), b.split())


def token_levenshtein_normalized(a: str, b: str) -> float:
    """Normalized token-level Levenshtein distance in [0, 1].

    0 means identical token sequences, 1 means completely different.
    """
    return _require_levenshtein().normalized_distance(a.split(), b.split())


class SimilarTransliterationRetriever:
    """Retrieve k most similar transliterations from a corpus using token-level
    Levenshtein distance.

    Parameters
    ----------
    corpus : list[str]
        The transliteration strings to search over.
    metadata : list[dict] | None
        Optional per-row metadata (e.g. translations) returned alongside results.
    """

    def __init__(
        self,
        corpus: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        self.corpus = corpus
        self.metadata = metadata
        if metadata is not None and len(metadata) != len(corpus):
            raise ValueError("corpus and metadata must have the same length")

    def query(
        self,
        transliteration: str,
        k: int = 5,
        exclude_self: bool = True,
    ) -> list[dict]:
        """Find the k most similar transliterations to the query.

        Parameters
        ----------
        transliteration : str
            Query transliteration.
        k : int
            Number of neighbours to return.
        exclude_self : bool
            If True, exact string matches are excluded from results.

        Returns
        -------
        list[dict]
            Each dict contains 'index', 'transliteration', 'distance', 'normalized_distance',
            and any keys from metadata if provided.
        """
        query_tokens = transliteration.split()
        distances = np.array(
            [
                _require_levenshtein().distance(query_tokens, candidate.split())
                for candidate in self.corpus
            ]
        )

        # Mask exact matches if requested
        if exclude_self:
            exact_mask = np.array(
                [c == transliteration for c in self.corpus], dtype=bool
            )
            distances[exact_mask] = np.iinfo(distances.dtype).max

        # Get top-k indices (smallest distance)
        if k >= len(self.corpus):
            top_k_indices = np.argsort(distances)
        else:
            top_k_indices = np.argpartition(distances, k)[:k]
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        results = []
        for idx in top_k_indices:
            dist = int(distances[idx])
            if dist == np.iinfo(distances.dtype).max:
                continue
            candidate = self.corpus[idx]
            query_len = len(query_tokens)
            cand_len = len(candidate.split())
            max_len = max(query_len, cand_len)
            norm_dist = dist / max_len if max_len > 0 else 0.0

            result = {
                "index": int(idx),
                "transliteration": candidate,
                "distance": dist,
                "normalized_distance": norm_dist,
            }
            if self.metadata is not None:
                result.update(self.metadata[idx])
            results.append(result)

        return results
