"""Genre similarity scoring via cosine similarity on embeddings."""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def genre_score(embedding_a: list[float], embedding_b: list[float]) -> float:
    """Score genre compatibility between two tracks. Returns value in [0, 1].

    Maps cosine similarity from [-1, 1] to [0, 1].
    """
    sim = cosine_similarity(embedding_a, embedding_b)
    return (sim + 1.0) / 2.0
