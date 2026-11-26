from __future__ import annotations

import math


def hit_at_k(rels: list[bool], k: int) -> float:
    return float(any(rels[:k]))


def reciprocal_rank(rels: list[bool], k: int | None = None) -> float:
    for r, ok in enumerate(rels[:k] if k else rels, 1):
        if ok:
            return 1.0 / r
    return 0.0


def ndcg_at_k(rels: list[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, ok in enumerate(rels[:k], 1) if ok)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0
