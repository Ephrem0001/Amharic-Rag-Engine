"""
Retrieval evaluation metrics: Recall@k, MRR@k, nDCG@k, Hit Rate@k.

Ground truth: set of relevant chunk IDs (or page numbers). For each query we have
an ordered list of retrieved chunk IDs. Relevance is binary: 1 if in ground truth, else 0.
"""

from __future__ import annotations

import math
from typing import List, Set


def _relevance_vector(retrieved_ids: List[str], relevant_ids: Set[str]) -> List[int]:
    """Binary relevance for each retrieved item (1 if relevant, 0 otherwise)."""
    return [1 if cid in relevant_ids else 0 for cid in retrieved_ids]


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|.
    Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def recall_at_k_batch(
    list_retrieved: List[List[str]],
    list_relevant: List[Set[str]],
    k: int,
) -> float:
    """Mean Recall@k over a batch of (retrieved, relevant) pairs."""
    if not list_retrieved or len(list_retrieved) != len(list_relevant):
        return 0.0
    scores = [
        recall_at_k(ret, rel, k)
        for ret, rel in zip(list_retrieved, list_relevant)
    ]
    return sum(scores) / len(scores)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Mean Reciprocal Rank @k: 1 / (rank of first relevant in top-k), 0 if none.
    Rank is 1-based.
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def mrr_at_k_batch(
    list_retrieved: List[List[str]],
    list_relevant: List[Set[str]],
    k: int,
) -> float:
    """Mean MRR@k over a batch of (retrieved, relevant) pairs."""
    if not list_retrieved or len(list_retrieved) != len(list_relevant):
        return 0.0
    scores = [
        mrr_at_k(ret, rel, k)
        for ret, rel in zip(list_retrieved, list_relevant)
    ]
    return sum(scores) / len(scores)


def dcg_at_k(relevances: List[int], k: int) -> float:
    """DCG@k with binary relevance: sum(rel_i / log2(rank_i + 1)), rank 1-based."""
    total = 0.0
    for i in range(min(k, len(relevances))):
        total += relevances[i] / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k. Binary relevance.
    Returns 0.0 if relevant_ids is empty or IDCG is 0.
    """
    if not relevant_ids:
        return 0.0
    rel_vec = _relevance_vector(retrieved_ids, relevant_ids)
    dcg = dcg_at_k(rel_vec, k)
    # IDCG: ideal ordering = all relevances 1 first, then 0
    num_relevant = min(k, len(relevant_ids))
    ideal = [1] * num_relevant + [0] * max(0, k - num_relevant)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def ndcg_at_k_batch(
    list_retrieved: List[List[str]],
    list_relevant: List[Set[str]],
    k: int,
) -> float:
    """Mean nDCG@k over a batch."""
    if not list_retrieved or len(list_retrieved) != len(list_relevant):
        return 0.0
    scores = [
        ndcg_at_k(ret, rel, k)
        for ret, rel in zip(list_retrieved, list_relevant)
    ]
    return sum(scores) / len(scores)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Hit Rate@k: 1 if at least one relevant in top-k, else 0."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in relevant_ids for cid in top_k) else 0.0


def hit_rate_at_k_batch(
    list_retrieved: List[List[str]],
    list_relevant: List[Set[str]],
    k: int,
) -> float:
    """Proportion of queries with at least one relevant in top-k."""
    if not list_retrieved or len(list_retrieved) != len(list_relevant):
        return 0.0
    scores = [
        hit_rate_at_k(ret, rel, k)
        for ret, rel in zip(list_retrieved, list_relevant)
    ]
    return sum(scores) / len(scores)


def compute_all_metrics(
    list_retrieved: List[List[str]],
    list_relevant: List[Set[str]],
    k_values: List[int] = (5, 10),
) -> dict:
    """
    Compute Recall@k, MRR@k, nDCG@k, Hit Rate@k for each k in k_values.
    Returns a dict with keys like "recall_at_5", "mrr_at_10", "ndcg_at_5", "hit_rate_at_10".
    """
    out = {}
    for k in k_values:
        out[f"recall_at_{k}"] = recall_at_k_batch(list_retrieved, list_relevant, k)
        out[f"mrr_at_{k}"] = mrr_at_k_batch(list_retrieved, list_relevant, k)
        out[f"ndcg_at_{k}"] = ndcg_at_k_batch(list_retrieved, list_relevant, k)
        out[f"hit_rate_at_{k}"] = hit_rate_at_k_batch(list_retrieved, list_relevant, k)
    return out
