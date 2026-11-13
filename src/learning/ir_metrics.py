"""Ranking metrics and significance tests for the course-retrieval benchmark.

All metrics take ``ranked`` (list of doc ids, best first) and ``rels`` ({doc_id: grade>0}).
Unjudged documents count as non-relevant (silver labels are exhaustive over the corpus:
every resource carries Coursera skill tags).
"""
from __future__ import annotations

import math

import numpy as np


def dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(ranked, rels, k=10):
    ideal = dcg(sorted(rels.values(), reverse=True)[:k])
    return dcg([rels.get(d, 0) for d in ranked[:k]]) / ideal if ideal else float("nan")


def recall_at(ranked, rels, k=10):
    return len(set(ranked[:k]) & set(rels)) / len(rels) if rels else float("nan")


def precision_at(ranked, rels, k=5):
    return len(set(ranked[:k]) & set(rels)) / k


def mrr_at(ranked, rels, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1 / (i + 1)
    return 0.0


def ap_at(ranked, rels, k=100):
    """Average precision truncated at k, normalised by min(|rel|, k)."""
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(rels), k) if rels else float("nan")


METRICS = {"ndcg@10": lambda r, g: ndcg_at(r, g, 10), "map@100": lambda r, g: ap_at(r, g, 100),
           "mrr@10": lambda r, g: mrr_at(r, g, 10), "p@5": lambda r, g: precision_at(r, g, 5),
           "recall@10": lambda r, g: recall_at(r, g, 10), "recall@100": lambda r, g: recall_at(r, g, 100)}


def evaluate_run(run: dict, qrels: dict, qids=None) -> "pd.DataFrame":
    """Per-query metric table for one system. ``run``: qid -> ranked doc ids."""
    import pandas as pd
    rows = []
    for qid in (qids if qids is not None else run):
        rels = qrels.get(qid, {})
        if not rels:
            continue
        rows.append({"qid": qid, **{m: f(run[qid], rels) for m, f in METRICS.items()}})
    return pd.DataFrame(rows)


def bootstrap_ci(values, n=2000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_randomization(a, b, n=10000, seed=0):
    """Two-sided paired sign-flip randomisation test on per-query differences (Smucker et al., 2007)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], (n, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def holm(pvals: dict) -> dict:
    """Holm–Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
