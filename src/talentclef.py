"""
Member A (data / evaluation) and Member D (matching results).
TalentCLEF 2026 Task A — Contextualized Job-Person Matching (English development set).
Gasco et al. (2026), Overview of TalentCLEF 2026, CLEF. Data: https://doi.org/10.5281/zenodo.19652670 (CC BY 4.0)

10 job descriptions (queries) x 472 resumes (corpus); expert-annotated binary relevance in qrels.tsv
(only relevant pairs are listed; every other pair counts as not relevant). The test-set qrels are not public.
This is the direction our system is built for: rank candidates for one job.

Metrics (TREC style, per query then averaged): MAP, NDCG@10, P@10, R-precision.
Official reference (test set, English): TalentCLEF baseline MAP 0.4001, best team MAP 0.7122.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_taskA(root: str, split: str = "development", lang: str = "en"):
    d = Path(root) / "TaskA" / split / lang
    queries = {p.name: p.read_text(encoding="utf-8") for p in sorted((d / "queries").iterdir())}
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted((d / "corpus").iterdir())}
    qrels = None
    if (d / "qrels.tsv").exists():
        q = pd.read_csv(d / "qrels.tsv", sep="\t", header=None, names=["q", "iter", "c", "rel"], dtype=str)
        q["rel"] = q["rel"].astype(int)
        qrels = {qid: set(g[g.rel > 0].c) for qid, g in q.groupby("q")}
    return queries, corpus, qrels


def score_all(matcher, queries: dict, corpus: dict, verbose: bool = True) -> dict[str, dict[str, float]]:
    """Returns run[qid][cid] = score (higher = more suitable)."""
    cids, ctexts = list(corpus), list(corpus.values())
    if hasattr(matcher, "fit") and not getattr(matcher, "_fitted", True):
        matcher.fit(ctexts)
    run = {}
    for i, (qid, qtext) in enumerate(queries.items()):
        s = np.asarray(matcher.score_pairs([qtext] * len(ctexts), ctexts), dtype=float)
        s = np.nan_to_num(s, nan=-1e9)
        run[qid] = dict(zip(cids, s))
        if verbose:
            print(f"    {getattr(matcher, 'name', matcher)}: query {i + 1}/{len(queries)}", flush=True)
    return run


def _ap(ranked: list[str], rel: set[str]) -> float:
    hits, total = 0, 0.0
    for k, c in enumerate(ranked, 1):
        if c in rel:
            hits += 1
            total += hits / k
    return total / len(rel) if rel else 0.0


def _ndcg(ranked: list[str], rel: set[str], k: int = 10) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg else 0.0


def evaluate_run(run: dict, qrels: dict, name: str = "") -> dict:
    per = []
    for qid, rel in qrels.items():
        ranked = [c for c, _ in sorted(run[qid].items(), key=lambda x: -x[1])]
        R = len(rel)
        per.append({"map": _ap(ranked, rel), "ndcg@10": _ndcg(ranked, rel, 10),
                    "p@10": sum(c in rel for c in ranked[:10]) / 10,
                    "r-prec": sum(c in rel for c in ranked[:R]) / R})
    df = pd.DataFrame(per)
    return {"method": name, **{k: round(float(df[k].mean()), 4) for k in df.columns}, "queries": len(per)}


def random_baseline(qrels: dict, corpus: dict, n: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cids = list(corpus)
    runs = []
    for _ in range(n):
        run = {q: dict(zip(cids, rng.random(len(cids)))) for q in qrels}
        runs.append(evaluate_run(run, qrels))
    df = pd.DataFrame(runs)
    return {"method": "Random ranking (expected)", **{k: round(float(df[k].mean()), 4)
                                                      for k in ("map", "ndcg@10", "p@10", "r-prec")},
            "queries": len(qrels)}


def write_trec_run(run: dict, path: str, tag: str = "group32", depth: int = 100):
    """TREC run file for the official evaluation script / Codabench submission."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, scores in run.items():
            for rank, (cid, s) in enumerate(sorted(scores.items(), key=lambda x: -x[1])[:depth], 1):
                f.write(f"{qid} Q0 {cid} {rank} {s:.6f} {tag}\n")
