"""
Member A (data / evaluation) and Member D (matching results).
Human-annotated benchmark: Vanetik & Kogan (2023), "Job Vacancy Ranking with Sentence Embeddings,
Keywords, and Named Entities", Information 14(8):468. https://github.com/NataliaVanetik/vacancy-resume-matching-dataset

30 CVs x 5 vacancies. Two independent annotators rank, for each CV, the 5 vacancies from best to worst
(following instructions from a senior HR specialist).

Task for a matcher: for each CV, rank the 5 vacancies. Metrics per CV, averaged:
  * Spearman rho between the system ranking and an annotator ranking (also the mean of the two annotators)
  * top-1 agreement: the system's best vacancy = the annotator's best vacancy
  * NDCG@5 with graded relevance = 5 - human rank
Human agreement (annotator 1 vs annotator 2, same metrics) is the ceiling a system can be expected to reach.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .pipeline import clean_text, read_document


def _parse_rankings(text: str, name: str) -> list[list[int]]:
    block = re.search(rf"{name}\s*=\s*(\[.*?\]\])", text, re.S).group(1)
    block = re.sub(r"#[^\n]*", "", block)
    return ast.literal_eval(block)


def load_benchmark(root: str):
    """Returns vacancies (list of 5 texts in annotation order), cvs {cv_id: text}, rankings DataFrame."""
    root = Path(root)
    vac = pd.read_csv(root / "5_vacancies.csv")
    vacancies = [clean_text(f"{r.job_title}\n{r.job_description}") for r in vac.itertuples()]
    raw = (root / "annotations-for-the-first-30-vacancies.txt").read_text(encoding="utf-8")
    rows = []
    for ann in ("ANNOTATOR_1_RANKINGS", "ANNOTATOR_2_RANKINGS"):
        for cv, order in enumerate(_parse_rankings(raw, ann), start=1):
            valid = sorted(order) == [1, 2, 3, 4, 5]
            # order[k] = vacancy placed at rank k+1  ->  rank of each vacancy
            rank = {v: k + 1 for k, v in enumerate(order)} if valid else {}
            rows.append({"annotator": ann[10], "cv": cv, "order": order, "valid": valid,
                         **{f"rank_v{v}": rank.get(v) for v in range(1, 6)}})
    rankings = pd.DataFrame(rows)
    cvs = {i: clean_text(read_document(root / "CV" / f"{i}.docx")) for i in range(1, 31)}
    return vacancies, cvs, rankings


def human_rank_matrix(rankings: pd.DataFrame, annotator: str) -> pd.DataFrame:
    r = rankings[rankings.annotator == annotator].set_index("cv")
    return r[[f"rank_v{v}" for v in range(1, 6)]].astype(float)  # NaN rows = invalid ranking


def consensus_ranks(rankings: pd.DataFrame) -> pd.DataFrame:
    """Mean of the two annotators' ranks (only where both are valid), re-ranked 1..5 (ties averaged)."""
    a, b = human_rank_matrix(rankings, "1"), human_rank_matrix(rankings, "2")
    m = (a + b) / 2
    return m.rank(axis=1, method="average")


def _metrics(sys_rank: pd.Series, human_rank: pd.Series) -> dict:
    if human_rank.isna().any() or sys_rank.isna().any():
        return {}
    rho = spearmanr(sys_rank.values, human_rank.values).statistic
    top1 = float(sys_rank.values.argmin() == human_rank.values.argmin())
    rel = 5 - human_rank.values                       # best = 4 ... worst = 0
    order = np.argsort(sys_rank.values)
    dcg = ((2 ** rel[order] - 1) / np.log2(np.arange(2, 7))).sum()
    idcg = ((2 ** np.sort(rel)[::-1] - 1) / np.log2(np.arange(2, 7))).sum()
    return {"spearman": rho, "top1": top1, "ndcg@5": dcg / idcg}


def evaluate_rank_matrix(sys_ranks: pd.DataFrame, rankings: pd.DataFrame, name: str) -> dict:
    """sys_ranks: index cv, columns rank_v1..rank_v5 (1 = best)."""
    out = {"method": name}
    targets = {"ann1": human_rank_matrix(rankings, "1"), "ann2": human_rank_matrix(rankings, "2"),
               "consensus": consensus_ranks(rankings)}
    for tname, H in targets.items():
        ms = [_metrics(sys_ranks.loc[cv], H.loc[cv]) for cv in H.index]
        ms = [m for m in ms if m]
        for k in ("spearman", "top1", "ndcg@5"):
            out[f"{k}_{tname}"] = round(float(np.mean([m[k] for m in ms])), 3) if ms else np.nan
        out[f"n_{tname}"] = len(ms)
    return out


def human_agreement(rankings: pd.DataFrame) -> dict:
    """Annotator 1 vs annotator 2 on CVs where both rankings are valid."""
    a, b = human_rank_matrix(rankings, "1"), human_rank_matrix(rankings, "2")
    ms = [_metrics(a.loc[cv], b.loc[cv]) for cv in a.index]
    ms = [m for m in ms if m]
    rhos = [m["spearman"] for m in ms]
    return {"method": "Human: annotator 1 vs 2", "spearman_mean": round(float(np.mean(rhos)), 3),
            "spearman_median": round(float(np.median(rhos)), 3),
            "top1_agreement": round(float(np.mean([m["top1"] for m in ms])), 3),
            "n_cvs": len(ms),
            "cvs_with_negative_rho": int(sum(r < 0 for r in rhos))}


def scores_to_ranks(scores: np.ndarray) -> pd.Series:
    """Higher score = better = rank 1."""
    s = pd.Series(-np.asarray(scores, dtype=float))
    return s.rank(method="average").set_axis([f"rank_v{v}" for v in range(1, 6)])


def run_matchers(matchers: list, vacancies: list[str], cvs: dict) -> dict[str, pd.DataFrame]:
    """Each matcher must expose score_pairs(jds, resumes). Returns {name: rank matrix}."""
    out = {}
    for m in matchers:
        if hasattr(m, "fit") and not getattr(m, "_fitted", True):
            m.fit(vacancies + list(cvs.values()))
        if isinstance(m, type) or not hasattr(m, "score_pairs"):
            continue
        rows = {}
        for cv, text in cvs.items():
            s = m.score_pairs(vacancies, [text] * 5)
            s = np.nan_to_num(np.asarray(s, dtype=float), nan=-1e9)
            rows[cv] = scores_to_ranks(s)
        out[m.name] = pd.DataFrame(rows).T
    return out


def random_baseline(n_draws: int = 2000, seed: int = 0) -> dict:
    """Expected metrics of a random ranking (for reference)."""
    rng = np.random.default_rng(seed)
    rhos, top1 = [], []
    for _ in range(n_draws):
        a, b = rng.permutation(5) + 1, rng.permutation(5) + 1
        rhos.append(spearmanr(a, b).statistic)
        top1.append(float(a.argmin() == b.argmin()))
    return {"method": "Random ranking (expected)", "spearman": round(float(np.mean(rhos)), 3),
            "top1": round(float(np.mean(top1)), 3)}
