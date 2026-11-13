"""Turn a set of skill gaps into a short, non-redundant learning plan.

Three strategies are compared in ``experiments/learning/L2_plan_evaluation.py``:

* ``concat``  — naive baseline: all gap names in one query, take the top-k courses.
* ``pergap``  — one query per gap, scores summed with priority weights (the existing
                SkillBridge design), take the top-k.
* ``xquad``   — explicit aspect diversification (Santos et al., 2010, xQuAD): greedily pick
                the course maximising
                ``(1-λ)·P(d|q) + λ·Σ_g w_g·P(d|g)·Π_{s∈S}(1-P(s|g)) / Σ_g w_g·Π_{s∈S}(1-P(s|g))``
                so a course that covers a still-uncovered, high-priority gap wins over a
                second course for an already-covered gap. Two changes to the original:
                the diversity term is normalised by the still-uncovered weight, and the
                candidate pool is the union of each gap's own top results (without them a
                PREFERRED Docker gap never reached the pool next to REQUIRED SQL courses).

P(d|g) is the retriever score of course d for gap g, min-max normalised per gap. The
planner never looks at Coursera's skill tags (those are the evaluation labels).
"""
from __future__ import annotations

import numpy as np

PRIORITY_WEIGHT = {"MANDATORY": 3.0, "REQUIRED": 2.0, "UNSPECIFIED": 1.0, "PREFERRED": 0.5}


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def plan(gaps, retriever, k=5, strategy="xquad", lam=0.7, use_priority=True, pool=100, alias_fn=None,
         normalise_div=True, per_gap_pool=True):
    """Return a list of document indices (length <= k).

    ``gaps``: list of {"skill", "priority"}; ``retriever``: fitted Retriever;
    ``alias_fn``: optional skill -> query-text expansion.
    """
    if not gaps:
        return []
    w = np.array([PRIORITY_WEIGHT.get(g.get("priority", "REQUIRED"), 1.0) if use_priority else 1.0 for g in gaps])
    w = w / w.sum()
    texts = [alias_fn(g["skill"]) if alias_fn else g["skill"] for g in gaps]
    if strategy == "concat":
        s = retriever.score([" ".join(texts)])[0]
        return list(np.argsort(-s)[:k])
    G = np.stack([_minmax(r) for r in retriever.score(texts)])  # (n_gaps, n_docs) in [0,1]
    agg = w @ G
    if strategy == "pergap":
        return list(np.argsort(-agg)[:k])
    if strategy != "xquad":
        raise ValueError(strategy)
    # candidate pool = union of each gap's own top results, so a low-priority or niche gap
    # (e.g. Docker next to five SQL courses) still has candidates to be covered by
    if per_gap_pool:
        per_gap = max(5, pool // len(gaps))
        cand = np.unique(np.concatenate([np.argsort(-G[i])[:per_gap] for i in range(len(gaps))]))
    else:  # original xQuAD: re-rank the top of the aggregated ranking
        cand = np.argsort(-agg)[:pool]
    p_q = _minmax(agg[cand])
    p_g = G[:, cand]
    not_covered = np.ones(len(gaps))
    chosen, avail = [], np.ones(len(cand), bool)
    for _ in range(min(k, len(cand))):
        need = w * not_covered
        # normalise by the remaining (uncovered) weight so the diversity term stays on the
        # same [0,1] scale as relevance even after the big gaps are covered
        div = need @ p_g / (max(need.sum(), 1e-9) if normalise_div else 1.0)
        obj = (1 - lam) * p_q + lam * div
        obj[~avail] = -np.inf
        j = int(np.argmax(obj))
        chosen.append(int(cand[j]))
        avail[j] = False
        not_covered *= 1 - p_g[:, j]
    return chosen


def plan_metrics(chosen_ids, gaps, gap_relevant, k):
    """Set-level quality of a plan using silver gap->course relevance.

    gap_relevant: skill -> set of relevant doc ids.
    weighted_coverage: priority-weighted share of gaps covered by >=1 chosen course.
    redundancy: share of chosen courses that cover no *new* gap (in rank order).
    precision: share of chosen courses relevant to at least one gap.
    """
    w = {g["skill"]: PRIORITY_WEIGHT.get(g.get("priority", "REQUIRED"), 1.0) for g in gaps}
    covered, wasted, useful = set(), 0, 0
    for d in chosen_ids[:k]:
        hits = {g for g in w if d in gap_relevant.get(g, ())}
        useful += bool(hits)
        if not hits - covered:
            wasted += 1
        covered |= hits
    n = max(len(chosen_ids[:k]), 1)
    must = [g for g in gaps if g.get("priority") in ("MANDATORY", "REQUIRED")]
    return {"weighted_coverage": sum(w[g] for g in covered) / sum(w.values()),
            "gap_recall": len(covered) / len(w), "precision": useful / n, "redundancy": wasted / n,
            "mandatory_required_covered": (sum(g["skill"] in covered for g in must) / len(must)) if must else float("nan")}
