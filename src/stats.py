"""Bootstrap confidence intervals for the main comparisons (Member A — evaluation methodology)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cluster_bootstrap_spearman(df: pd.DataFrame, score_col: str, label_col: str = "label", cluster_col: str = "jd_id",
                               n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Spearman rho with a 95% CI, resampling whole JDs (pairs of one JD are not independent)."""
    d = df.dropna(subset=[score_col])
    groups = {k: g for k, g in d.groupby(cluster_col)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    est = spearmanr(d[score_col], d[label_col]).statistic
    boots = []
    for _ in range(n):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        boots.append(spearmanr(s[score_col], s[label_col]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def per_query_ap(run: dict, qrels: dict) -> pd.Series:
    from .talentclef import _ap
    return pd.Series({q: _ap([c for c, _ in sorted(run[q].items(), key=lambda x: -x[1])], rel)
                      for q, rel in qrels.items()})


def paired_bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 5000, seed: int = 0) -> dict:
    """Mean difference a-b over queries with a 95% CI and the share of resamples where a <= b."""
    d = (a - b).dropna().values
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": round(float(d.mean()), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "p_not_better": round(float((boots <= 0).mean()), 4), "n_queries": len(d)}
