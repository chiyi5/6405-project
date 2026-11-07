"""
Member E — Score calibration, learned weighting and ablation study.

E1  hand-crafted score (original notebook: fixed thresholds + fixed priority weights)
E2  grid-searched thresholds / priority weights (tuned on a dev split, reported on test)
E3  learned scorer: requirement-level match statistics -> logistic / ridge model (cross-validated)
E4  ablation: switch off one component at a time and measure the drop
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from .pipeline import MatchingPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Run the pipeline once per pair and keep the requirement-level details
# ---------------------------------------------------------------------------
def run_pipeline_on_pairs(pipe: MatchingPipeline, pairs: pd.DataFrame, verbose: bool = True) -> list[dict]:
    jd_cache, results = {}, []
    for i, r in pairs.reset_index(drop=True).iterrows():
        if r.jd_id not in jd_cache:
            jd_cache[r.jd_id] = pipe.extract_jd_requirements(r.jd_text)
        res = pipe.match(r.jd_text, r.resume_text, jd_df=jd_cache[r.jd_id])
        res.update({"jd_id": r.jd_id, "resume_id": r.resume_id, "label": r.get("label")})
        results.append(res)
        if verbose and (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(pairs)} pairs")
    return results


def pair_features(res: dict) -> dict:
    """Interpretable features from one pipeline result (used by the learned scorer)."""
    d = res["details"].copy()
    f = {}
    for pr in ("REQUIRED", "UNSPECIFIED", "PREFERRED"):
        sub = d[(d.category == "Skill") & (d.priority.isin([pr] + (["MANDATORY"] if pr == "REQUIRED" else [])))]
        n = len(sub)
        f[f"{pr.lower()}_n"] = n
        f[f"{pr.lower()}_mean_score"] = float(sub.match_score.fillna(0).mean()) if n else 0.0
        f[f"{pr.lower()}_exact_frac"] = float((sub.match_type == "EXACT").mean()) if n else 0.0
        f[f"{pr.lower()}_missing_frac"] = float((sub.match_score.fillna(0) < 40).mean()) if n else 0.0
        f[f"{pr.lower()}_mean_sim"] = float(sub.semantic_similarity.fillna(0).mean()) if n else 0.0
    edu = d[d.category == "Education"].match_score
    exp = d[d.category == "Experience"].match_score
    f["education_score"] = float(edu.max()) if edu.notna().any() else 50.0     # 50 = unknown
    f["education_known"] = int(edu.notna().any())
    f["experience_score"] = float(exp.max()) if exp.notna().any() else 50.0
    f["experience_known"] = int(exp.notna().any())
    f["resume_years"] = res["profile"]["experience_summary"].get("years") or 0.0
    f["n_resume_skills"] = len(res["profile"]["skills"])
    f["handcrafted_score"] = res["final_score"] if res["final_score"] is not None else np.nan
    return f


def build_feature_table(results: list[dict]) -> pd.DataFrame:
    rows = [{"jd_id": r["jd_id"], "resume_id": r["resume_id"], "label": r["label"], **pair_features(r)}
            for r in results]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E3 learned scorer
# ---------------------------------------------------------------------------
FEATURE_COLS = None  # set lazily = all numeric except ids/label/handcrafted


def evaluate_learned_scorer(feat: pd.DataFrame, seed: int = 42, n_splits: int = 5):
    """
    Compare hand-crafted score vs. learned models with GroupKFold over JDs
    (a JD never appears in both train and test). Returns (summary, coefficients).
    """
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat = feat.dropna(subset=["label"]).reset_index(drop=True)
    cols = [c for c in feat.columns if c not in ("jd_id", "resume_id", "label", "handcrafted_score")]
    X, y = feat[cols].fillna(0).values, feat["label"].astype(int).values
    groups = feat["jd_id"].values
    if len(set(groups)) >= n_splits:
        splits = list(GroupKFold(n_splits).split(X, y, groups))
    else:
        splits = list(StratifiedKFold(min(n_splits, np.bincount(y).min()), shuffle=True,
                                      random_state=seed).split(X, y))
    models = {
        "E3a_logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "E3b_gboost": lambda: GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=2),
    }
    out = []
    # E1: hand-crafted score -> label via CV logistic on the single score
    hc = feat["handcrafted_score"].fillna(feat["handcrafted_score"].median()).values
    pred_hc, exp_hc = np.zeros_like(y), np.zeros(len(y))
    for tr, te in splits:
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(hc[tr].reshape(-1, 1), y[tr])
        pred_hc[te] = m.predict(hc[te].reshape(-1, 1))
    out.append({"method": "E1_handcrafted", "spearman": spearmanr(hc, y).statistic,
                "macro_f1": f1_score(y, pred_hc, average="macro")})
    coef_table = None
    for name, fac in models.items():
        pred, expv = np.zeros_like(y), np.zeros(len(y))
        for tr, te in splits:
            m = fac().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
            proba = m.predict_proba(X[te])
            expv[te] = proba @ m.classes_ if hasattr(m, "classes_") else proba @ m[-1].classes_
        out.append({"method": name, "spearman": spearmanr(expv, y).statistic,
                    "macro_f1": f1_score(y, pred, average="macro")})
        if name == "E3a_logreg":
            full = fac().fit(X, y)
            lr = full[-1]
            coef_table = pd.DataFrame(lr.coef_.T, index=cols,
                                      columns=[f"class_{c}" for c in lr.classes_]).round(3)
    return pd.DataFrame(out).round(3), coef_table


# ---------------------------------------------------------------------------
# E2 grid search of thresholds / weights (re-scoring only, no re-extraction)
# ---------------------------------------------------------------------------
def rescore(details: pd.DataFrame, thresholds, weights, bonus_max=10.0) -> float:
    d = details.copy()
    d["match_score"] = d["match_score"].astype(float)
    sim = d.semantic_similarity
    is_sem = d.category.eq("Skill") & d.match_type.ne("EXACT") & sim.notna()

    def m(s):
        for th, sc in thresholds:
            if s >= th:
                return sc
        return 0.0
    d.loc[is_sem, "match_score"] = sim[is_sem].map(m)
    d = d[d.match_score.notna()]
    base = d[d.priority.isin(["MANDATORY", "REQUIRED", "UNSPECIFIED"])]
    pref = d[d.priority == "PREFERRED"]
    if base.empty:
        return np.nan
    w = base.priority.map(weights)
    b = (base.match_score * w).sum() / w.sum()
    bonus = bonus_max * pref.match_score.mean() / 100 if len(pref) else 0.0
    return min(100.0, b + bonus)


def grid_search_scoring(results: list[dict], dev_mask: np.ndarray):
    """Tune thresholds/weights on dev pairs (maximise Spearman); report on the held-out pairs."""
    from scipy.stats import spearmanr
    y = np.array([r["label"] for r in results], dtype=float)
    top_thr = [0.85, 0.80, 0.75, 0.70]
    gap = [0.10, 0.15]
    w_req = [0.7, 0.9, 1.0]
    w_uns = [0.2, 0.5, 0.8]
    best, log = None, []
    for t0, g, wr, wu in itertools.product(top_thr, gap, w_req, w_uns):
        th = [(t0, 85.0), (t0 - g, 70.0), (t0 - 2 * g, 50.0), (t0 - 3 * g, 25.0)]
        w = {"MANDATORY": 1.0, "REQUIRED": wr, "UNSPECIFIED": wu, "PREFERRED": 0.3}
        s = np.array([rescore(r["details"], th, w) for r in results])
        ok = dev_mask & ~np.isnan(s)
        rho = spearmanr(s[ok], y[ok]).statistic if ok.sum() > 2 else np.nan
        log.append({"top_threshold": t0, "gap": g, "w_required": wr, "w_unspecified": wu, "dev_spearman": rho})
        if best is None or (rho is not None and rho > best[0]):
            best = (rho, th, w, s)
    rho_dev, th, w, s = best
    test = ~dev_mask & ~np.isnan(s)
    default = np.array([rescore(r["details"], PipelineConfig().sim_thresholds, PipelineConfig().priority_weights)
                        for r in results])
    summary = pd.DataFrame([
        {"setting": "E1_default", "test_spearman": spearmanr(default[test], y[test]).statistic},
        {"setting": "E2_tuned", "test_spearman": spearmanr(s[test], y[test]).statistic,
         "thresholds": th, "weights": w},
    ])
    return summary, pd.DataFrame(log).sort_values("dev_spearman", ascending=False)


# ---------------------------------------------------------------------------
# E4 ablation
# ---------------------------------------------------------------------------
ABLATIONS = {
    "full_system": {},
    "- exact match": {"use_exact_match": False},
    "- semantic match": {"use_semantic_match": False},
    "- priority weights": {"use_priority_weights": False},
    "- education": {"use_education": False},
    "- experience": {"use_experience": False},
    "- section headers": {"use_section_headers": False},
}


def run_ablation(pairs: pd.DataFrame, base_cfg: PipelineConfig, embedder=None, skill_extractor=None,
                 ablations: dict = ABLATIONS) -> pd.DataFrame:
    from scipy.stats import spearmanr
    from .matching import cv_macro_f1
    y = pairs["label"].astype(int).values
    rows = []
    for name, over in ablations.items():
        cfg = replace(base_cfg, **over)
        pipe = MatchingPipeline(cfg, embedder=embedder, skill_extractor=skill_extractor)
        res = run_pipeline_on_pairs(pipe, pairs, verbose=False)
        s = np.array([r["final_score"] if r["final_score"] is not None else np.nan for r in res], dtype=float)
        ok = ~np.isnan(s)
        rows.append({"variant": name, "spearman": spearmanr(s[ok], y[ok]).statistic,
                     "macro_f1_cv": cv_macro_f1(s[ok], y[ok]), "n_scored": int(ok.sum())})
    df = pd.DataFrame(rows)
    df["delta_spearman"] = df.spearman - df.spearman.iloc[0]
    return df.round(3)
