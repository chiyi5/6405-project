"""L2 — End-to-end learning-plan evaluation on real JD/resume gap profiles + ablations.

    python experiments/learning/L2_plan_evaluation.py [--neural]

Gap profiles come from real JDs (requirement labels give priority) and real resumes
(``data/learning/benchmark/profiles.json``). A plan of k courses is scored with silver
gap->course relevance from Coursera skill tags (never shown to the planner).

Outputs (results/learning/bench/):
    L2_strategies.csv     concat vs per-gap vs xQuAD, for several retrievers, k in {3,5,10}
    L2_ablation.csv       xQuAD λ sweep, priority weights on/off, alias expansion on/off
    L2_significance.csv   paired randomisation tests (xQuAD vs per-gap vs concat)
    L2_per_profile.csv    per-profile scores (error analysis)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning import retrievers as R  # noqa: E402
from src.learning.benchmark import load_benchmark, relevant_ids  # noqa: E402
from src.learning.engine import ALIASES, canonical  # noqa: E402
from src.learning.ir_metrics import bootstrap_ci, holm, paired_randomization  # noqa: E402
from src.learning.planner import plan, plan_metrics  # noqa: E402

OUT = ROOT / "results" / "learning" / "bench"


def alias_fn(skill):
    return skill + " " + " ".join(ALIASES.get(canonical(skill), []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    corpus, _, _, profiles = load_benchmark()
    texts, ids = list(corpus.text), list(corpus.id)
    gap_rel = {g["skill"]: relevant_ids(g["skill"], corpus) for p in profiles for g in p["gaps"]}

    rets = {"BM25": R.BM25Retriever().fit(texts), "TF-IDF": R.TfidfRetriever().fit(texts),
            "LSA": R.LSARetriever(300).fit(texts)}
    try:
        rets["Word2Vec"] = R.EmbeddingRetriever("word2vec").fit(texts)
    except ImportError:
        rets["GloVe"] = R.EmbeddingRetriever("glove-wiki-gigaword-100").fit(texts)
        print("[note] gensim missing: Word2Vec replaced by GloVe in the fusion retriever")
    emb = rets.get("Word2Vec", rets.get("GloVe"))
    rets["Linear fusion"] = R.LinearFusion([rets["BM25"], rets["LSA"], emb], [1.0, 0.5, 0.5])
    if args.neural:
        rets["Dense"] = R.DenseRetriever(args.dense_model).fit(texts)

    rows = []
    for rname, ret in rets.items():
        for strategy in ("concat", "pergap", "xquad"):
            for k in (3, 5, 10):
                for p in profiles:
                    chosen = [ids[i] for i in plan(p["gaps"], ret, k=k, strategy=strategy)]
                    rows.append({"retriever": rname, "strategy": strategy, "k": k, "pid": p["pid"],
                                 "n_gaps": len(p["gaps"]), **plan_metrics(chosen, p["gaps"], gap_rel, k)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "L2_per_profile.csv", index=False)
    summ = []
    for (r, s, k), g in df.groupby(["retriever", "strategy", "k"]):
        lo, hi = bootstrap_ci(g.weighted_coverage)
        summ.append({"retriever": r, "strategy": s, "k": k, "profiles": len(g),
                     "weighted_coverage": g.weighted_coverage.mean(), "cov_lo": lo, "cov_hi": hi,
                     "gap_recall": g.gap_recall.mean(), "req_covered": g.mandatory_required_covered.mean(),
                     "precision": g.precision.mean(), "redundancy": g.redundancy.mean()})
    pd.DataFrame(summ).round(4).to_csv(OUT / "L2_strategies.csv", index=False)

    # significance at k=5, per retriever
    sig = []
    for r in rets:
        t = df[(df.retriever == r) & (df.k == 5)].pivot_table(index="pid", columns="strategy", values="weighted_coverage")
        pv = {f"xquad vs {b}": paired_randomization(t["xquad"], t[b]) for b in ("pergap", "concat")}
        pv["pergap vs concat"] = paired_randomization(t["pergap"], t["concat"])
        adj = holm(pv)
        for name, p in pv.items():
            a, b = name.split(" vs ")
            sig.append({"retriever": r, "comparison": name, "delta_coverage@5": t[a].mean() - t[b].mean(),
                        "p_raw": p, "p_holm": adj[name]})
    pd.DataFrame(sig).round(4).to_csv(OUT / "L2_significance.csv", index=False)

    # ablations on the fusion retriever, k=5
    ab = []
    ret = rets["Linear fusion"]
    configs = [("xQuAD λ=%.1f" % lam, dict(strategy="xquad", lam=lam)) for lam in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)]
    configs += [("xQuAD λ=0.7, original (no normalisation, aggregated pool)", dict(strategy="xquad", lam=0.7, normalise_div=False, per_gap_pool=False)),
                ("xQuAD λ=0.7, + per-gap pool only", dict(strategy="xquad", lam=0.7, normalise_div=False)),
                ("xQuAD λ=0.7, + normalisation only", dict(strategy="xquad", lam=0.7, per_gap_pool=False)),
                ("xQuAD λ=0.7, no priority weights", dict(strategy="xquad", lam=0.7, use_priority=False)),
                ("xQuAD λ=0.7, + alias expansion", dict(strategy="xquad", lam=0.7, alias_fn=alias_fn)),
                ("per-gap, no priority weights", dict(strategy="pergap", use_priority=False)),
                ("per-gap, + alias expansion", dict(strategy="pergap", alias_fn=alias_fn))]
    for label, kw in configs:
        vals = [plan_metrics([ids[i] for i in plan(p["gaps"], ret, k=5, **kw)], p["gaps"], gap_rel, 5) for p in profiles]
        v = pd.DataFrame(vals)
        ab.append({"config": label, **v.mean().to_dict()})
    pd.DataFrame(ab).round(4).to_csv(OUT / "L2_ablation.csv", index=False)
    print(pd.DataFrame(summ)[lambda d: d.k == 5].round(3).to_string(index=False))
    print(pd.DataFrame(ab).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
