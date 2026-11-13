"""L5 — Human relevance judgements: make the annotation sheet, then score it.

Silver labels (Coursera tags) are noisy: L3 shows ~1/3 of top-10 "non-relevant" hits
mention the skill verbatim. This script builds a pooled sheet for two team members and,
once filled, reports inter-annotator agreement and re-scores the systems on human labels.

    python experiments/learning/L5_human_judgements.py make     # -> data/learning/human/sheet_A.csv, sheet_B.csv
    # each annotator fills `grade` (0 not useful / 1 marginal / 2 useful / 3 directly teaches the skill)
    python experiments/learning/L5_human_judgements.py score    # -> results/learning/bench/L5_*.csv

Pooling: top-5 of four systems of different families for every dev+test query (the
standard TREC depth-k pool). Rows are shuffled and system names hidden (blind judging).
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning.benchmark import load_benchmark  # noqa: E402
from src.learning.ir_metrics import METRICS  # noqa: E402

H = ROOT / "data" / "learning" / "human"
B = ROOT / "results" / "learning" / "bench"
POOL_SYSTEMS = ["BM25 (dev-tuned)", "LSA (dev-tuned)", "Word2Vec (trained on corpus)", "LTR logistic regression (train/dev)"]


def make(depth=5, seed=0):
    corpus, queries, qrels, _ = load_benchmark()
    runs = pickle.load(open(B / "L1_runs.pkl", "rb"))["runs"]
    info = corpus.set_index("id")
    rows = []
    for _, q in queries[queries.split.isin(["dev", "test"])].iterrows():
        pool = []
        for s in POOL_SYSTEMS:
            pool += [d for d in runs[s][q.qid][:depth] if d not in pool]
        for d in pool:
            rows.append({"qid": q.qid, "skill_gap": q.skill, "query_text": q["query"], "doc_id": d,
                         "course_title": info.at[d, "title"], "description": info.at[d, "description"][:600],
                         "grade": "", "annotator": "", "notes": ""})
    random.Random(seed).shuffle(rows)
    H.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for who in ("A", "B"):
        df.to_csv(H / f"sheet_{who}.csv", index=False, encoding="utf-8-sig")
    (H / "README.md").write_text(
        "# Human relevance judgements\n\n"
        "Two people label `sheet_A.csv` and `sheet_B.csv` independently (do not discuss until both are done).\n\n"
        "grade: 0 = not useful for this gap; 1 = mentions it in passing; 2 = useful, covers part of it; "
        "3 = directly teaches this skill.\nJudge the skill_gap; query_text shows how the JD phrased it.\n"
        "Fill `annotator` with your initials. Then run `python experiments/learning/L5_human_judgements.py score`.\n",
        encoding="utf-8")
    print(f"{len(df)} rows for {df.qid.nunique()} queries -> {H}")


def score():
    from sklearn.metrics import cohen_kappa_score
    a = pd.read_csv(H / "sheet_A.csv").dropna(subset=["grade"])
    b = pd.read_csv(H / "sheet_B.csv").dropna(subset=["grade"])
    m = a.merge(b, on=["qid", "doc_id"], suffixes=("_a", "_b"))
    ga, gb = m.grade_a.astype(int), m.grade_b.astype(int)
    agree = {"pairs": len(m), "kappa_4level": cohen_kappa_score(ga, gb),
             "kappa_weighted_quadratic": cohen_kappa_score(ga, gb, weights="quadratic"),
             "kappa_binary(>=2)": cohen_kappa_score(ga >= 2, gb >= 2), "exact_agreement": (ga == gb).mean()}
    # consensus = mean grade, rounded down (conservative); silver vs human agreement
    m["grade"] = ((ga + gb) / 2).astype(int)
    _, queries, silver, _ = load_benchmark()
    m["silver"] = [int(d in silver.get(q, {})) for q, d in zip(m.qid, m.doc_id)]
    agree["kappa_silver_vs_human(>=2)"] = cohen_kappa_score(m.silver, m.grade >= 2)
    agree["silver_precision"] = (m[m.silver == 1].grade >= 2).mean()
    agree["silver_false_negative_rate"] = (m[m.silver == 0].grade >= 2).mean()
    pd.DataFrame([agree]).round(4).to_csv(B / "L5_agreement.csv", index=False)
    m[["qid", "doc_id", "grade"]].to_csv(H / "qrels_human_consensus.csv", index=False)

    human = {}
    for q, d, g in zip(m.qid, m.doc_id, m.grade):
        if g > 0:
            human.setdefault(q, {})[d] = int(g)
    runs = pickle.load(open(B / "L1_runs.pkl", "rb"))["runs"]
    rows = []
    for s, run in runs.items():
        # judged-only evaluation (Sakai, 2007): unjudged docs are removed, not counted as non-relevant
        judged = {q: set(m[m.qid == q].doc_id) for q in m.qid.unique()}
        vals = {k: [] for k in ("ndcg@10", "p@5", "mrr@10")}
        for q, rel in human.items():
            ranked = [d for d in run.get(q, []) if d in judged[q]]
            for k in vals:
                vals[k].append(METRICS[k](ranked, rel))
        rows.append({"system": s, "queries": len(human), **{k: sum(v) / len(v) for k, v in vals.items() if v}})
    pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).round(4).to_csv(B / "L5_human_eval.csv", index=False)
    print(pd.DataFrame([agree]).round(3).T)


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
