"""L3 — Error analysis, silver-label audit and report figures for the learning module.

    python experiments/learning/L3_error_analysis.py      (after L1 and L2)

Outputs (results/learning/bench/):
    L3_error_buckets.csv      nDCG@10 by query length, number of relevant courses and query type
    L3_label_audit.csv        top-10 "false positives" of the best system whose title/description
                              literally mention the skill -> lower bound on silver-label false negatives
    L3_worst_queries.csv      the 10 hardest test queries with the top-3 retrieved titles
    fig_L1_retrieval.png      nDCG@10 with 95% CI per system (test split)
    fig_L2_strategies.png     weighted gap coverage@5: concat vs per-gap vs xQuAD
    fig_L2_lambda.png         xQuAD λ sensitivity
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning.benchmark import load_benchmark  # noqa: E402
from src.plots import GRID, INK, MUTED, SERIES, _style  # noqa: E402

B = ROOT / "results" / "learning" / "bench"


def main():
    import pickle
    corpus, queries, qrels, _ = load_benchmark()
    title = dict(zip(corpus.id, corpus.title))
    text = dict(zip(corpus.id, corpus.text.str.lower()))
    pq = pd.read_csv(B / "L1_per_query.csv").merge(queries[["qid", "query"]], on="qid")
    summ = pd.read_csv(B / "L1_summary.csv")
    runs = pickle.load(open(B / "L1_runs.pkl", "rb"))["runs"]
    best = summ[summ.query_type == "all"].sort_values("ndcg@10", ascending=False).iloc[0].system

    # --- error buckets
    t = pq[pq.split == "test"].copy()
    t["n_relevant"] = t.qid.map(lambda q: len(qrels[q]))
    t["rel_bucket"] = pd.cut(t.n_relevant, [0, 10, 50, 1000], labels=["<=10", "11-50", ">50"])
    t["len_bucket"] = pd.cut(t["query"].str.split().str.len(), [0, 2, 15, 200], labels=["1-2 words", "3-15", ">15"])
    rows = []
    for col in ("kind", "rel_bucket", "len_bucket"):
        g = t.groupby(["system", col], observed=True)["ndcg@10"].agg(["mean", "count"]).reset_index()
        g.columns = ["system", "bucket_value", "ndcg@10", "n"]
        g.insert(1, "bucket", col)
        rows.append(g)
    pd.concat(rows).round(4).to_csv(B / "L3_error_buckets.csv", index=False)

    # --- silver-label audit: retrieved-but-"irrelevant" courses that literally mention the skill
    audit = []
    for qid, sk in zip(queries.qid, queries.skill):
        if qid not in runs[best]:
            continue
        pat = re.compile(r"(?<![\w+#])" + re.escape(sk.lower()) + r"(?![\w+#])")
        for d in runs[best][qid][:10]:
            if d not in qrels[qid]:
                audit.append({"qid": qid, "skill": sk, "doc_id": d, "title": title[d],
                              "mentions_skill": bool(pat.search(text[d]))})
    a = pd.DataFrame(audit)
    a.to_csv(B / "L3_label_audit.csv", index=False)
    if len(a):
        print(f"{best}: {len(a)} top-10 non-relevant hits, {a.mentions_skill.mean():.1%} mention the skill verbatim "
              "(candidate silver-label false negatives -> human check)")

    # --- worst queries
    w = t[t.system == best].nsmallest(10, "ndcg@10")
    w["top3_titles"] = w.qid.map(lambda q: " | ".join(title[d] for d in runs[best][q][:3]))
    w[["qid", "kind", "skill", "query", "n_relevant", "ndcg@10", "top3_titles"]].to_csv(B / "L3_worst_queries.csv", index=False)

    # --- compact tables for the showcase UI (Experiments page)
    lb = summ[summ.query_type == "all"][["system", "ndcg@10", "ndcg@10_lo", "ndcg@10_hi", "map@100", "mrr@10", "recall@100"]]
    lb.sort_values("ndcg@10", ascending=False).round(3).to_csv(B / "L1_leaderboard.csv", index=False)
    st5 = pd.read_csv(B / "L2_strategies.csv")
    st5[st5.k == 5].assign(method=lambda d: d.retriever + " / " + d.strategy)[
        ["method", "weighted_coverage", "cov_lo", "cov_hi", "req_covered", "precision", "redundancy"]] \
        .sort_values("weighted_coverage", ascending=False).round(3).to_csv(B / "L2_leaderboard_k5.csv", index=False)

    # --- figures
    s = summ[summ.query_type == "all"].sort_values("ndcg@10")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1.2))
    y = range(len(s))
    ax.hlines(y, s["ndcg@10_lo"], s["ndcg@10_hi"], color=SERIES[0], linewidth=2, alpha=0.45)
    ax.plot(s["ndcg@10"], y, "o", color=SERIES[0], markersize=7)
    for yi, (v, hi) in enumerate(zip(s["ndcg@10"], s["ndcg@10_hi"])):
        ax.text(hi + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y), s.system, color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("nDCG@10 (test split, mean and 95% bootstrap CI)", color=MUTED)
    ax.set_title("Course retrieval: skill-gap query -> Coursera course", loc="left", color=INK, fontsize=11)
    _style(ax)
    plt.tight_layout()
    plt.savefig(B / "fig_L1_retrieval.png", dpi=160)
    plt.close()

    st = pd.read_csv(B / "L2_strategies.csv")
    st = st[st.k == 5]
    order = st[st.strategy == "xquad"].sort_values("weighted_coverage", ascending=False).retriever.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = {"concat": "one concatenated query", "pergap": "per-gap queries", "xquad": "per-gap + xQuAD diversification"}
    for i, strat in enumerate(["concat", "pergap", "xquad"]):
        d = st[st.strategy == strat].set_index("retriever").loc[order]
        xs = [j + (i - 1) * 0.26 for j in range(len(order))]
        ax.bar(xs, d.weighted_coverage, width=0.24, color=SERIES[i], label=labels[strat],
               edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(order)), order, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("priority-weighted gap coverage @5", color=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=8.5)
    ax.set_title("Learning plan quality on 23 real JD/resume gap profiles", loc="left", color=INK, fontsize=11, pad=26)
    plt.tight_layout()
    plt.savefig(B / "fig_L2_strategies.png", dpi=160)
    plt.close()

    ab = pd.read_csv(B / "L2_ablation.csv")
    lam = ab[ab.config.str.match(r"^xQuAD λ=[\d.]+$")].copy()
    lam["lambda"] = lam.config.str.extract(r"([\d.]+)$")[0].astype(float)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(lam["lambda"], lam.weighted_coverage, "-o", color=SERIES[0], linewidth=2, markersize=6, label="weighted coverage")
    ax.plot(lam["lambda"], lam.redundancy, "-o", color=SERIES[1], linewidth=2, markersize=6, label="redundancy")
    ax.set_xlabel("λ (0 = relevance only, 1 = diversification only)", color=MUTED)
    ax.set_ylim(0, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("xQuAD trade-off (fusion retriever, k=5)", loc="left", color=INK, fontsize=11)
    plt.tight_layout()
    plt.savefig(B / "fig_L2_lambda.png", dpi=160)
    plt.close()
    print("figures ->", B)


if __name__ == "__main__":
    main()
