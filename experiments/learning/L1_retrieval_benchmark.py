"""L1 — Course retrieval benchmark: 10+ retrievers, dev-set tuning, test-set comparison.

Run from the project root:
    python experiments/learning/L1_retrieval_benchmark.py              # lexical / topic / embedding (CPU, ~3 min)
    python experiments/learning/L1_retrieval_benchmark.py --neural     # + Sentence-BERT, cross-encoder (needs HF models)

Outputs (results/learning/bench/):
    L1_dev_grid.csv          hyper-parameter search on the dev split
    L1_summary.csv           test-split mean + 95% bootstrap CI per system and query type
    L1_significance.csv      paired randomisation test vs BM25 (Holm-corrected)
    L1_per_query.csv         per-query metrics (for error analysis)
    L1_efficiency.csv        index time, latency per query, index size
    L1_lda_topics.csv        top words per LDA topic (interpretation)
    L1_runs.pkl              ranked lists (consumed by L2/L3)
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning.benchmark import load_benchmark  # noqa: E402
from src.learning.ir_metrics import METRICS, bootstrap_ci, evaluate_run, holm, paired_randomization  # noqa: E402
from src.learning import retrievers as R  # noqa: E402

OUT = ROOT / "results" / "learning" / "bench"
PRIMARY = "ndcg@10"


def to_run(ret, queries, doc_ids, k=100):
    t = time.perf_counter()
    idx, _ = ret.rank(list(queries["query"]), k)
    secs = time.perf_counter() - t
    return {qid: [doc_ids[j] for j in row] for qid, row in zip(queries.qid, idx)}, secs


def mean_metric(run, qrels, qids, m=PRIMARY):
    return evaluate_run(run, qrels, qids)[m].mean()


def sizeof(ret):
    try:
        return len(pickle.dumps(ret, protocol=4)) / 2 ** 20
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true", help="add Sentence-BERT / cross-encoder systems")
    ap.add_argument("--dense-models", default="sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2,BAAI/bge-small-en-v1.5")
    ap.add_argument("--finetuned", default="", help="path to a bi-encoder fine-tuned by L4_finetune_biencoder.py")
    ap.add_argument("--quick", action="store_true", help="small grids (for tests)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels, _ = load_benchmark()
    texts, doc_ids = list(corpus.text), list(corpus.id)
    dev, test, train = (queries[queries.split == s] for s in ("dev", "test", "train"))
    print(f"corpus={len(corpus)} queries: train={len(train)} dev={len(dev)} test={len(test)}")

    fitted, eff = {}, []

    def fit(ret):
        t = time.perf_counter()
        ret.fit(texts)
        fit_s = time.perf_counter() - t
        fitted[ret.name] = ret
        return ret, fit_s

    # ---------------- 1. dev-set tuning ----------------
    grid_rows = []
    k1s, bs = ([0.9, 1.2], [0.75]) if args.quick else ([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9])
    for k1, b in itertools.product(k1s, bs):
        r, _ = fit(R.BM25Retriever(k1, b))
        run, _ = to_run(r, dev, doc_ids)
        grid_rows.append({"family": "BM25", "params": f"k1={k1},b={b}", PRIMARY: mean_metric(run, qrels, dev.qid)})
    for k in ([100] if args.quick else [100, 200, 300, 500]):
        r, _ = fit(R.LSARetriever(k))
        run, _ = to_run(r, dev, doc_ids)
        grid_rows.append({"family": "LSA", "params": f"k={k}", PRIMARY: mean_metric(run, qrels, dev.qid)})
    for k in ([50] if args.quick else [50, 100, 200]):
        r, _ = fit(R.LDARetriever(k, max_iter=10 if args.quick else 20))
        run, _ = to_run(r, dev, doc_ids)
        grid_rows.append({"family": "LDA", "params": f"k={k}", PRIMARY: mean_metric(run, qrels, dev.qid)})
    for analyzer, ng in [("word", (1, 1)), ("word", (1, 2)), ("char_wb", (3, 5))]:
        r, _ = fit(R.TfidfRetriever(analyzer, ng))
        run, _ = to_run(r, dev, doc_ids)
        grid_rows.append({"family": "TF-IDF", "params": f"{analyzer},{ng}", PRIMARY: mean_metric(run, qrels, dev.qid)})
    grid = pd.DataFrame(grid_rows)

    def best(family):
        g = grid[grid.family == family]
        return g.loc[g[PRIMARY].idxmax(), "params"]

    def pick(prefix_cls, params):
        for name, ret in fitted.items():
            if name.startswith(prefix_cls) and params.replace(" ", "") in name.replace(" ", ""):
                return ret
        raise KeyError(prefix_cls + params)

    bm25 = pick("BM25", f"k1={best('BM25').split(',')[0].split('=')[1]},b={best('BM25').split(',')[1].split('=')[1]}")
    lsa = pick("LSA", f"k={best('LSA').split('=')[1]}")
    lda = pick("LDA", f"k={best('LDA').split('=')[1]}")
    tf_word = fitted["TF-IDF(word,1-2)"]
    tf_char = fitted["TF-IDF(char_wb,3-5)"]
    print("dev-selected:", bm25.name, lsa.name, lda.name)

    glove, _ = fit(R.EmbeddingRetriever("glove-wiki-gigaword-100"))
    try:
        w2v, _ = fit(R.EmbeddingRetriever("word2vec"))
    except ImportError:  # gensim has no wheel for some Python versions; report instead of failing
        w2v = None
        print("[skipped] Word2Vec: gensim not installed (results from an environment with gensim are kept in git history)")
    systems = {"BM25 (default k1=1.2,b=0.75)": fitted["BM25(k1=1.2,b=0.75)"], "BM25 (dev-tuned)": bm25,
               "TF-IDF word 1-2gram": tf_word, "TF-IDF char 3-5gram": tf_char,
               "LSA (dev-tuned)": lsa, "LDA topics (dev-tuned)": lda,
               "GloVe-100 IDF-avg": glove}
    if w2v is not None:
        systems["Word2Vec (trained on corpus)"] = w2v

    neural = {}
    if args.neural:
        for m in [x for x in args.dense_models.split(",") if x]:
            d, _ = fit(R.DenseRetriever(m))
            neural[f"Dense {m.split('/')[-1]}"] = d
        if args.finetuned:
            d, _ = fit(R.DenseRetriever(args.finetuned))
            neural["Dense fine-tuned (L4)"] = d
        systems.update(neural)

    # fusion: weights tuned on dev
    lex_sem = [bm25, lsa, glove] + ([list(neural.values())[0]] if neural else [])
    systems["RRF (BM25+LSA+GloVe" + ("+Dense" if neural else "") + ")"] = R.RRFFusion(lex_sem)
    best_w, best_v = None, -1
    for w in itertools.product([0, 0.5, 1.0], repeat=len(lex_sem)):
        if not any(w):
            continue
        lin = R.LinearFusion(lex_sem, list(w))
        v = mean_metric(to_run(lin, dev, doc_ids)[0], qrels, dev.qid)
        grid_rows.append({"family": "LinearFusion", "params": str(w), PRIMARY: v})
        if v > best_v:
            best_w, best_v = list(w), v
    systems["Linear fusion (dev-tuned weights)"] = R.LinearFusion(lex_sem, best_w)

    # learning to rank on train, C tuned on dev
    doc_feats = np.stack([np.log1p(corpus.description.str.len().values),
                          corpus.rating.fillna(corpus.rating.median()).values,
                          (corpus.level == "Beginner").astype(float).values], 1)
    lab = lambda qs: [{doc_ids.index(d) for d in qrels[q]} for q in qs.qid]  # noqa: E731
    members = [m for m in (bm25, tf_char, lsa, lda, glove, w2v) if m is not None] + list(neural.values())
    best_ltr, best_v = None, -1
    for C in ([1.0] if args.quick else [0.01, 0.1, 1.0, 10.0]):
        ltr = R.LTRReranker(members, doc_feats, "logreg", C=C).train(list(train["query"]), lab(train))
        v = mean_metric(to_run(ltr, dev, doc_ids)[0], qrels, dev.qid)
        grid_rows.append({"family": "LTR-logreg", "params": f"C={C}", PRIMARY: v})
        if v > best_v:
            best_ltr, best_v = ltr, v
    systems["LTR logistic regression (train/dev)"] = best_ltr
    gbt = R.LTRReranker(members, doc_feats, "gbt").train(list(train["query"]), lab(train))
    grid_rows.append({"family": "LTR-gbt", "params": "depth=3", PRIMARY: mean_metric(to_run(gbt, dev, doc_ids)[0], qrels, dev.qid)})
    systems["LTR gradient boosting (train)"] = gbt
    if neural:
        ce, _ = fit(R.CrossEncoderReranker(systems["Linear fusion (dev-tuned weights)"]))
        systems["Linear fusion >> cross-encoder rerank"] = ce
    pd.DataFrame(grid_rows).to_csv(OUT / "L1_dev_grid.csv", index=False)

    # ---------------- 2. test evaluation ----------------
    per_query, runs, eff = [], {}, []
    allq = pd.concat([dev, test])
    for name, ret in systems.items():
        run, secs = to_run(ret, allq, doc_ids)
        runs[name] = run
        pq = evaluate_run(run, qrels, allq.qid).merge(allq[["qid", "kind", "split", "skill"]], on="qid")
        pq["system"] = name
        per_query.append(pq)
        eff.append({"system": name, "ms_per_query": 1000 * secs / len(allq), "index_MB": sizeof(ret)})
        print(f"{name:45s} test nDCG@10={pq[pq.split == 'test'][PRIMARY].mean():.3f}")
    pq = pd.concat(per_query)
    pq.to_csv(OUT / "L1_per_query.csv", index=False)

    rows = []
    for (name, kind), g in pq[pq.split == "test"].groupby(["system", "kind"]):
        row = {"system": name, "query_type": kind, "n": len(g)}
        for m in METRICS:
            lo, hi = bootstrap_ci(g[m])
            row[m], row[m + "_lo"], row[m + "_hi"] = g[m].mean(), lo, hi
        rows.append(row)
    for name, g in pq[pq.split == "test"].groupby("system"):
        row = {"system": name, "query_type": "all", "n": len(g)}
        for m in METRICS:
            lo, hi = bootstrap_ci(g[m])
            row[m], row[m + "_lo"], row[m + "_hi"] = g[m].mean(), lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["query_type", PRIMARY], ascending=[True, False])
    summary.round(4).to_csv(OUT / "L1_summary.csv", index=False)

    base = "BM25 (dev-tuned)"
    t = pq[pq.split == "test"].pivot_table(index="qid", columns="system", values=PRIMARY)
    pv = {s: paired_randomization(t[s], t[base]) for s in t.columns if s != base}
    adj = holm(pv)
    pd.DataFrame([{"system": s, "baseline": base, "delta_ndcg@10": t[s].mean() - t[base].mean(),
                   "p_raw": pv[s], "p_holm": adj[s], "significant_0.05": adj[s] < 0.05} for s in pv]) \
        .sort_values("delta_ndcg@10", ascending=False).round(4).to_csv(OUT / "L1_significance.csv", index=False)

    # efficiency: fit time measured separately on a fresh instance for the main families
    for label, mk in [("BM25", lambda: R.BM25Retriever()), ("TF-IDF word", lambda: R.TfidfRetriever()),
                      ("LSA", lambda: R.LSARetriever(lsa.k)), ("LDA", lambda: R.LDARetriever(lda.k)),
                      ("Word2Vec", lambda: R.EmbeddingRetriever("word2vec"))]:
        try:
            _, s = R.timed(mk().fit, texts)
        except ImportError:
            continue
        eff.append({"system": label + " (index build)", "index_build_s": s})
    pd.DataFrame(eff).round(3).to_csv(OUT / "L1_efficiency.csv", index=False)
    pd.DataFrame({"topic": range(lda.k), "top_words": [" ".join(w) for w in lda.top_words(10)]}) \
        .to_csv(OUT / "L1_lda_topics.csv", index=False)
    with open(OUT / "L1_runs.pkl", "wb") as f:
        pickle.dump({"runs": runs, "selected": {"bm25": bm25.name, "lsa": lsa.name, "lda": lda.name,
                                                 "linear_weights": best_w, "ltr": best_ltr.name}}, f)
    print(summary[summary.query_type == "all"][["system", "n", PRIMARY, PRIMARY + "_lo", PRIMARY + "_hi", "map@100", "mrr@10"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
