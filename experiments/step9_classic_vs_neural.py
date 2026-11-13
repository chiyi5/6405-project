"""
STEP 9: traditional-NLP vs transformer matchers on the same 600 test pairs,
supervised fit classifiers, efficiency benchmark and model interpretation.

    python experiments/step9_classic_vs_neural.py                   # CPU, ~10-20 min
    python experiments/step9_classic_vs_neural.py --finetune-ce     # + cross-encoder fine-tuning grid (GPU)
    python experiments/step9_classic_vs_neural.py --hf-dir <dir>    # use another train/test.csv.gz folder

Needs data/hf/train.csv.gz and test.csv.gz (created by step1_prepare.py). Uses the same
balanced samples as step2 (600 test pairs / 3000 train pairs, same seeds), so the numbers
are directly comparable with results/D_matching.csv and D7_finetune.csv.

Outputs: results/step9/
    S9_dev_selection.csv        hyper-parameters chosen on a JD-grouped dev split of the training pairs
    S9_matching_all.csv         every matcher (old D1-D7 + new D9-D12): Spearman [95% CI by JD], AUC, macro-F1, NDCG@5
    S9_classifiers.csv          supervised classifiers: accuracy, macro-F1, per-class F1 on the test pairs
    S9_confusion_<model>.csv    confusion matrix of the best classifier
    S9_efficiency.csv           fit time, ms per pair, peak memory (tracemalloc) per matcher
    S9_logreg_top_terms.csv     interpretation: most predictive terms for good / no fit
    S9_lda_topics.csv           interpretation: LDA topics and their mean weight in good- vs no-fit resumes
    S9_occlusion_examples.csv   interpretation: which resume sentences drive each model's score
    S9_ce_finetune_grid.csv     (optional) cross-encoder lr x epochs grid on dev
    run_log_step9.txt
"""
import argparse
import json
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import matching as mt  # noqa: E402
from src.classic_nlp import (LDAMatcher, LSAMatcher, PairClassifier, WordVectorMatcher,  # noqa: E402
                             sentence_occlusion)
from src.stats import cluster_bootstrap_spearman  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf-dir", default=str(ROOT / "data" / "hf"))
ap.add_argument("--n_eval", type=int, default=600)
ap.add_argument("--n_train", type=int, default=3000)
ap.add_argument("--finetune-ce", action="store_true")
ap.add_argument("--quick", action="store_true", help="tiny grids, for smoke tests")
ap.add_argument("--out", default=str(ROOT / "results" / "step9"))
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "run_log_step9.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def balanced(df, n, seed=0):  # identical to step2_run_all.balanced
    k = max(1, n // df.label.nunique())
    return pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in df.groupby("label")]).reset_index(drop=True)


H = Path(args.hf_dir)
train, test = pd.read_csv(H / "train.csv.gz"), pd.read_csv(H / "test.csv.gz")
ev, tr = balanced(test, args.n_eval), balanced(train, args.n_train)
log(f"eval {len(ev)} pairs / {ev.jd_id.nunique()} JDs; train {len(tr)} pairs / {tr.jd_id.nunique()} JDs")

# JD-grouped dev split of the training pairs for model selection (test pairs never used for tuning)
rng = np.random.default_rng(0)
jds = tr.jd_id.unique()
dev_j = set(rng.choice(jds, max(1, len(jds) // 5), replace=False))
tr_fit, tr_dev = tr[~tr.jd_id.isin(dev_j)], tr[tr.jd_id.isin(dev_j)]
corpus_train = pd.concat([tr.jd_text, tr.resume_text]).drop_duplicates().tolist()
corpus_fit = pd.concat([tr_fit.jd_text, tr_fit.resume_text]).drop_duplicates().tolist()


def spearman(m, df):
    from scipy.stats import spearmanr
    s = m.score_pairs(df.jd_text.tolist(), df.resume_text.tolist())
    return spearmanr(s, df.label).statistic


# ------------------------------------------------------------------ 1. dev selection
sel = []
best = {}
grids = {"LSA": [100] if args.quick else [50, 100, 200, 400],
         "LDA": [20] if args.quick else [20, 50, 100],
         "logreg": [1.0] if args.quick else [0.01, 0.1, 1.0, 10.0],
         "linear_svm": [1.0] if args.quick else [0.01, 0.1, 1.0]}
for k in grids["LSA"]:
    sel.append({"family": "LSA", "param": k, "dev_spearman": spearman(LSAMatcher(k).fit(corpus_fit), tr_dev)})
for k in grids["LDA"]:
    sel.append({"family": "LDA", "param": k, "dev_spearman": spearman(LDAMatcher(k, max_iter=10 if args.quick else 20).fit(corpus_fit), tr_dev)})
for fam in ("logreg", "linear_svm"):
    for C in grids[fam]:
        sel.append({"family": fam, "param": C, "dev_spearman": spearman(PairClassifier(fam, C=C).train(tr_fit), tr_dev)})
sel = pd.DataFrame(sel)
sel.round(4).to_csv(OUT / "S9_dev_selection.csv", index=False)
for fam, g in sel.groupby("family"):
    best[fam] = g.loc[g.dev_spearman.idxmax(), "param"]
log("dev-selected:", best)

# ------------------------------------------------------------------ 2. final matchers (refit on all training pairs)
eff = []


def build(label, make):
    tracemalloc.start()
    t = time.perf_counter()
    m = make()
    fit_s = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    eff.append({"matcher": m.name, "fit_s": round(fit_s, 2), "fit_peak_MB": round(peak, 1)})
    log(f"built {m.name} in {fit_s:.1f}s")
    return m


matchers = [
    build("lsa", lambda: LSAMatcher(int(best["LSA"])).fit(corpus_train)),
    build("lda", lambda: LDAMatcher(int(best["LDA"])).fit(corpus_train)),
    build("glove", lambda: WordVectorMatcher("glove-wiki-gigaword-100").fit(corpus_train)),
    build("nb", lambda: PairClassifier("naive_bayes").train(tr)),
    build("lr", lambda: PairClassifier("logreg", C=float(best["logreg"])).train(tr)),
    build("svm", lambda: PairClassifier("linear_svm", C=float(best["linear_svm"])).train(tr)),
    build("rf", lambda: PairClassifier("random_forest").train(tr)),
]
try:
    matchers.insert(3, build("w2v", lambda: WordVectorMatcher("word2vec").fit(corpus_train)))
except ImportError:
    log("[skipped] Word2Vec matcher: gensim not installed")
lexical = [mt.TfidfMatcher(), mt.BM25Matcher()]  # re-run for latency comparison only

if args.finetune_ce:
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader

        class CEMatcher:
            def __init__(self, model, name):
                self.m, self.name, self._fitted = model, name, True

            def score_pairs(self, jds, resumes):
                return self.m.predict([(j[:2000], r[:2000]) for j, r in zip(jds, resumes)], show_progress_bar=False)

        grid = []
        ex = [InputExample(texts=[j[:2000], r[:2000]], label=float(l) / 2) for j, r, l in
              zip(tr_fit.jd_text, tr_fit.resume_text, tr_fit.label)]
        best_ce, best_v = None, -9
        for lr in ([2e-5] if args.quick else [1e-5, 2e-5, 5e-5]):
            for ep in ([1] if args.quick else [1, 2]):
                ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1, max_length=512)
                ce.fit(train_dataloader=DataLoader(ex, shuffle=True, batch_size=16), epochs=ep, warmup_steps=50,
                       optimizer_params={"lr": lr}, show_progress_bar=False)
                v = spearman(CEMatcher(ce, "tmp"), tr_dev)
                grid.append({"lr": lr, "epochs": ep, "dev_spearman": v})
                log(f"CE lr={lr} ep={ep} dev rho={v:.3f}")
                if v > best_v:
                    best_ce, best_v, best_cfg = ce, v, (lr, ep)
        pd.DataFrame(grid).round({"dev_spearman": 4}).to_csv(OUT / "S9_ce_finetune_grid.csv", index=False)  # keep lr unrounded
        matchers.append(CEMatcher(best_ce, f"D13_finetuned_cross_encoder_lr{best_cfg[0]:g}_ep{best_cfg[1]}"))
    except Exception:
        log("cross-encoder fine-tuning failed:\n" + traceback.format_exc()[-800:])

# ------------------------------------------------------------------ 3. evaluation on the 600 test pairs
summ, scores = mt.evaluate_matchers(matchers + lexical, ev, k=5)
old_scores = ROOT / "results" / "D_pair_scores.csv"
if old_scores.exists():  # merge earlier transformer scores for one leaderboard with CIs
    old = pd.read_csv(old_scores)
    scores = scores.merge(old.drop(columns=["label"] + [c for c in old.columns if c in scores.columns and c not in ("jd_id", "resume_id")]),
                          on=["jd_id", "resume_id"], how="left")
rows = []
for col in [c for c in scores.columns if c not in ("jd_id", "resume_id", "label")]:
    if scores[col].notna().sum() < 10:
        continue
    rho, lo, hi = cluster_bootstrap_spearman(scores, col)
    rows.append({"method": col, "spearman": rho, "ci_low": lo, "ci_high": hi, "n_pairs": int(scores[col].notna().sum())})
lb = pd.DataFrame(rows).merge(summ.drop(columns=["spearman", "n_pairs"], errors="ignore"), on="method", how="left")
d7 = ROOT / "results" / "D7_finetune.csv"
if d7.exists():
    lb = pd.concat([lb, pd.read_csv(d7).assign(note="from D7_finetune.csv (no per-pair scores, CI n/a)")])
family = {"D1": "lexical", "D2": "lexical", "D9": "latent semantic", "D10": "topic model", "D11": "word embedding",
          "D12": "supervised classifier", "D3": "transformer bi-encoder", "D4": "transformer bi-encoder",
          "D5": "transformer cross-encoder", "D6": "our pipeline", "D7": "fine-tuned transformer",
          "D13": "fine-tuned transformer"}
lb["family"] = lb.method.str.extract(r"^(D\d+)")[0].map(family)
lb.sort_values("spearman", ascending=False).round(3).to_csv(OUT / "S9_matching_all.csv", index=False)
scores.to_csv(OUT / "S9_pair_scores.csv", index=False)
log(lb.sort_values("spearman", ascending=False)[["method", "family", "spearman", "ci_low", "ci_high"]].round(3).to_string(index=False))

# supervised classifiers as 3-class classifiers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
crow = []
for m in [m for m in matchers if isinstance(m, PairClassifier)]:
    pred = m.predict(ev.jd_text.tolist(), ev.resume_text.tolist())
    f1s = f1_score(ev.label, pred, average=None, labels=[0, 1, 2])
    crow.append({"model": m.name, "accuracy": accuracy_score(ev.label, pred), "macro_f1": f1_score(ev.label, pred, average="macro"),
                 "f1_no_fit": f1s[0], "f1_potential": f1s[1], "f1_good": f1s[2]})
crow = pd.DataFrame(crow).sort_values("macro_f1", ascending=False)
crow.loc[len(crow)] = {"model": "majority class", "accuracy": ev.label.value_counts(normalize=True).max(),
                       "macro_f1": f1_score(ev.label, [ev.label.mode()[0]] * len(ev), average="macro")}
crow.round(4).to_csv(OUT / "S9_classifiers.csv", index=False)
bestc = next(m for m in matchers if getattr(m, "name", "") == crow.iloc[0].model)
pd.DataFrame(confusion_matrix(ev.label, bestc.predict(ev.jd_text.tolist(), ev.resume_text.tolist()), labels=[0, 1, 2]),
             index=["gold_no", "gold_potential", "gold_good"], columns=["pred_no", "pred_potential", "pred_good"]) \
    .to_csv(OUT / f"S9_confusion_{bestc.name}.csv")
log(crow.round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. efficiency
sample = ev.sample(min(100, len(ev)), random_state=1)
for m in matchers + lexical:
    if not getattr(m, "_fitted", True):
        m.fit(corpus_train)
    tracemalloc.start()
    t = time.perf_counter()
    m.score_pairs(sample.jd_text.tolist(), sample.resume_text.tolist())
    secs = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
    tracemalloc.stop()
    row = next((e for e in eff if e["matcher"] == m.name), None)
    if row is None:
        row = {"matcher": m.name}
        eff.append(row)
    row.update({"ms_per_pair": round(1000 * secs / len(sample), 2), "score_peak_MB": round(peak, 1)})
eff = pd.DataFrame(eff)
eff.to_csv(OUT / "S9_efficiency.csv", index=False)
log(eff.to_string(index=False))

# ------------------------------------------------------------------ 5. interpretation
lr = next(m for m in matchers if m.name.startswith("D12_logreg"))
lr.top_terms(20).round(4).to_csv(OUT / "S9_logreg_top_terms.csv", index=False)
lda = next(m for m in matchers if m.name.startswith("D10"))
th = lda.topics(ev.resume_text.tolist())
tj = lda.topics(ev.jd_text.tolist())
topics = pd.DataFrame({"topic": range(lda.k), "top_words": lda.top_words(10),
                       "mean_weight_resume_good": th[ev.label.values == 2].mean(0),
                       "mean_weight_resume_no": th[ev.label.values == 0].mean(0),
                       "mean_weight_jd": tj.mean(0)})
topics["good_minus_no"] = topics.mean_weight_resume_good - topics.mean_weight_resume_no
topics.sort_values("good_minus_no", ascending=False).round(4).to_csv(OUT / "S9_lda_topics.csv", index=False)

occ = []
explainers = {"BM25": lexical[1], "LSA": matchers[0], "LogReg": lr}
for lab_value in (2, 0):
    ex_row = ev[ev.label == lab_value].iloc[0]
    for name, m in explainers.items():
        f = lambda j, r, m=m: float(m.score_pairs([j], [r])[0])  # noqa: E731
        o = sentence_occlusion(f, ex_row.jd_text, ex_row.resume_text).head(3)
        o.insert(0, "model", name)
        o.insert(0, "pair", f"{ex_row.jd_id}/{ex_row.resume_id} (gold={lab_value})")
        occ.append(o)
pd.concat(occ).round(4).to_csv(OUT / "S9_occlusion_examples.csv", index=False)
log("done ->", OUT)
