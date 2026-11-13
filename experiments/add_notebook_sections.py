"""Append (or refresh) notebook sections 9.3-9.7 and 10 that present the new experiments.

    python experiments/add_notebook_sections.py

Idempotent: cells carrying the metadata tag "upgrade-2026-11" are removed and re-inserted.
The cells only *read* result files (fast to run); each one names the script that produced them.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Group32_NLPproject_extended.ipynb"
TAG = "upgrade-2026-11"


def md(text):
    return {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {"tags": [TAG]}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
### 9.3 From 12 hand-picked units to a 3,428-course benchmark

The first version of this module could only be checked on 12 author-written cases. To evaluate it like the
rest of the project, we built a benchmark from public data (`src/learning/benchmark.py`):

| Component | Source | Size |
|---|---|---|
| Corpus | Coursera catalogue (title + description) + 12 curated official-doc units | 3,428 resources |
| Skill queries | matcher skill vocabulary with ≥3 tagged courses | 96 |
| Sentence queries | real JD requirement sentences (REQUIRED/PREFERRED labels) naming one of those skills | 37 |
| Gap profiles | real JD (30 labelled) × real resume → skills missing from the resume, with priority | 23 profiles, 4.3 gaps each |
| Silver relevance | Coursera's own skill tags (**never shown to the retrievers**) | 5,642 query–course labels |

Splits are by **skill** (60/20/20): a test skill never appears in training, so the supervised rankers
cannot memorise it. Common-word skills (Excel, Agile, Swift…) must keep their capitalisation, which removed
false queries such as *"You'll excel if you can collaborate"* produced by the dictionary extractor.
"""),
    code("""
import json, os
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
if "learning_root" in globals():
    os.chdir(learning_root)  # all paths below are relative to the project root
B = Path("results/learning/bench")
print(json.dumps({k: v for k, v in json.load(open("data/learning/benchmark/stats.json")).items() if k != "domains"}, indent=1))
display(pd.read_csv("data/learning/benchmark/queries.csv").groupby(["kind", "split"]).size().unstack())
"""),
    md("""
### 9.4 Retrieval: 12 systems, one per EE6405 topic (`experiments/learning/L1_retrieval_benchmark.py`)

BM25 (term weighting), TF-IDF word / char n-grams (BoW), LSA (SVD feature extraction), LDA (topic model),
GloVe and corpus-trained Word2Vec (word embeddings), RRF and linear fusion (ensembles), and learning-to-rank
with logistic regression / gradient boosting over all of them (supervised ML). With `--neural`, Sentence-BERT
bi-encoders, a cross-encoder re-ranker and our fine-tuned bi-encoder (L4) are added.

**Result (RTX 5060 run, test nDCG@10):** LTR 0.730, fusion + cross-encoder re-rank 0.719, BM25 0.684,
pre-trained Sentence-BERT 0.62–0.63, our fine-tuned bi-encoder 0.619 (best dev score, highest Recall@100 among
dense models, but no gain on unseen test skills with only 75 training queries), GloVe 0.325, LDA 0.164.
Only GloVe and LDA differ significantly from BM25. Word2Vec (0.594) comes from the cloud CPU run
(`results/learning/bench_cloud_cpu/`) because gensim cannot be installed on Python 3.14.
Hyper-parameters (BM25 k1/b, LSA k, LDA k, fusion weights, LTR C) are chosen on the dev split only.
Metrics: nDCG@10 (primary), MAP@100, MRR@10, P@5, Recall@10/100, with 95% bootstrap CIs and a paired
randomisation test against BM25 (Holm-corrected).
"""),
    code("""
display(pd.read_csv(B / "L1_leaderboard.csv"))
display(Image(B / "fig_L1_retrieval.png", width=650))
display(pd.read_csv(B / "L1_significance.csv"))
s = pd.read_csv(B / "L1_summary.csv")
display(s[s.query_type != "all"].pivot_table(index="system", columns="query_type", values="ndcg@10").round(3)
        .sort_values("skill", ascending=False))
display(pd.read_csv(B / "L1_efficiency.csv"))
"""),
    md("""
### 9.5 From a ranked list to a learning plan (`experiments/learning/L2_plan_evaluation.py`)

A candidate missing *SQL (REQUIRED)* and *Docker (PREFERRED)* does not need five SQL courses. We compare
three ways to build a k-course plan from the gaps: one concatenated query, per-gap queries summed with
priority weights (the original SkillBridge design), and **xQuAD** aspect diversification
(`src/learning/planner.py`). Our xQuAD variant normalises the diversity term by the still-uncovered
priority weight and draws candidates from each gap's own top results; the ablation shows which change matters.
Metric: priority-weighted share of gaps covered by at least one chosen course (silver labels), plus
precision and redundancy.
"""),
    code("""
display(pd.read_csv(B / "L2_leaderboard_k5.csv"))
display(Image(B / "fig_L2_strategies.png", width=700))
display(pd.read_csv(B / "L2_significance.csv"))
display(pd.read_csv(B / "L2_ablation.csv"))
display(Image(B / "fig_L2_lambda.png", width=520))
"""),
    md("""
### 9.6 Error analysis and label audit (`experiments/learning/L3_error_analysis.py`)

* Buckets: query type (skill name vs JD sentence), query length, number of relevant courses.
* **Silver-label audit:** among the best system's top-10 courses that the silver labels call *non-relevant*,
  we count how many literally mention the skill in their description. These are likely missing tags, so the
  absolute scores are a lower bound; this motivated the human judgements in 9.7.
"""),
    code("""
eb = pd.read_csv(B / "L3_error_buckets.csv")
display(eb[eb.system.isin(["BM25 (dev-tuned)", "LTR logistic regression (train/dev)", "Word2Vec (trained on corpus)"])]
        .pivot_table(index=["bucket", "bucket_value"], columns="system", values="ndcg@10").round(3))
audit = pd.read_csv(B / "L3_label_audit.csv")
print(f"{len(audit)} top-10 'non-relevant' hits; {audit.mentions_skill.mean():.1%} mention the skill verbatim")
display(audit[audit.mentions_skill].head(8))
display(pd.read_csv(B / "L3_worst_queries.csv").head(6))
"""),
    md("""
### 9.7 Live demo of the benchmark winner and human evaluation

`src/learning/course_search.py` serves the configuration that won above (dev-tuned BM25 per gap + normalised
xQuAD) over all 3,428 courses; every recommendation shows the description sentence that names the skill.
The same planner is exposed at `POST /api/courses` in `showcase_app.py`.

Human judgements: `python experiments/learning/L5_human_judgements.py make` writes two blind, pooled sheets
(`data/learning/human/`). After two team members grade them, `... score` reports Cohen's κ between annotators,
agreement between silver and human labels, and re-scores every system on the human labels.
"""),
    code("""
from src.learning.course_search import CourseSearch
cs = CourseSearch()
demo_gaps = [{"skill": g["skill"], "priority": g["priority"]} for g in learning_plan["gaps"]] if "learning_plan" in globals() \\
    else [{"skill": "SQL", "priority": "REQUIRED"}, {"skill": "Docker", "priority": "PREFERRED"}, {"skill": "Tableau", "priority": "REQUIRED"}]
plan_out = cs.plan(demo_gaps, k=5)
display(pd.DataFrame([{"rank": r["rank"], "course": r["title"], "provider": r["provider"],
                       "covers": ", ".join(r["evidence"]), "evidence": " | ".join(r["evidence"].values())[:160]}
                      for r in plan_out["recommendations"]]))
print("evidence coverage", plan_out["evidence_coverage"], "uncovered", plan_out["uncovered"], f"{plan_out['latency_ms']} ms")
for f in ["L5_agreement.csv", "L5_human_eval.csv"]:
    if (B / f).exists():
        display(pd.read_csv(B / f))
"""),
    md("""
---
## 10. Traditional NLP vs transformers on the fit benchmark (`experiments/step9_classic_vs_neural.py`)

Section 5 compared lexical baselines with transformer matchers. Step 9 fills in the rest of the course's
traditional toolbox on **the same 600 test pairs**: LSA (D9), LDA topic similarity (D10), GloVe and
Word2Vec averaged embeddings (D11), and four supervised fit classifiers on pair features (D12: logistic
regression, linear SVM, multinomial NB, random forest; trained on the 3,000 training pairs). Hyper-parameters
are chosen on a JD-grouped dev split of the training pairs. Optionally (`--finetune-ce`, GPU) a cross-encoder
is fine-tuned with a learning-rate × epoch grid.

**Result (RTX 5060 run):** supervised classifiers on TF-IDF pair features are the best matchers on this
benchmark (Linear SVM ρ = 0.317 [0.188, 0.447], random forest 0.310), ahead of the fine-tuned bi-encoder D7
(0.281) and the newly fine-tuned cross-encoder D13 (0.181, up from −0.075 zero-shot). Unsupervised LSA / LDA /
GloVe and zero-shot transformers are near zero. The logistic-regression weights explain why: its 20 strongest
features are all JD-side words, i.e. the classifiers mostly learn which vacancies tend to be labelled "good fit"
— the same vacancy prior found in Section 5 — so these scores should not be read as person–job fit.

Also reported: efficiency (fit time, ms per pair, peak memory) and interpretation — logistic-regression term
weights, LDA topics that separate good-fit from no-fit resumes, and sentence-occlusion explanations that show
which resume sentences drive each model's score.
"""),
    code("""
S9 = Path("results/step9")
if not (S9 / "S9_matching_all.csv").exists():
    print("Run: python experiments/step9_classic_vs_neural.py   (needs data/hf from step1)")
else:
    display(pd.read_csv(S9 / "S9_matching_all.csv"))
    display(pd.read_csv(S9 / "S9_classifiers.csv"))
    display(pd.read_csv(S9 / "S9_efficiency.csv"))
    display(pd.read_csv(S9 / "S9_logreg_top_terms.csv"))
    display(pd.read_csv(S9 / "S9_lda_topics.csv").head(8))
    display(pd.read_csv(S9 / "S9_occlusion_examples.csv"))
"""),
]


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"] = [c for c in nb["cells"] if TAG not in c.get("metadata", {}).get("tags", [])]
    nb["cells"].extend(CELLS)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"notebook now has {len(nb['cells'])} cells ({len(CELLS)} from this upgrade)")


if __name__ == "__main__":
    main()
