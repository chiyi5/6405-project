"""Build Group32_NLPproject_extended.ipynb = original notebook (Part I, outputs kept) + Part II experiments."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG = Path(sys.argv[1])
nb = json.loads(ORIG.read_text(encoding="utf-8"))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


head = [md("""
# Group 32 — Explainable JD–Resume Matching: Pipeline, Benchmarks and Evaluation

**Part I** is our first end-to-end prototype (one JD, one resume) and serves as the **baseline system**.
**Part II** turns it into an evaluated NLP study: an annotated dataset, alternative methods for every
component, quantitative comparison, ablations, fairness/robustness tests and a demo.

| Member | Module | Research question | Main metrics |
|---|---|---|---|
| A | Data & annotation | Can humans agree on skills / priorities / fit? | Cohen's κ, Fleiss' κ |
| B | Skill extraction | Taxonomy NER vs. dictionary vs. LLM vs. fine-tuned NER | P / R / F1 (strict, soft) |
| C | Requirement classification | Rules vs. TF-IDF+LR vs. zero-shot NLI vs. DistilBERT | Accuracy, macro-F1 |
| D | Matching models | Lexical vs. bi-encoder vs. cross-encoder vs. our pipeline | Spearman ρ, AUC, macro-F1, NDCG@k |
| E | Scoring & ablation | Are hand-set thresholds/weights good? Which component matters? | Spearman ρ, Δ ablation |
| F | Fairness, robustness, errors, demo | Does the score change with names/wording? Where does it fail? | Δ score, error taxonomy |
"""),
        md("""# Part I — Baseline prototype (original notebook)

> **Read me first.** Part I is kept unchanged as the record of our first prototype and is the *baseline* in every
> comparison. It is **superseded by the modular code in `src/`** used in Part II. Some functions exist in both places
> with the same name but different behaviour (e.g. `clean_text`, `extract_education`, `classify_requirement`):
> Part II imports the `src/` versions explicitly, so the two never mix. For the current system, read `src/pipeline.py`.
""")]

for c in nb["cells"]:
    c.setdefault("metadata", {})

part2 = [
md("""
# Part II — Evaluated system

## 0. Setup
Install once: `pip install -r requirements.txt` (and for the ESCO baseline `python -m spacy download en_core_web_sm`).

* `USE_DEMO = True` runs everything on the tiny **synthetic** demo set in `data/demo/` to check the code.
  **Demo numbers must not be reported.** Switch to `False` after Member A has collected and annotated real data.
* `RUN_HEAVY = True` enables transformer models (zero-shot NLI, DistilBERT, cross-encoder, fine-tuning) — use a GPU (a T4-class GPU is enough).
"""),
code("""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))

from src.pipeline import MatchingPipeline, PipelineConfig, gap_report, clean_text
from src import data_utils as du, skill_extraction as se, requirement_clf as rc
from src import matching as mt, scoring as sc, fairness as fa, plots
from src.nlp_backend import Embedder, HAS_SPACY, HAS_ST

USE_DEMO  = True
RUN_HEAVY = False
EMBEDDER  = "sentence-transformers/all-MiniLM-L6-v2" if HAS_ST else "tfidf-char"
DATA      = Path("data/demo") if USE_DEMO else Path("data")
Path("results").mkdir(exist_ok=True)
print("spaCy:", HAS_SPACY, "| sentence-transformers:", HAS_ST, "| embedder:", EMBEDDER)
"""),
md("""
## 1. Bug fixes found while building the evaluation (Part I → Part II)
| Problem observed in Part I | Fix |
|---|---|
| `NumPy` split into "Num" / "Py," by the PDF line break | `fix_broken_words` re-joins tokens whose concatenation is a known skill |
| Projects section contained phone/e-mail/LinkedIn | contact lines are routed to a separate `contact` section |
| Generic phrases extracted as skills ("national initiative driven", "workshops") | dictionary extractor + post-processing filter (Member B) |
| "NITEC, Diploma, Bachelor's, Master's or PhD" → four separate requirements (PhD effectively required) | alternatives joined by *or* → one requirement at the **lowest** level |
| "Nanyang Technological University" in the company intro → education requirement | education rows without degree/field context are dropped |
| Bullets under "Nice to have:" were UNSPECIFIED | sentences inherit the priority of their section header |
| `safe_list` used `json` before import; experience score always `None` | fixed; experience gets a proportional score |
"""),
code("""
print(clean_text("Programming: Python (Pandas, Num\\nPy, Scikit-learn)"))
pipe = MatchingPipeline(PipelineConfig(embedder_name=EMBEDDER, skill_extractor="dictionary+esco"))
aisg = Path("data/demo/JD_example1_ntu_AI.txt").read_text(encoding="utf-8")
aisg_req = pipe.extract_jd_requirements(aisg)
display(aisg_req[["category", "requirement", "priority", "education_level"]])
"""),

md("""
---
## 2. Member A — Dataset construction & annotation

**Target size** (realistic for 6 people in ~2 weeks):
* 30–50 JDs from different roles (data / software / ML / business) — MyCareersFuture, LinkedIn, company sites
* 40–80 resumes — anonymised team/volunteer resumes + Kaggle *Resume Dataset* (`Resume.csv`)
* ~300 JD–resume pairs labelled **0 = No fit / 1 = Potential / 2 = Good fit** (each pair by 2 annotators)
* all JD sentences labelled MANDATORY / REQUIRED / PREFERRED / UNSPECIFIED (≥20% double-annotated)
* gold skill lists for ≥20 JDs and ≥20 resumes
* External: HuggingFace `cnamuangtoun/resume-job-description-fit` (labelled pairs) for training / extra testing.

Annotation guidelines: `data/annotation_guidelines.md`.
"""),
code("""
jds     = du.load_folder(DATA / "jds")
resumes = du.load_folder(DATA / "resumes")
# Optional: add Kaggle resumes
# kaggle = du.load_kaggle_resumes("data/raw/Resume.csv", categories=["INFORMATION-TECHNOLOGY", "ENGINEERING"], n_per_category=20)
# resumes.update(dict(zip(kaggle.resume_id, kaggle.resume_text)))
"""),
code("""
# Generate annotation sheets (run once, then annotate in Excel / Google Sheets)
if not USE_DEMO:
    du.make_requirement_sheet(jds, "data/annotations/requirements_to_label.csv")
    du.make_skill_sheet(jds, "data/annotations/jd_skills_to_label.csv", "JD", suggestions=se.DictionarySkillExtractor())
    du.make_skill_sheet(resumes, "data/annotations/resume_skills_to_label.csv", "Resume", suggestions=se.DictionarySkillExtractor())
    du.make_match_sheet(jds, resumes, "data/annotations/pairs_to_label.csv")
"""),
code("""
# Load labels
match_file = DATA / ("demo_match_labels.csv" if USE_DEMO else "annotations/pairs_labelled.csv")
req_file   = DATA / ("demo_requirement_labels.csv" if USE_DEMO else "annotations/requirements_labelled.csv")
skill_file = DATA / ("demo_skill_gold.csv" if USE_DEMO else "annotations/jd_skills_labelled.csv")

labels = pd.read_csv(match_file)
if "annotator" in labels and labels.annotator.nunique() > 1:          # inter-annotator agreement
    display(du.agreement_report(labels.assign(item=labels.jd_id + "|" + labels.resume_id), "item", "label"))
    labels = du.adjudicate(labels.assign(item=labels.jd_id + "|" + labels.resume_id), "item", "label")
    labels[["jd_id", "resume_id"]] = labels["item"].str.split("|", expand=True)
pairs = labels.assign(jd_text=labels.jd_id.map(jds), resume_text=labels.resume_id.map(resumes)).dropna(subset=["jd_text", "resume_text"])
pairs["label"] = pairs["label"].astype(int)

req_labels = pd.read_csv(req_file)
if "annotator" in req_labels and req_labels.annotator.nunique() > 1:
    display(du.agreement_report(req_labels, "sent_id", "gold_label"))
    req_labels = du.adjudicate(req_labels, "sent_id", "gold_label").merge(
        req_labels.drop_duplicates("sent_id")[["sent_id", "jd_id", "sentence", "section_header"]], on="sent_id")

skill_gold = du.load_skill_gold(skill_file)
stats = du.dataset_statistics(jds, resumes, pairs); display(stats); stats.to_csv("results/A_dataset_stats.csv", index=False)
"""),
code("""
# Optional: large external labelled set (needs internet + `datasets`)
if RUN_HEAVY:
    hf_pairs = du.load_hf_fit_pairs("train", n=1500)
    hf_test  = du.load_hf_fit_pairs("test",  n=600)
    print(hf_pairs.label.value_counts())
"""),

md("""
---
## 3. Member B — Skill extraction

| ID | Method |
|---|---|
| M1 | ESCO taxonomy extractor (`ojd_daps_skills`) — **Part I baseline** |
| M2 | M1 + post-processing (split enumerations, remove generic verb phrases, alias normalisation) |
| M3 | Curated dictionary (≈300 skills + aliases) with spaCy PhraseMatcher |
| M4 | M2 ∪ M3 |
| M5 | LLM few-shot extraction (JSON output) |
| M6 | JobBERT fine-tuned on SkillSpan (token classification) |
"""),
code("""
extractors = {"M3_dictionary": se.DictionarySkillExtractor()}
try:
    extractors["M1_esco"] = se.ESCOSkillExtractor()
    extractors["M2_esco_filtered"] = se.ESCOFilteredExtractor()
    extractors["M4_combined"] = se.CombinedExtractor(extractors["M3_dictionary"], extractors["M2_esco_filtered"])
except ImportError:
    print("ojd_daps_skills not installed -> M1/M2/M4 skipped")

# M5 LLM (set an API key in the environment first)
# from src.llm_clients import openai_llm, anthropic_llm, hf_local_llm
# extractors["M5_llm"] = se.LLMSkillExtractor(hf_local_llm())

# M6 fine-tuned NER (GPU, ~15 min):
# se.train_skill_ner("models/skill_ner")
# extractors["M6_jobbert_skillspan"] = se.TokenClassifierExtractor("models/skill_ner")

docs = {d: jds.get(d) or resumes.get(d) for d in skill_gold if (jds.get(d) or resumes.get(d))}
skill_summary, skill_per_doc = se.evaluate_extractors(extractors, docs, skill_gold)
display(skill_summary); skill_summary.to_csv("results/B_skill_extraction.csv", index=False)
plots.grouped_bars(skill_summary, ["precision_soft", "recall_soft", "f1_soft"], title="Skill extraction (soft match)");
"""),
code("""
# Error analysis for the best method: false positives / false negatives
best = skill_summary.sort_values("f1_soft").method.iloc[-1]
errs = se.extraction_errors(extractors[best], docs, skill_gold)
display(errs.groupby("type").skill.apply(lambda s: ", ".join(s.head(15))))
"""),

md("""
---
## 4. Member C — Requirement priority classification
Labels: MANDATORY merged into REQUIRED for evaluation (too few MANDATORY examples) → 3 classes.
Trainable models use **GroupKFold by JD** so that sentences from one JD never appear in train and test.
"""),
code("""
classifiers = {
    "C1_rules (Part I)":   lambda: rc.RuleClassifier(use_section_headers=False),
    "C2_rules+sections":   lambda: rc.RuleClassifier(use_section_headers=True),
    "C3_tfidf_lr":         lambda: rc.TfidfLRClassifier(),
}
if RUN_HEAVY:
    _zs = rc.ZeroShotNLIClassifier(labels=rc.LABELS_3, device=0)
    classifiers["C4_zero_shot_nli"] = lambda: _zs
    classifiers["C5_distilbert"] = lambda: rc.BertClassifier(epochs=6)

clf_summary, clf_preds = rc.evaluate_classifiers(classifiers, req_labels, n_splits=5)
display(clf_summary); clf_summary.to_csv("results/C_requirement_clf.csv", index=False)
plots.method_bar(clf_summary, "macro_f1", title="Requirement classification — macro-F1");
"""),
code("""
fig, axes = plt.subplots(1, len(classifiers), figsize=(4.5 * len(classifiers), 4))
for ax, m in zip(np.atleast_1d(axes), classifiers):
    rc.plot_confusion(clf_preds, m, ax=ax)
plt.tight_layout()
display(rc.TfidfLRClassifier().fit(req_labels.sentence.tolist(), req_labels.gold_label.str.replace("MANDATORY", "REQUIRED").tolist(),
                                   req_labels.section_header.tolist()).top_features(8))
"""),

md("""
---
## 5. Member D — Matching models
"""),
code("""
matchers = [mt.TfidfMatcher(), mt.BM25Matcher(),
            mt.EmbeddingMatcher(EMBEDDER, "doc"), mt.EmbeddingMatcher(EMBEDDER, "maxsim"),
            mt.PipelineMatcher(pipe, "D6_our_pipeline")]
if RUN_HEAVY:
    for name in ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2"]:
        matchers.append(mt.EmbeddingMatcher(name, "maxsim"))
    matchers.append(mt.CrossEncoderMatcher())

match_summary, match_scores = mt.evaluate_matchers(matchers, pairs, k=5)
display(match_summary); match_summary.to_csv("results/D_matching.csv", index=False)
match_scores.to_csv("results/D_pair_scores.csv", index=False)
plots.grouped_bars(match_summary, ["spearman", "macro_f1_cv", "ndcg@5"], title="JD–resume matching");
"""),
code("""
# D7: fine-tune a bi-encoder on the external labelled pairs, then test on OUR annotated pairs (domain transfer)
if RUN_HEAVY:
    path = mt.fine_tune_biencoder(hf_pairs, epochs=1)
    ft = mt.EmbeddingMatcher(path, "doc"); ft.name = "D7_finetuned_biencoder"
    s_ours, _ = mt.evaluate_matchers([ft, mt.EmbeddingMatcher(EMBEDDER, "doc")], pairs)
    s_hf, _   = mt.evaluate_matchers([ft, mt.EmbeddingMatcher(EMBEDDER, "doc")], hf_test)
    display(s_ours.assign(test_set="ours"), s_hf.assign(test_set="hf_test"))
"""),

md("""
---
## 6. Member E — Score calibration, learned scorer, ablation
"""),
code("""
results = sc.run_pipeline_on_pairs(pipe, pairs)
feat = sc.build_feature_table(results)
learned, coefs = sc.evaluate_learned_scorer(feat, n_splits=5)
display(learned); learned.to_csv("results/E_learned_scorer.csv", index=False)
display(coefs)   # which features drive "Good fit"?
"""),
code("""
train_pairs, test_pairs = du.split_by_jd(pairs, test_size=0.3)
dev_mask = pairs.jd_id.isin(train_pairs.jd_id).values
grid_summary, grid_log = sc.grid_search_scoring(results, dev_mask)
display(grid_summary); display(grid_log.head(10))
"""),
code("""
ablation = sc.run_ablation(pairs, PipelineConfig(embedder_name=EMBEDDER, skill_extractor="dictionary+esco"),
                           embedder=pipe.embedder, skill_extractor=pipe.skill_extractor)
display(ablation); ablation.to_csv("results/E_ablation.csv", index=False)
plots.method_bar(ablation, "spearman", label_col="variant", title="Ablation — Spearman ρ with human labels");
"""),

md("""
---
## 7. Member F — Fairness, robustness, error analysis, explanations
"""),
code("""
some_jd = list(jds)[0]
sample  = dict(list(resumes.items())[:5])
name_summary, name_detail = fa.counterfactual_name_test(pipe, jds[some_jd], sample)
display(name_summary); name_summary.to_csv("results/F_name_fairness.csv", index=False)
display(fa.gender_wording_test(pipe, jds[some_jd], sample))
"""),
code("""
rob_summary, rob_detail = fa.robustness_test(pipe, jds[some_jd], sample)
display(rob_summary); rob_summary.to_csv("results/F_robustness.csv", index=False)
"""),
code("""
worst = fa.worst_pairs(match_scores, "D6_our_pipeline", k=5)
display(worst)
fa.error_annotation_sheet(worst, "results/F_errors_to_categorise.csv")   # categorise manually, see fa.ERROR_CATEGORIES
display(fa.misclassified_sentences(clf_preds, "C2_rules+sections").head(15))
"""),
code("""
# Explanation for one candidate + ranking of all resumes for one JD
r = pipe.match(jds[some_jd], list(resumes.values())[0])
print(gap_report(r))
display(pipe.rank_resumes(jds[some_jd], resumes))
# from src.llm_clients import hf_local_llm; print(fa.llm_feedback(r, hf_local_llm()))
"""),
md("""
Demo UI: run `python app.py` in a terminal → upload a JD and a resume → score, requirement table, gap report.

---
## 8. Summary of results (fill in from `results/*.csv`)
| Module | Baseline (Part I) | Best method | Gain |
|---|---|---|---|
| Skill extraction (soft F1) | M1 = … | … | … |
| Requirement classification (macro-F1) | C1 = … | … | … |
| Matching (Spearman ρ) | D6 hand-crafted = … | … | … |
| Scoring (Spearman ρ) | E1 = … | … | … |
| Fairness (max abs Δ over names) | – | … | – |

**Limitations:** small annotated set, single annotator for part of the data, resumes mostly from one region/field,
LLM methods are not reproducible across API versions.
"""),
]

nb["cells"] = head + nb["cells"] + part2
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "Group32_NLPproject_extended.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", out, len(nb["cells"]), "cells")
