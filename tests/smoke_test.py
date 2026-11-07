"""Run every module end-to-end on the synthetic demo data (works without GPU / heavy models)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import MatchingPipeline, PipelineConfig, clean_text, gap_report, split_resume_sections  # noqa
from src import data_utils as du, skill_extraction as se, requirement_clf as rc, matching as mt, scoring as sc, fairness as fa  # noqa

EMB = sys.argv[1] if len(sys.argv) > 1 else "tfidf-char"
D = ROOT / "data" / "demo"
jds = du.load_folder(D / "jds")
resumes = du.load_folder(D / "resumes")
labels = pd.read_csv(D / "demo_match_labels.csv")
pairs = labels.assign(jd_text=labels.jd_id.map(jds), resume_text=labels.resume_id.map(resumes))

print("\n=== bug fixes ===")
print(clean_text("Pandas, Num\nPy, Scikit-learn"))
print(split_resume_sections("Projects\n8619 6329 a@b.com www.linkedin.com/in/x\nBuilt a thing")["projects"])

print("\n=== Pipeline on original AISG JD ===")
pipe = MatchingPipeline(PipelineConfig(embedder_name=EMB, skill_extractor="dictionary"))
aisg = (D / "JD_example1_ntu_AI.txt").read_text()
jd_df = pipe.extract_jd_requirements(aisg)
print(jd_df[["category", "requirement", "priority"]].to_string(max_colwidth=40))
res = pipe.match(aisg, resumes["R05_ml_phd"])
print(gap_report(res))

print("\n=== A: stats + agreement ===")
print(du.dataset_statistics(jds, resumes, pairs))
rq = pd.read_csv(D / "demo_requirement_labels.csv")
rng = np.random.default_rng(0)
long = []
for ann in ["ann1", "ann2", "ann3"]:
    lab = rq.gold_label.where(rng.random(len(rq)) > 0.15, "UNSPECIFIED")
    long.append(rq.assign(annotator=ann, gold_label=lab, item=rq.index))
long = pd.concat(long)
print(du.agreement_report(long, item_col="item"))
du.make_requirement_sheet(jds, "/tmp/req_sheet.csv")

print("\n=== B: skill extraction ===")
gold = du.load_skill_gold(D / "demo_skill_gold.csv")
summ, _ = se.evaluate_extractors({"M3_dictionary": se.DictionarySkillExtractor()}, jds, gold)
print(summ)
print(se.extraction_errors(se.DictionarySkillExtractor(), jds, gold).head(8))
print(se.postprocess_spans(["Build and maintain scalable data pipelines for multilingual text ingestion, cleaning",
                            "national initiative driven", "Linux, Docker and CI/CD", "Num", "Py"]))

print("\n=== C: requirement classification ===")
summ, preds = rc.evaluate_classifiers({
    "C1_rules": lambda: rc.RuleClassifier(use_section_headers=False),
    "C2_rules+sections": lambda: rc.RuleClassifier(use_section_headers=True),
    "C3_tfidf_lr": lambda: rc.TfidfLRClassifier(),
}, rq, n_splits=3)
print(summ)
rc.plot_confusion(preds, "C2_rules+sections")
print(rc.TfidfLRClassifier().fit(rq.sentence.tolist(), rq.gold_label.tolist(), rq.section_header.tolist()).top_features(3))

print("\n=== D: matching ===")
matchers = [mt.TfidfMatcher(), mt.BM25Matcher(), mt.EmbeddingMatcher(EMB, "doc"), mt.EmbeddingMatcher(EMB, "maxsim"),
            mt.PipelineMatcher(pipe)]
summ, scores = mt.evaluate_matchers(matchers, pairs, k=3)
print(summ)

print("\n=== E: scoring ===")
results = sc.run_pipeline_on_pairs(pipe, pairs, verbose=False)
feat = sc.build_feature_table(results)
s, coefs = sc.evaluate_learned_scorer(feat, n_splits=3)
print(s)
dev = pairs.jd_id.isin(["JD_data_analyst", "JD_backend_engineer"]).values
gs, log = sc.grid_search_scoring(results, dev)
print(gs[["setting", "test_spearman"]])
print(sc.run_ablation(pairs, PipelineConfig(embedder_name=EMB, skill_extractor="dictionary"),
                      embedder=pipe.embedder, skill_extractor=pipe.skill_extractor))

print("\n=== F: fairness / robustness / errors ===")
sub = {k: resumes[k] for k in ["R01_analyst_strong", "R08_data_scientist"]}
fs, _ = fa.counterfactual_name_test(pipe, jds["JD_data_analyst"], sub)
print(fs)
print(fa.gender_wording_test(pipe, jds["JD_data_analyst"], sub))
rs, _ = fa.robustness_test(pipe, jds["JD_data_analyst"], sub)
print(rs)
print(fa.worst_pairs(scores, "D6_our_pipeline", k=2)[["jd_id", "resume_id", "label", "error"]])
print("\nRanking demo:")
print(pipe.rank_resumes(jds["JD_ml_engineer"], resumes))
print("\nSMOKE TEST PASSED")
