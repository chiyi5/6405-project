"""
STEP 2 (GPU recommended): run every experiment of modules A–F, save tables + figures.
Needs data/annotations/requirements_labelled.csv (requirement labels; LLM-annotated, see README).
Output: results_step2.zip

  python experiments/step2_run_all.py                 # full run
  python experiments/step2_run_all.py --demo --light  # offline check (no transformers)
Each experiment is wrapped in try/except: one failure never stops the others (see results/run_log.txt).
"""
import argparse
import json
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_utils as du, fairness as fa, matching as mt, plots, requirement_clf as rc  # noqa: E402
from src import scoring as sc, skill_extraction as se  # noqa: E402
from src.nlp_backend import HAS_ST, Embedder  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig, gap_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--light", action="store_true", help="skip transformer experiments")
ap.add_argument("--n_eval", type=int, default=600, help="balanced test pairs for D/E/F")
ap.add_argument("--n_train", type=int, default=3000, help="train pairs for fine-tuning")
ap.add_argument("--n_ablation", type=int, default=300)
ap.add_argument("--run_llm", action="store_true", help="M5: local Qwen LLM skill extraction (slow)")
ap.add_argument("--only", default="", help="comma list of steps to run: A,B,C,D,D7,E,E2,F,F2 (default all)")
args = ap.parse_args()

R = ROOT / "results"
R.mkdir(exist_ok=True)
ONLY = {x.strip().upper() for x in args.only.split(",") if x.strip()}
LOG = open(R / ("run_log.txt" if not ONLY else f"run_log_{'_'.join(sorted(ONLY))}.txt"), "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def save(df, name):
    df.to_csv(R / name, index=False)
    log(f"\n### {name}\n{df.to_string(max_colwidth=60)}")


def fig(name):
    plt.savefig(R / name, dpi=160, bbox_inches="tight")
    plt.close("all")


def step(title):
    key = title.split()[0].upper()
    if title.startswith("E ablation"):
        key = "E2"
    elif title.startswith("F error"):
        key = "F2"

    def deco(fn):
        if ONLY and key not in ONLY:
            log(f"[skip] {title} (not in --only)")
            return fn
        t0 = time.time()
        log(f"\n================ {title} ================")
        try:
            fn()
            log(f"[ok] {title} ({time.time() - t0:.0f}s)")
        except Exception:
            log(f"[FAILED] {title}\n{traceback.format_exc()}")
        return fn
    return deco


try:
    import torch
    GPU = torch.cuda.is_available()
except Exception:
    GPU = False
HEAVY = (not args.light) and HAS_ST
EMB = "sentence-transformers/all-MiniLM-L6-v2" if HAS_ST else "tfidf-char"
env = {"python": platform.python_version(), "gpu": GPU, "heavy": HEAVY, "embedder": EMB, "demo": args.demo,
       "args": vars(args)}
try:
    import sentence_transformers, transformers  # noqa
    env.update(transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__)
except Exception:
    pass
log(json.dumps(env, indent=1))

# ------------------------------------------------------------------ data
H = ROOT / "data" / "hf"
train = pd.read_csv(H / "train.csv.gz")
test = pd.read_csv(H / "test.csv.gz")


def balanced(df, n, seed=0):
    if df.empty:
        return df
    k = max(1, n // df.label.nunique())
    return pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in df.groupby("label")]) \
        .reset_index(drop=True)


ev = balanced(test, args.n_eval)
tr = balanced(train, args.n_train)
log(f"eval pairs {len(ev)} ({ev.jd_id.nunique()} JDs), train pairs {len(tr)}")
save(ev.label.value_counts().sort_index().rename_axis("label").reset_index(name="n"), "A_eval_label_dist.csv")

lab_path = ROOT / "data" / "annotations" / "requirements_labelled.csv"
if not lab_path.exists() and args.demo:
    lab_path = ROOT / "data" / "demo" / "demo_requirement_labels.csv"
req = pd.read_csv(lab_path)
req = req[req.gold_label.notna() & (req.gold_label.astype(str).str.strip() != "")]
save(req.gold_label.value_counts().rename_axis("label").reset_index(name="n"), "A_requirement_label_dist.csv")

embedder = Embedder(EMB)
skill_ex = se.build_skill_extractor("dictionary+esco")
pipe = MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder, skill_extractor=skill_ex)
STATE = {}


# ------------------------------------------------------------------ B skill extraction
@step("B skill extraction")
def _b():
    ssj = H / "skillspan_test.json"
    if ssj.exists():
        d = json.loads(ssj.read_text(encoding="utf-8"))
        docs, gold = d["docs"], d["gold"]
        keys = list(docs)[:150]
        docs, gold = {k: docs[k] for k in keys}, {k: gold[k] for k in keys}
        src = "SkillSpan test (human gold)"
    else:
        docs = du.load_folder(ROOT / "data" / "demo" / "jds")
        gold = du.load_skill_gold(ROOT / "data" / "demo" / "demo_skill_gold.csv")
        src = "demo"
    log("gold source:", src, "| docs:", len(docs))
    ex = {"M3_dictionary": se.DictionarySkillExtractor()}
    try:
        ex["M1_esco"] = se.ESCOSkillExtractor()
        ex["M2_esco_filtered"] = se.ESCOFilteredExtractor()
        ex["M4_combined"] = se.CombinedExtractor(ex["M3_dictionary"], ex["M2_esco_filtered"])
    except Exception as e:
        log("ESCO unavailable:", repr(e)[:200])
    if HEAVY and GPU:
        try:
            if not (ROOT / "models" / "skill_ner" / "config.json").exists():
                out = se.train_skill_ner(str(ROOT / "models" / "skill_ner"), epochs=3)
                log("SkillSpan NER test metrics:", out)
            else:
                log("re-using trained models/skill_ner")
            ex["M6_jobbert_skillspan"] = se.TokenClassifierExtractor(str(ROOT / "models" / "skill_ner"))
        except Exception:
            log("M6 failed:\n", traceback.format_exc())
    if args.run_llm and GPU:
        try:
            from src.llm_clients import hf_local_llm
            ex["M5_llm_qwen1.5b"] = se.LLMSkillExtractor(hf_local_llm("Qwen/Qwen2.5-1.5B-Instruct"))
        except Exception:
            log("M5 failed:\n", traceback.format_exc())
    summ, per = se.evaluate_extractors(ex, docs, gold)
    summ.insert(0, "gold_source", src)
    save(summ, "B_skill_extraction.csv")
    per.to_csv(R / "B_skill_per_doc.csv", index=False)
    plots.grouped_bars(summ, ["precision_soft", "recall_soft", "f1_soft"], title="Skill extraction (soft match)")
    fig("B_skill_extraction.png")
    best = summ.sort_values("f1_soft").method.iloc[-1]
    errs = se.extraction_errors(ex[best], docs, gold)
    errs.to_csv(R / "B_errors_best_method.csv", index=False)
    log("most frequent FP:", errs[errs.type == "FP"].skill.str.lower().value_counts().head(15).to_dict())
    log("most frequent FN:", errs[errs.type == "FN"].skill.str.lower().value_counts().head(15).to_dict())


# ------------------------------------------------------------------ C requirement classification
@step("C requirement classification")
def _c():
    clfs = {"C1_rules (Part I)": lambda: rc.RuleClassifier(use_section_headers=False),
            "C2_rules+sections": lambda: rc.RuleClassifier(use_section_headers=True),
            "C3_tfidf_lr": lambda: rc.TfidfLRClassifier(),
            "C3b_tfidf_lr_no_header": lambda: rc.TfidfLRClassifier(use_headers=False)}
    if HEAVY:
        zs = rc.ZeroShotNLIClassifier(labels=rc.LABELS_3, device=0 if GPU else -1)
        clfs["C4_zero_shot_nli"] = lambda: zs
        if GPU:
            clfs["C5_distilbert"] = lambda: rc.BertClassifier(epochs=6, output_dir=str(ROOT / "models" / "req_clf"))
    summ, preds = rc.evaluate_classifiers(clfs, req, n_splits=5)
    save(summ, "C_requirement_clf.csv")
    preds.to_csv(R / "C_predictions.csv", index=False)
    plots.method_bar(summ, "macro_f1", title="Requirement classification - macro-F1")
    fig("C_macro_f1.png")
    n = len(clfs)
    _, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
    for ax, m in zip(np.atleast_1d(axes), clfs):
        rc.plot_confusion(preds, m, ax=ax)
    plt.tight_layout()
    fig("C_confusion.png")
    from sklearn.metrics import classification_report
    best = summ.sort_values("macro_f1").method.iloc[-1]
    log("per-class report (best =", best, ")\n", classification_report(preds.gold, preds[best], zero_division=0))
    tf = rc.TfidfLRClassifier().fit(req.sentence.tolist(), req.gold_label.str.replace("MANDATORY", "REQUIRED").tolist(),
                                    req.section_header.where(req.section_header.notna(), None).tolist())
    save(tf.top_features(10), "C_top_features.csv")
    fa.misclassified_sentences(preds, best).head(40).to_csv(R / "C_errors_best.csv", index=False)


# ------------------------------------------------------------------ D matching
@step("D matching models")
def _d():
    ms = [mt.TfidfMatcher(), mt.BM25Matcher(),
          mt.EmbeddingMatcher(EMB, "doc", embedder=embedder), mt.EmbeddingMatcher(EMB, "maxsim", embedder=embedder),
          mt.PipelineMatcher(pipe, "D6_our_pipeline")]
    ms[2].name, ms[3].name = f"D3_{EMB.split('/')[-1]}_doc", f"D4_{EMB.split('/')[-1]}_maxsim"
    if HEAVY:
        for name in ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]:
            try:
                ms.append(mt.EmbeddingMatcher(name, "maxsim"))
            except Exception:
                log("embedder failed", name, traceback.format_exc()[-300:])
        try:
            ms.append(mt.CrossEncoderMatcher())
        except Exception:
            log("cross-encoder failed", traceback.format_exc()[-300:])
    summ, scores = mt.evaluate_matchers(ms, ev, k=5)
    STATE["match_scores"] = scores
    pm = [m for m in ms if isinstance(m, mt.PipelineMatcher)][0]
    if len(pm.results) == len(ev):
        for res, (_, r) in zip(pm.results, ev.iterrows()):
            res.update({"jd_id": r.jd_id, "resume_id": r.resume_id, "label": r.label})
        STATE["pipe_results"] = pm.results
    save(summ, "D_matching.csv")
    scores.to_csv(R / "D_pair_scores.csv", index=False)
    plots.grouped_bars(summ, ["spearman", "auc_goodfit", "macro_f1_cv"], title="JD-resume matching (test pairs)")
    fig("D_matching.png")


@step("D7 fine-tuned bi-encoder")
def _d7():
    if not (HEAVY and GPU):
        log("skipped (needs GPU)")
        return
    path = mt.fine_tune_biencoder(tr, output_dir=str(ROOT / "models" / "biencoder_fit"), epochs=1)
    ft = mt.EmbeddingMatcher(path, "doc")
    ft.name = "D7_finetuned_minilm_doc"
    base = mt.EmbeddingMatcher(EMB, "doc", embedder=embedder)
    base.name = "D3_minilm_doc (before FT)"
    summ, _ = mt.evaluate_matchers([base, ft], ev, k=5)
    save(summ, "D7_finetune.csv")


# ------------------------------------------------------------------ E scoring
@step("E learned scorer + calibration")
def _e():
    results = STATE.get("pipe_results") or sc.run_pipeline_on_pairs(pipe, ev)
    STATE["results"] = results
    feat = sc.build_feature_table(results)
    feat.to_csv(R / "E_features.csv", index=False)
    summ, coefs = sc.evaluate_learned_scorer(feat, n_splits=5)
    save(summ, "E_learned_scorer.csv")
    if coefs is not None:
        save(coefs.reset_index().rename(columns={"index": "feature"}), "E_logreg_coefficients.csv")
    tr_ids, _ = du.split_by_jd(ev, 0.3)
    dev = ev.jd_id.isin(tr_ids.jd_id).values
    gs, glog = sc.grid_search_scoring(results, dev)
    save(gs.astype(str), "E_grid_search.csv")
    glog.head(20).to_csv(R / "E_grid_log_top20.csv", index=False)


@step("E ablation")
def _e2():
    sub = balanced(ev, args.n_ablation, seed=1)
    abl = sc.run_ablation(sub, PipelineConfig(embedder_name=EMB), embedder=embedder, skill_extractor=skill_ex)
    save(abl, "E_ablation.csv")
    plots.method_bar(abl, "spearman", label_col="variant", title="Ablation - Spearman with gold labels")
    fig("E_ablation.png")


# ------------------------------------------------------------------ F fairness / robustness / errors
@step("F fairness + robustness")
def _f():
    jd_row = ev[ev.label == 2].iloc[0] if (ev.label == 2).any() else ev.iloc[0]
    sub = ev[ev.jd_id == jd_row.jd_id]
    resumes = dict(zip(sub.resume_id, sub.resume_text))
    if len(resumes) < 5:
        extra = ev.drop_duplicates("resume_id").sample(min(8, ev.resume_id.nunique()), random_state=0)
        resumes.update(dict(zip(extra.resume_id, extra.resume_text)))
    resumes = dict(list(resumes.items())[:8])
    before = MatchingPipeline(PipelineConfig(embedder_name=EMB, filter_name_lines=False), embedder=embedder,
                              skill_extractor=se.DictionarySkillExtractor(caps_fix=False))
    for tag, p in (("before_fix", before), ("after_fix", pipe)):
        s, detail = fa.counterfactual_name_test(p, jd_row.jd_text, resumes)
        save(s, f"F_name_fairness_{tag}.csv")
        detail.to_csv(R / f"F_name_fairness_detail_{tag}.csv", index=False)
        rs, _ = fa.robustness_test(p, jd_row.jd_text, resumes)
        save(rs, f"F_robustness_{tag}.csv")
    save(fa.gender_wording_test(pipe, jd_row.jd_text, resumes), "F_gender_wording.csv")
    # same test on 5 more JDs for a more reliable estimate
    rows = []
    for jid in ev.jd_id.drop_duplicates().sample(min(5, ev.jd_id.nunique()), random_state=3):
        sub2 = ev[ev.jd_id == jid]
        r2 = dict(list(zip(sub2.resume_id, sub2.resume_text))[:5])
        for tag, p in (("before_fix", before), ("after_fix", pipe)):
            s2, _ = fa.counterfactual_name_test(p, sub2.jd_text.iloc[0], r2)
            rows.append({"jd_id": jid, "version": tag, "max_abs_delta": s2.max_abs_delta.max(),
                         "mean_abs_group_delta": s2.mean_delta.abs().mean()})
    save(pd.DataFrame(rows), "F_name_fairness_multi_jd.csv")


@step("F error analysis + example report")
def _f2():
    if "match_scores" in STATE:
        w = fa.worst_pairs(STATE["match_scores"], "D6_our_pipeline", k=10)
        w = w.merge(ev[["jd_id", "resume_id", "jd_text", "resume_text"]], on=["jd_id", "resume_id"], how="left")
        w["jd_text"] = w.jd_text.str[:1200]
        w["resume_text"] = w.resume_text.str[:1200]
        w.to_csv(R / "F_worst_pairs.csv", index=False)
    if "results" in STATE:
        r = max(STATE["results"], key=lambda x: (x["label"] == 2, x["final_score"] or 0))
        (R / "F_example_gap_report.txt").write_text(gap_report(r), encoding="utf-8")
        r["details"].to_csv(R / "F_example_details.csv", index=False)
        ranking = pipe.rank_resumes(ev[ev.jd_id == r["jd_id"]].jd_text.iloc[0],
                                    dict(zip(ev[ev.jd_id == r["jd_id"]].resume_id, ev[ev.jd_id == r["jd_id"]].resume_text)))
        ranking.merge(ev[ev.jd_id == r["jd_id"]][["resume_id", "label"]], on="resume_id").to_csv(
            R / "F_example_ranking.csv", index=False)


# ------------------------------------------------------------------ package
LOG.close()
zp = ROOT / ("results_step2.zip" if not ONLY else "results_step2_rerun.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(R.glob("*")):
        if f.stat().st_size < 15_000_000:
            z.write(f, f.name)
print(f"\nALL DONE -> {zp} ({zp.stat().st_size // 1024} KB). All outputs are packed into this zip.")
