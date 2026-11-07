"""
STEP 3 (GPU recommended): follow-up experiments that close the open issues of the first runs.
  F1  JobBERT inside the pipeline: dictionary vs JobBERT+dictionary on the 600 test pairs (coverage + metrics)
  F2  Robustness fixes at scale: 40 resumes, before vs after the date / name / caps fixes
  F3  Name fairness on 8 JDs x 4 resumes, before vs after the name filter
  F4  Human-annotated benchmark (Vanetik & Kogan 2023) with all neural matchers
Needs data/hf/*.csv.gz (experiments/step1_prepare.py). Output: results_step3.zip

  python experiments/step3_followup.py            # full
  python experiments/step3_followup.py --demo     # offline check (no GPU, tiny)
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import fairness as fa, human_benchmark as hb, matching as mt, skill_extraction as se  # noqa: E402
from src.nlp_backend import HAS_ST, Embedder  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--n_eval", type=int, default=600)
ap.add_argument("--bench", default=str(ROOT / "vacancy-resume-matching-dataset"))
args = ap.parse_args()

R = ROOT / "results_followup"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_followup.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def save(df, name):
    df.to_csv(R / name, index=False)
    log(f"\n### {name}\n{df.to_string(max_colwidth=60)}")


def step(title):
    def deco(fn):
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
EMB = "sentence-transformers/all-MiniLM-L6-v2" if HAS_ST else "tfidf-char"
log(json.dumps({"gpu": GPU, "embedder": EMB, "demo": args.demo}))


def balanced(df, n, seed=0):          # identical to step2 -> same 600 evaluation pairs
    if df.empty:
        return df
    k = max(1, n // df.label.nunique())
    return pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in df.groupby("label")]) \
        .reset_index(drop=True)


H = ROOT / "data" / "hf"
test, train = pd.read_csv(H / "test.csv.gz"), pd.read_csv(H / "train.csv.gz")
ev = balanced(test, args.n_eval)
log(f"eval pairs {len(ev)} ({ev.jd_id.nunique()} JDs)")

embedder = Embedder(EMB)
NER_DIR = ROOT / "models" / "skill_ner"
STATE = {}


@step("F0 train / load JobBERT skill model")
def _f0():
    if args.demo or not GPU:
        log("no GPU / demo: JobBERT skipped")
        return
    if not (NER_DIR / "config.json").exists():
        log("SkillSpan test metrics:", se.train_skill_ner(str(NER_DIR), epochs=3))
    STATE["jobbert"] = se.JobBERTFilteredExtractor(str(NER_DIR))


def pipelines():
    p = {"D6_dictionary (v2 fixes)": MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder,
                                                      skill_extractor=se.DictionarySkillExtractor())}
    if "jobbert" in STATE:
        p["D6_jobbert+dictionary"] = MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder,
                                                      skill_extractor=STATE["jobbert"])
    return p


@step("F1 JobBERT inside the pipeline (600 test pairs)")
def _f1():
    rows, matchers = [], []
    for name, p in pipelines().items():
        jd_texts = ev.drop_duplicates("jd_id").jd_text.tolist()
        n_req = [len(p.extract_jd_requirements(j)) for j in jd_texts]
        rows.append({"variant": name, "jds": len(jd_texts), "jds_with_no_requirement": int(sum(n == 0 for n in n_req)),
                     "mean_requirements_per_jd": round(float(np.mean(n_req)), 1)})
        matchers.append(mt.PipelineMatcher(p, name))
    save(pd.DataFrame(rows), "F1_requirement_coverage.csv")
    summ, scores = mt.evaluate_matchers(matchers, ev, k=5)
    save(summ, "F1_pipeline_matching.csv")
    scores.to_csv(R / "F1_pair_scores.csv", index=False)
    ex = [r for r in matchers[-1].results if r["final_score"] is not None][:3]
    for r in ex:
        log("example requirements:", r["jd_requirements"].requirement.head(15).tolist())


@step("F2 robustness fixes at scale (40 resumes)")
def _f2():
    sub = ev.drop_duplicates("resume_id").drop_duplicates("jd_id").head(40 if not args.demo else 6)
    before = MatchingPipeline(PipelineConfig(embedder_name=EMB, experience_fallback=False, filter_name_lines=False),
                              embedder=embedder, skill_extractor=se.DictionarySkillExtractor(caps_fix=False))
    after = MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder,
                             skill_extractor=se.DictionarySkillExtractor())
    rows = []
    for tag, p in (("before_fix", before), ("after_fix", after)):
        for _, r in sub.iterrows():
            s, d = fa.robustness_test(p, r.jd_text, {r.resume_id: r.resume_text})
            d["version"] = tag
            rows.append(d)
    d = pd.concat(rows)
    d["changed"] = d.delta.fillna(1).abs() > 0.01   # None (unscorable after perturbation) counts as changed
    summ = d.groupby(["version", "perturbation"]).agg(
        resumes=("resume_id", "nunique"), mean_delta=("delta", "mean"),
        max_abs_delta=("delta", lambda x: x.abs().max()), share_changed=("changed", "mean")).round(3).reset_index()
    save(summ, "F2_robustness_at_scale.csv")


@step("F3 name fairness on 8 JDs x 4 resumes")
def _f3():
    before = MatchingPipeline(PipelineConfig(embedder_name=EMB, filter_name_lines=False), embedder=embedder,
                              skill_extractor=se.DictionarySkillExtractor())
    after = MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder,
                             skill_extractor=se.DictionarySkillExtractor())
    jds = ev.jd_id.drop_duplicates().sample(min(8 if not args.demo else 2, ev.jd_id.nunique()), random_state=11)
    rows = []
    for jid in jds:
        g = ev[ev.jd_id == jid]
        res = dict(list(zip(g.resume_id, g.resume_text))[:4 if not args.demo else 2])
        for tag, p in (("before_fix", before), ("after_fix", after)):
            s, det = fa.counterfactual_name_test(p, g.jd_text.iloc[0], res)
            if det.score.isna().all():
                continue
            rows.append({"jd_id": jid, "version": tag, "resumes": len(res),
                         "max_abs_delta": round(float(det.delta.abs().max()), 3),
                         "resumes_affected": int((det.groupby("resume_id").delta.apply(lambda x: x.abs().max()) > 0.01).sum())})
    save(pd.DataFrame(rows), "F3_name_fairness_8jd.csv")


@step("F4 human-annotated benchmark with neural matchers")
def _f4():
    bench = Path(args.bench)
    if not bench.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/NataliaVanetik/vacancy-resume-matching-dataset.git",
                        str(bench)], check=True)
    V, C, Rk = hb.load_benchmark(str(bench))
    ms = [mt.TfidfMatcher(), mt.BM25Matcher()]
    for name, p in pipelines().items():
        ms.append(mt.PipelineMatcher(p, name))
    if HAS_ST and not args.demo:
        for name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
                     "BAAI/bge-small-en-v1.5"]:
            ms += [mt.EmbeddingMatcher(name, "doc"), mt.EmbeddingMatcher(name, "maxsim")]
        try:
            ms.append(mt.CrossEncoderMatcher())
        except Exception as e:
            log("cross-encoder failed:", e)
        if GPU:
            path = mt.fine_tune_biencoder(balanced(train, 3000), output_dir=str(ROOT / "models" / "biencoder_fit"))
            m = mt.EmbeddingMatcher(path, "doc")
            m.name = "D7_finetuned_biencoder (fit benchmark)"
            ms.append(m)
    ranks = hb.run_matchers(ms, V, C)
    save(pd.DataFrame([hb.evaluate_rank_matrix(r, Rk, n) for n, r in ranks.items()]), "F4_human_benchmark_all.csv")
    save(pd.DataFrame([hb.human_agreement(Rk), hb.random_baseline()]), "F4_human_agreement.csv")


LOG.close()
zp = ROOT / "results_step3.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(R.glob("*")):
        z.write(f, f.name)
print(f"\nALL DONE -> {zp} ({zp.stat().st_size // 1024} KB). All outputs are packed into this zip.")
