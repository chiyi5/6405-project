"""
STEP 4 (GPU preferred, CPU works but slower): finish the two open items.
  G1  JobBERT in the pipeline on the 600 test pairs: dictionary vs JobBERT (unfiltered) vs JobBERT (filtered)
  G2  Robustness fixes at scale: 40 resumes, before vs after
  G3  Human-annotated benchmark for the filtered JobBERT pipeline
All tables are also printed, so nothing is lost if the download fails.
Output: results_step4.zip
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

R = ROOT / "results_step4"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_step4.txt", "w", encoding="utf-8")


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


def balanced(df, n, seed=0):          # identical to step2/step3 -> same 600 evaluation pairs
    k = max(1, n // df.label.nunique())
    return pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in df.groupby("label")]) \
        .reset_index(drop=True)


ev = balanced(pd.read_csv(ROOT / "data" / "hf" / "test.csv.gz"), args.n_eval)
log(f"eval pairs {len(ev)} ({ev.jd_id.nunique()} JDs)")
embedder = Embedder(EMB)
NER = ROOT / "models" / "skill_ner"
STATE = {}


@step("G0 train / load JobBERT")
def _g0():
    if args.demo:
        log("demo: JobBERT skipped")
        return
    if not (NER / "config.json").exists():
        epochs = 3 if GPU else 1
        log(f"training JobBERT for {epochs} epoch(s) on {'GPU' if GPU else 'CPU'}")
        log("SkillSpan test metrics:", se.train_skill_ner(str(NER), epochs=epochs))
    STATE["jb_strict"] = se.JobBERTFilteredExtractor(str(NER), strict=True)
    STATE["jb_loose"] = se.JobBERTFilteredExtractor(str(NER), strict=False)


def make(extractor, **cfg):
    return MatchingPipeline(PipelineConfig(embedder_name=EMB, **cfg), embedder=embedder, skill_extractor=extractor)


def variants():
    v = {"D6_dictionary": make(se.DictionarySkillExtractor())}
    if "jb_loose" in STATE:
        v["D6_jobbert_unfiltered+dictionary"] = make(STATE["jb_loose"])
        v["D6_jobbert_filtered+dictionary"] = make(STATE["jb_strict"])
    return v


@step("G1 JobBERT in the pipeline (600 test pairs)")
def _g1():
    rows, ms = [], []
    jd_texts = ev.drop_duplicates("jd_id").jd_text.tolist()
    for name, p in variants().items():
        n_req = [len(p.extract_jd_requirements(j)) for j in jd_texts]
        rows.append({"variant": name, "jds": len(jd_texts), "jds_with_no_requirement": int(sum(n == 0 for n in n_req)),
                     "mean_requirements_per_jd": round(float(np.mean(n_req)), 1)})
        ms.append(mt.PipelineMatcher(p, name))
    save(pd.DataFrame(rows), "G1_requirement_coverage.csv")
    summ, scores = mt.evaluate_matchers(ms, ev, k=5)
    save(summ, "G1_pipeline_matching.csv")
    scores.to_csv(R / "G1_pair_scores.csv", index=False)
    if len(ms) > 2:
        jd = jd_texts[0]
        for m in ms:
            log(f"{m.name} requirements of JD 1:", m.pipe.extract_jd_requirements(jd).requirement.tolist()[:25])


@step("G2 robustness fixes at scale (40 resumes)")
def _g2():
    sub = ev.drop_duplicates("resume_id").drop_duplicates("jd_id")
    sub = pd.concat([sub, ev[~ev.resume_id.isin(sub.resume_id)].drop_duplicates("resume_id")]).head(
        40 if not args.demo else 6)
    before = make(se.DictionarySkillExtractor(caps_fix=False), experience_fallback=False, filter_name_lines=False)
    after = make(se.DictionarySkillExtractor())
    parts = []
    for tag, p in (("before_fix", before), ("after_fix", after)):
        for _, r in sub.iterrows():
            _, d = fa.robustness_test(p, r.jd_text, {r.resume_id: r.resume_text})
            d["version"] = tag
            parts.append(d)
    d = pd.concat(parts)
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d["changed"] = d.delta.isna() | (d.delta.abs() > 0.01)   # unscorable after perturbation counts as changed
    d.to_csv(R / "G2_robustness_detail.csv", index=False)
    summ = d.groupby(["version", "perturbation"]).agg(
        resumes=("resume_id", "nunique"), mean_delta=("delta", "mean"),
        max_abs_delta=("delta", lambda x: x.abs().max()), share_changed=("changed", "mean")).round(3).reset_index()
    save(summ, "G2_robustness_at_scale.csv")


@step("G3 human benchmark for the JobBERT variants")
def _g3():
    bench = Path(args.bench)
    if not bench.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/NataliaVanetik/vacancy-resume-matching-dataset.git",
                        str(bench)], check=True)
    V, C, Rk = hb.load_benchmark(str(bench))
    ms = [mt.PipelineMatcher(p, n) for n, p in variants().items()]
    ranks = hb.run_matchers(ms, V, C)
    save(pd.DataFrame([hb.evaluate_rank_matrix(r, Rk, n) for n, r in ranks.items()]), "G3_human_benchmark_jobbert.csv")


LOG.close()
zp = ROOT / "results_step4.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(R.glob("*")):
        z.write(f, f.name)
print(f"\nALL DONE -> {zp} ({zp.stat().st_size // 1024} KB). All outputs are packed into this zip.")
