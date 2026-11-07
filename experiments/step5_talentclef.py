"""
STEP 5 (GPU preferred): TalentCLEF 2026 Task A + final pipeline (v3) everywhere.
  T1  TalentCLEF English development set (10 JDs x 472 CVs, expert qrels): all matchers, MAP / NDCG@10 / P@10
  T2  TREC run files for the TalentCLEF test set (40 JDs, qrels not public) -> can be submitted to Codabench
  T3  Final pipeline v3 (+ hybrid) on the 600 fit-benchmark pairs and the human-ranked benchmark
All tables are also printed to the console.
"""
import argparse
import io
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
from src import human_benchmark as hb, matching as mt, skill_extraction as se, talentclef as tc  # noqa: E402
from src.nlp_backend import HAS_ST, Embedder  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tc_root", default=str(ROOT / "talentclef"))
ap.add_argument("--bench", default=str(ROOT / "vacancy-resume-matching-dataset"))
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()

R = ROOT / "results_step5"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_step5.txt", "w", encoding="utf-8")


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
log(json.dumps({"gpu": GPU, "embedder": EMB}))
embedder = Embedder(EMB)
NER = ROOT / "models" / "skill_ner"
STATE = {}


def get_talentclef(root: Path):
    if (root / "TaskA").exists():
        return
    import requests
    root.mkdir(parents=True, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/19652670", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == "TaskA.zip"][0]
    zipfile.ZipFile(io.BytesIO(requests.get(f["links"]["self"], timeout=300).content)).extractall(root)


def pipe(extractor=None, **kw):
    return MatchingPipeline(PipelineConfig(embedder_name=EMB, **kw), embedder=embedder,
                            skill_extractor=extractor or se.DictionarySkillExtractor())


def matchers(include_neural=True):
    ms = [mt.TfidfMatcher(), mt.BM25Matcher(),
          mt.PipelineMatcher(pipe(), "D6_pipeline_v3"),
          mt.PipelineMatcher(pipe(doc_sim_weight=0.5), "D8_hybrid_v3 (0.5 x doc sim)")]
    if "jobbert" in STATE:
        ms.append(mt.PipelineMatcher(pipe(STATE["jobbert"]), "D6_pipeline_v3 + JobBERT"))
    if include_neural and HAS_ST and not args.demo:
        for name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2",
                     "BAAI/bge-small-en-v1.5"]:
            ms += [mt.EmbeddingMatcher(name, "doc"), mt.EmbeddingMatcher(name, "maxsim")]
        try:
            ms.append(mt.CrossEncoderMatcher())
        except Exception as e:
            log("cross-encoder failed:", e)
    return ms


@step("T0 data + JobBERT")
def _t0():
    get_talentclef(Path(args.tc_root))
    if GPU and not args.demo:
        if not (NER / "config.json").exists():
            log("SkillSpan:", se.train_skill_ner(str(NER), epochs=3))
        STATE["jobbert"] = se.JobBERTFilteredExtractor(str(NER), strict=True)


@step("T1 TalentCLEF development set (English)")
def _t1():
    Q, C, Rq = tc.load_taskA(args.tc_root, "development", "en")
    if args.demo:
        Q = dict(list(Q.items())[:2]); Rq = {q: Rq[q] for q in Q}
    rows = [tc.random_baseline(Rq, C)]
    runs = {}
    for m in matchers():
        t0 = time.time()
        try:
            runs[m.name] = tc.score_all(m, Q, C, verbose=False)
        except Exception as e:
            log(f"[matcher failed] {m.name}: {e!r}")
            continue
        rows.append({**tc.evaluate_run(runs[m.name], Rq, m.name), "seconds": round(time.time() - t0)})
        log(rows[-1])
    save(pd.DataFrame(rows).sort_values("map", ascending=False), "T1_talentclef_dev.csv")
    STATE["best_dev"] = sorted(rows[1:], key=lambda r: -r["map"])[0]["method"]


@step("T2 TalentCLEF test-set run files (for Codabench)")
def _t2():
    Q, C, _ = tc.load_taskA(args.tc_root, "test", "en")
    if args.demo:
        Q = dict(list(Q.items())[:1])
    keep = {"D1_tfidf", "D8_hybrid_v3 (0.5 x doc sim)", "D6_pipeline_v3", STATE.get("best_dev", "")}
    for m in matchers():
        if m.name not in keep:
            continue
        run = tc.score_all(m, Q, C, verbose=False)
        fn = R / f"T2_test_run_{m.name.split(' ')[0]}.trec"
        tc.write_trec_run(run, str(fn), tag=f"group32_{m.name.split(' ')[0]}")
        log("wrote", fn.name)


@step("T3 final pipeline on the fit benchmark and the human-ranked benchmark")
def _t3():
    H = ROOT / "data" / "hf"
    test = pd.read_csv(H / "test.csv.gz")
    k = 200 if not args.demo else 4
    ev = pd.concat([g.sample(min(len(g), k), random_state=0) for _, g in test.groupby("label")]).reset_index(drop=True)
    ms = [mt.PipelineMatcher(pipe(), "D6_pipeline_v3"),
          mt.PipelineMatcher(pipe(doc_sim_weight=0.5), "D8_hybrid_v3 (0.5 x doc sim)")]
    if "jobbert" in STATE:
        ms.append(mt.PipelineMatcher(pipe(STATE["jobbert"]), "D6_pipeline_v3 + JobBERT"))
    summ, _ = mt.evaluate_matchers(ms, ev, k=5)
    save(summ, "T3_fit_benchmark_v3.csv")
    bench = Path(args.bench)
    if not bench.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/NataliaVanetik/vacancy-resume-matching-dataset.git",
                        str(bench)], check=True)
    V, C, Rk = hb.load_benchmark(str(bench))
    ms = [mt.PipelineMatcher(pipe(), "D6_pipeline_v3"),
          mt.PipelineMatcher(pipe(doc_sim_weight=0.5), "D8_hybrid_v3 (0.5 x doc sim)")]
    ranks = hb.run_matchers(ms, V, C)
    save(pd.DataFrame([hb.evaluate_rank_matrix(r, Rk, n) for n, r in ranks.items()]), "T3_human_benchmark_v3.csv")


LOG.close()
zp = ROOT / "results_step5.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(R.glob("*")):
        z.write(f, f.name)
print(f"\nALL DONE -> {zp} ({zp.stat().st_size // 1024} KB). All outputs are packed into this zip.")
