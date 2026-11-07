"""
Step 6 — TalentCLEF dev-set ablation + weight sensitivity with the REAL MiniLM embedder,
plus paired bootstrap significance tests over the 10 queries.

Why: the ablation in results/T_talentclef_ablation_local.csv was run without sentence-transformers
(fallback char-TF-IDF embedder). This re-runs it with all-MiniLM-L6-v2 and saves per-query AP so that
differences (e.g. hybrid 0.839 vs mpnet doc 0.869) can be tested.

Run (GPU optional; ~20-40 min on GPU, longer on CPU):
    python experiments/step6_ablation_minilm.py
Outputs: results_step6/  and  results_step6.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import matching as mt, skill_extraction as se, talentclef as tc  # noqa: E402
import numpy as np  # noqa: E402


class st:  # self-contained copy of src/stats.py helpers (older project copies lack that file)
    @staticmethod
    def per_query_ap(run, qrels):
        return pd.Series({q: tc._ap([c for c, _ in sorted(run[q].items(), key=lambda x: -x[1])], rel)
                          for q, rel in qrels.items()})

    @staticmethod
    def paired_bootstrap_diff(a, b, n=5000, seed=0):
        d = (a - b).dropna().values
        rng = np.random.default_rng(seed)
        boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean_diff": round(float(d.mean()), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                "p_not_better": round(float((boots <= 0).mean()), 4), "n_queries": len(d)}
from src.nlp_backend import HAS_ST, Embedder  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tc_root", default=str(ROOT / "talentclef"))
args = ap.parse_args()

R = ROOT / "results_step6"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_step6.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


if not HAS_ST:
    log("WARNING: sentence-transformers not installed -> run `!pip install sentence-transformers` first")
EMB = "sentence-transformers/all-MiniLM-L6-v2"
embedder = Embedder(EMB)
# Speed-up: memoise single-text encodes (JD / resume document vectors for the hybrid score, skill phrases).
# Encoding is deterministic, so scores are identical; without this each variant re-encodes 4,720 pairs (~17 min).
_enc, _memo = embedder.encode, {}


def _cached_encode(texts, *a, **k):
    if isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str):
        if texts[0] not in _memo:
            _memo[texts[0]] = _enc(texts, *a, **k)
        return _memo[texts[0]]
    return _enc(texts, *a, **k)


embedder.encode = _cached_encode
log(json.dumps({"embedder": embedder.name}))

if not (Path(args.tc_root) / "TaskA").exists():
    import io
    import requests
    Path(args.tc_root).mkdir(parents=True, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/19652670", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == "TaskA.zip"][0]
    zipfile.ZipFile(io.BytesIO(requests.get(f["links"]["self"], timeout=300).content)).extractall(args.tc_root)

Q, C, Rq = tc.load_taskA(args.tc_root, "development", "en")


def pipe(**kw):
    return mt.PipelineMatcher(
        MatchingPipeline(PipelineConfig(embedder_name=EMB, **kw), embedder=embedder,
                         skill_extractor=se.DictionarySkillExtractor()), "pipe")


FULL = dict(doc_sim_weight=0.5)
VARIANTS = [
    ("full: pipeline v3 + 0.5 doc sim", FULL),
    ("- JD skill lists", {**FULL, "use_skill_lists": False}),
    ("- exact match", {**FULL, "use_exact_match": False}),
    ("- semantic match", {**FULL, "use_semantic_match": False}),
    ("- priority weights", {**FULL, "use_priority_weights": False}),
    ("- education", {**FULL, "use_education": False}),
    ("- experience", {**FULL, "use_experience": False}),
    ("- section headings", {**FULL, "use_section_headers": False}),
    ("- doc similarity (pipeline v3 alone)", {"doc_sim_weight": 0.0}),
    ("weight 0.25", {"doc_sim_weight": 0.25}),
    ("weight 1.0", {"doc_sim_weight": 1.0}),
    ("weight 2.0", {"doc_sim_weight": 2.0}),
]
BASELINES = [
    ("doc similarity only (MiniLM doc)", lambda: mt.EmbeddingMatcher(EMB, "doc")),
    ("mpnet doc", lambda: mt.EmbeddingMatcher("sentence-transformers/all-mpnet-base-v2", "doc")),
    ("TF-IDF", lambda: mt.TfidfMatcher()),
    ("BM25", lambda: mt.BM25Matcher()),
]

rows, perq = [], {}
jobs = [(n, (lambda kw=kw: pipe(**kw))) for n, kw in VARIANTS] + BASELINES
for i, (name, make) in enumerate(jobs, 1):
    t0 = time.time()
    log(f"[{i}/{len(jobs)}] {name} ...")
    try:
        run = tc.score_all(make(), Q, C, verbose=False)
    except Exception:
        log(f"[FAILED] {name}\n{traceback.format_exc()}")
        continue
    rows.append({**tc.evaluate_run(run, Rq, name), "sec": round(time.time() - t0)})
    perq[name] = st.per_query_ap(run, Rq)
    log(rows[-1])

summary = pd.DataFrame(rows)
summary.to_csv(R / "T_talentclef_ablation_minilm.csv", index=False)
pd.DataFrame(perq).to_csv(R / "T_talentclef_per_query_ap_minilm.csv")
log("\n### ablation (MiniLM)\n" + summary.to_string())

full = "full: pipeline v3 + 0.5 doc sim"
tests = [(full, n) for n, _ in VARIANTS[1:]] + [
    ("mpnet doc", full), ("TF-IDF", full), (full, "doc similarity only (MiniLM doc)"),
    ("weight 2.0", full)]
sig = []
for a, b in tests:
    if a in perq and b in perq:
        sig.append({"a": a, "b": b, **st.paired_bootstrap_diff(perq[a], perq[b])})
sig = pd.DataFrame(sig)
sig.to_csv(R / "S_talentclef_paired_bootstrap_minilm.csv", index=False)
log("\n### paired bootstrap, MAP difference a - b over 10 queries\n" + sig.to_string())

LOG.close()
zp = ROOT / "results_step6.zip"
with zipfile.ZipFile(zp, "w") as z:
    for p in R.iterdir():
        z.write(p, p.name)
print("ALL DONE ->", zp)
