"""
Step 7 — Fairness and robustness re-tested on the FINAL system (pipeline v3 and the hybrid score),
on more job descriptions than before.

  N  name test: 24 names (Chinese / Malay / Indian / Western x M / F) prepended to each resume
  G  gendered-activity test (Men's Football Team vs Women in Tech Society)
  P  robustness: upper case, bullet style, abbreviations, expansions, section order
Data: TalentCLEF dev (10 JDs x 4 relevant resumes) + fit benchmark test split (20 JDs x 2 resumes).
Systems: D6 pipeline v3 (dictionary) and D8 hybrid v3 (0.5 x document similarity, the demo's score).

Needs the updated src/pipeline.py (document similarity ignores the name line).
Run after step 6:   python experiments/step7_fairness_v3.py      (~20-40 min on GPU)
Outputs: results_step7/ and results_step7.zip
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import fairness as fa, skill_extraction as se, talentclef as tc  # noqa: E402
from src.nlp_backend import HAS_ST, Embedder  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig  # noqa: E402

R = ROOT / "results_step7"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_step7.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


EMB = "sentence-transformers/all-MiniLM-L6-v2" if HAS_ST else "tfidf-char"
embedder = Embedder(EMB)
_enc, _memo = embedder.encode, {}


def _cached_encode(texts, *a, **k):          # same speed-up as step 6; scores unchanged
    if isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str):
        if texts[0] not in _memo:
            _memo[texts[0]] = _enc(texts, *a, **k)
        return _memo[texts[0]]
    return _enc(texts, *a, **k)


embedder.encode = _cached_encode
log(json.dumps({"embedder": embedder.name}))

import inspect  # noqa: E402
from src import pipeline as _pl  # noqa: E402
if "doc_text" not in inspect.getsource(_pl):
    log("WARNING: src/pipeline.py is the old version (no name filter for document similarity). Upload the new one.")

SYSTEMS = {
    "D6_pipeline_v3": ("final_score", MatchingPipeline(PipelineConfig(embedder_name=EMB), embedder=embedder,
                                                       skill_extractor=se.DictionarySkillExtractor())),
    "D8_hybrid_v3": ("hybrid_score", MatchingPipeline(PipelineConfig(embedder_name=EMB, doc_sim_weight=0.5),
                                                      embedder=embedder, skill_extractor=se.DictionarySkillExtractor())),
}

# ---------------- data: (dataset, jd_id, jd_text, resume_id, resume_text) ----------------
cases = []
tcroot = ROOT / "talentclef"
if not (tcroot / "TaskA").exists():                 # fresh runtime: download TalentCLEF Task A (Zenodo, CC BY 4.0)
    import io
    import requests
    tcroot.mkdir(parents=True, exist_ok=True)
    rec = requests.get("https://zenodo.org/api/records/19652670", timeout=60).json()
    f = [x for x in rec["files"] if x["key"] == "TaskA.zip"][0]
    zipfile.ZipFile(io.BytesIO(requests.get(f["links"]["self"], timeout=300).content)).extractall(tcroot)
if (tcroot / "TaskA").exists():
    Q, C, Rq = tc.load_taskA(str(tcroot), "development", "en")
    for q in Q:
        for c in sorted(Rq[q])[:4]:
            cases.append(("talentclef", q, Q[q], c, C[c]))
hf = ROOT / "data" / "hf" / "test.csv.gz"
if hf.exists():
    t = pd.read_csv(hf)
    for jd_id in t.jd_id.drop_duplicates().sample(20, random_state=7):
        g = t[t.jd_id == jd_id].drop_duplicates("resume_id").head(2)
        for _, r in g.iterrows():
            cases.append(("fit_benchmark", jd_id, r.jd_text, r.resume_id, r.resume_text))
log(f"{len(cases)} JD-resume cases, {len({(d, j) for d, j, *_ in cases})} JDs")

import re  # noqa: E402

_HEAD = re.compile(r"(?im)^(?:professional |work |relevant )?(?:summary|profile|objective|experience|employment|"
                   r"education|skills|technical skills|core competencies|projects|certifications?|languages|"
                   r"awards|publications|volunteer(?:ing)?|interests|additional information|other information)"
                   r"\s*:?\s*$")


def shuffle_sections(t: str) -> str:
    """Reverse the order of whole sections (heading + its content); the header block stays first.
    (fairness.PERTURBATIONS reverses blank-line paragraphs, which separates a heading from its content
    when the heading is followed by a blank line, as in TalentCLEF resumes.)"""
    starts = [m.start() for m in _HEAD.finditer(t)]
    if len(starts) < 2:
        return t
    head, parts = t[:starts[0]], [t[a:b].strip("\n") for a, b in zip(starts, starts[1:] + [len(t)])]
    return head + "\n\n".join(reversed(parts))


PERTURB = dict(fa.PERTURBATIONS)
PERTURB.pop("drop_skills_section")          # intended to change the score; not a robustness test
PERTURB["shuffle_sections"] = shuffle_sections

NAMES = [(eth, g, n) for (eth, g), ns in fa.NAME_GROUPS.items() for n in ns]
rows_n, rows_g, rows_p = [], [], []
t0 = time.time()
for sysname, (key, pipe) in SYSTEMS.items():
    jd_cache = {}
    for i, (ds, jd_id, jd, rid, res) in enumerate(cases, 1):
        if jd_id not in jd_cache:
            jd_cache[jd_id] = pipe.extract_jd_requirements(jd)
        jd_df = jd_cache[jd_id]

        def score(text):
            return pipe.match(jd, text, jd_df=jd_df).get(key)

        base = score(res)
        for eth, g, nm in NAMES:
            rows_n.append(dict(system=sysname, dataset=ds, jd_id=jd_id, resume_id=rid, ethnicity=eth, gender=g,
                               name=nm, score=score(f"{nm}\n{res}")))
        for g in ("N", "M", "F"):
            rows_g.append(dict(system=sysname, dataset=ds, jd_id=jd_id, resume_id=rid, variant=g,
                               score=score(fa.add_gendered_activity(res, g))))
        for pname, fn in PERTURB.items():
            s = score(fn(res))
            rows_p.append(dict(system=sysname, dataset=ds, jd_id=jd_id, resume_id=rid, perturbation=pname,
                               original=base, perturbed=s))
        if i % 10 == 0:
            log(f"  {sysname}: {i}/{len(cases)} cases ({time.time() - t0:.0f}s)")

N, G, P = pd.DataFrame(rows_n), pd.DataFrame(rows_g), pd.DataFrame(rows_p)
for df in (N, G, P):
    df.to_csv(R / f"step7_{'names' if df is N else 'gender' if df is G else 'robustness'}_detail.csv", index=False)

# name test: per resume, spread of scores across the 24 names
N["score"] = pd.to_numeric(N.score, errors="coerce")
spread = N.groupby(["system", "dataset", "jd_id", "resume_id"]).score.agg(lambda x: x.max() - x.min()).rename(
    "name_range").reset_index()
s_n = spread.groupby("system").agg(resumes=("resume_id", "size"), jds=("jd_id", "nunique"),
                                   resumes_changed=("name_range", lambda x: int((x > 0.01).sum())),
                                   max_range=("name_range", "max"), mean_range=("name_range", "mean")).round(3)
N["delta"] = N.score - N.groupby(["system", "jd_id", "resume_id"]).score.transform("mean")
s_eth = N.groupby(["system", "ethnicity", "gender"]).delta.agg(["mean", lambda x: x.abs().max()]).round(3)
s_eth.columns = ["mean_delta", "max_abs_delta"]

G["score"] = pd.to_numeric(G.score, errors="coerce")
gp = G.pivot_table(index=["system", "jd_id", "resume_id"], columns="variant", values="score").reset_index()
gp["F_minus_M"] = gp["F"] - gp["M"]
s_g = gp.groupby("system").F_minus_M.agg(["count", "mean", lambda x: x.abs().max(),
                                          lambda x: int((x.abs() > 0.01).sum())]).round(3)
s_g.columns = ["resumes", "mean_F_minus_M", "max_abs", "resumes_changed"]

P["delta"] = pd.to_numeric(P.perturbed, errors="coerce") - pd.to_numeric(P.original, errors="coerce")
P["changed"] = P.delta.isna() | (P.delta.abs() > 0.01)
s_p = P.groupby(["system", "perturbation"]).agg(resumes=("resume_id", "size"), share_changed=("changed", "mean"),
                                                mean_delta=("delta", "mean"),
                                                max_abs_delta=("delta", lambda x: x.abs().max())).round(3)

# ---------------- upper case: before vs after the ALL-CAPS fix (PipelineConfig.caps_fix) ----------------
s_caps = None
if "caps_fix" in inspect.getsource(_pl):
    rows_c = []
    for sysname, (key, pipe_on) in SYSTEMS.items():
        cfg_off = dict(pipe_on.cfg.__dict__, caps_fix=False)
        pipe_off = MatchingPipeline(PipelineConfig(**cfg_off), embedder=embedder,
                                    skill_extractor=se.DictionarySkillExtractor())
        for ds, jd_id, jd, rid, res in cases:
            jd_df = pipe_off.extract_jd_requirements(jd)
            b = pipe_off.match(jd, res, jd_df=jd_df).get(key)
            u = pipe_off.match(jd, res.upper(), jd_df=jd_df).get(key)
            rows_c.append(dict(system=sysname, dataset=ds, version="before_fix", original=b, perturbed=u))
        log(f"  caps before_fix: {sysname} done ({time.time() - t0:.0f}s)")
    after = P[P.perturbation == "upper_case"].assign(version="after_fix")
    C = pd.concat([pd.DataFrame(rows_c), after[["system", "dataset", "version", "original", "perturbed"]]])
    C["delta"] = pd.to_numeric(C.perturbed, errors="coerce") - pd.to_numeric(C.original, errors="coerce")
    C["changed"] = C.delta.isna() | (C.delta.abs() > 0.01)
    s_caps = C.groupby(["system", "dataset", "version"]).agg(
        resumes=("delta", "size"), share_changed=("changed", "mean"), mean_delta=("delta", "mean"),
        max_abs_delta=("delta", lambda x: x.abs().max())).round(3)
else:
    log("WARNING: src/pipeline.py has no caps_fix option - upload the new src/ to get the before/after table.")

for name, df in [("step7_names_summary", s_n), ("step7_names_by_group", s_eth), ("step7_gender_summary", s_g),
                 ("step7_robustness_summary", s_p), ("step7_caps_before_after", s_caps)]:
    if df is None:
        continue
    df.to_csv(R / f"{name}.csv")
    log(f"\n### {name}\n{df.to_string()}")

LOG.close()
zp = ROOT / "results_step7.zip"
with zipfile.ZipFile(zp, "w") as z:
    for p in R.iterdir():
        z.write(p, p.name)
print("ALL DONE ->", zp)
