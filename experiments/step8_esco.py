"""
Step 8 — Run the original Part I ESCO skill extractor (Nesta ojd-daps-skills) as a quantitative baseline.

The library does not install on Python 3.13, so this script builds a separate Python 3.11 (or 3.10)
environment with `uv`, runs the extractor there on the same 150 SkillSpan test documents used in Section 5,
and evaluates the predictions here with the same metric code.

Run:   python experiments/step8_esco.py        (~10-20 min; no GPU needed)
Outputs: results_step8/ and results_step8.zip
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import skill_extraction as se  # noqa: E402

R = ROOT / "results_step8"
R.mkdir(exist_ok=True)
LOG = open(R / "run_log_step8.txt", "w", encoding="utf-8")
ENV = Path("/content/esco_env") if Path("/content").exists() else ROOT / "esco_env"
WIN = sys.platform.startswith("win")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def sh(cmd, check=True):
    log("$", cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    if out:
        log("\n".join(out.splitlines()[-15:]))
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.returncode == 0


# ---------------- 1. data: the same 150 SkillSpan test documents as Section 5 ----------------
ssj = ROOT / "data" / "hf" / "skillspan_test.json"
if ssj.exists():
    d = json.loads(ssj.read_text(encoding="utf-8"))
    docs, gold = d["docs"], d["gold"]
else:
    docs, gold = se.load_skillspan_gold("test")
keys = list(docs)[:150]
docs, gold = {k: docs[k] for k in keys}, {k: gold[k] for k in keys}
(R / "docs.json").write_text(json.dumps(docs), encoding="utf-8")
log(f"{len(docs)} SkillSpan test documents")

# ---------------- 2. separate Python environment with ojd-daps-skills ----------------
import shutil  # noqa: E402

py = ENV / ("Scripts/python.exe" if WIN else "bin/python")
UV = f'"{sys.executable}" -m uv'
if not py.exists():
    sh(f'"{sys.executable}" -m pip -q install uv')
    ok = False
    for ver in ("3.11", "3.10"):
        if sh(f'{UV} venv "{ENV}" --python {ver} --seed', check=False) and \
           sh(f'{UV} pip install --python "{py}" ojd-daps-skills click', check=False):
            ok = True
            log(f"ojd-daps-skills installed with Python {ver}")
            break
        shutil.rmtree(ENV, ignore_errors=True)
    if not ok:
        raise SystemExit("Could not install ojd-daps-skills - see results_step8/run_log_step8.txt")
    sh(f'"{py}" -m spacy download en_core_web_sm', check=False)
sh(f'{UV} pip install --python "{py}" click', check=False)      # missing dependency seen in practice
sh(f'"{py}" -m pip show ojd-daps-skills spacy', check=False)

# ---------------- 3. run the extractor in that environment ----------------
worker = R / "esco_worker.py"
worker.write_text('''
import json, re, sys
from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor
docs = json.load(open(sys.argv[1], encoding="utf-8"))
sm = SkillsExtractor(taxonomy_name="esco")
ids, texts = list(docs), list(docs.values())
out = {}
for i in range(0, len(ids), 10):
    for did, doc in zip(ids[i:i + 10], sm(texts[i:i + 10])):
        spans = [re.sub(r"\\s+", " ", sp if isinstance(sp, str) else getattr(sp, "text", "")).strip()
                 for sp in doc._.skill_spans]
        mapped = []
        try:
            mapped = [str(m) for m in doc._.mapped_skills]
        except Exception:
            pass
        out[did] = {"spans": list(dict.fromkeys(s for s in spans if s)), "mapped": mapped}
    print(f"  {min(i + 10, len(ids))}/{len(ids)}", flush=True)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"))
''', encoding="utf-8")
pred_path = R / "esco_predictions.json"
sh(f'"{py}" "{worker}" "{R / 'docs.json'}" "{pred_path}"')
pred = json.loads(pred_path.read_text(encoding="utf-8"))
log("example:", keys[0], pred[keys[0]]["spans"][:10])


# ---------------- 4. evaluate with the Section 5 metric ----------------
class Precomputed:
    def __init__(self, f):
        self.f = f

    def extract(self, text):
        return self.f(text)


by_text = {docs[k]: pred[k]["spans"] for k in keys}
dic = se.DictionarySkillExtractor()
m1 = Precomputed(lambda t: by_text[t])
m2 = Precomputed(lambda t: se.postprocess_spans(by_text[t], dic))
ex = {"M1_esco (Part I extractor)": m1, "M2_esco_filtered": m2, "M3_dictionary": dic,
      "M4_dictionary+esco_filtered (Part I default)": se.CombinedExtractor(dic, m2)}
ner = ROOT / "models" / "skill_ner"
if (ner / "config.json").exists():
    try:
        ex["M6_jobbert_skillspan"] = se.TokenClassifierExtractor(str(ner))
    except Exception as e:
        log("JobBERT not loaded:", repr(e)[:200])
summary, per = se.evaluate_extractors(ex, docs, gold)
summary.to_csv(R / "B_skill_extraction_with_esco.csv", index=False)
per.to_csv(R / "B_skill_per_doc_with_esco.csv", index=False)
log("\n### B_skill_extraction_with_esco.csv\n" + summary.to_string())
err = se.extraction_errors(m1, docs, gold)
err.to_csv(R / "B_errors_esco.csv", index=False)
log("\nESCO false positives (sample):", err[err.type == "FP"].skill.head(25).tolist())

LOG.close()
zp = ROOT / "results_step8.zip"
with zipfile.ZipFile(zp, "w") as z:
    for p in R.iterdir():
        if p.suffix in (".csv", ".txt", ".json") and p.name != "docs.json":
            z.write(p, p.name)
print("ALL DONE ->", zp)
