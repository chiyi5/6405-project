"""
STEP 1: download public datasets, build the requirement annotation sheet.
Output: results_step1.zip

  python experiments/step1_prepare.py            # real data (needs internet)
  python experiments/step1_prepare.py --demo     # offline check with data/demo
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_utils as du  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
ap.add_argument("--n_jds_to_label", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

OUT = ROOT / "data" / "hf"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "annotations").mkdir(parents=True, exist_ok=True)
# --demo writes its result files to results/demo_run/ so it cannot overwrite the real results/A_* files
RES = ROOT / "results" / ("demo_run" if args.demo else "")
RES.mkdir(parents=True, exist_ok=True)
if args.demo:
    print("DEMO MODE: data/hf/*.csv.gz will contain demo pairs - re-run without --demo before any real experiment.")

# ---------------------------------------------------------------- pairs
if args.demo:
    D = ROOT / "data" / "demo"
    jds, res = du.load_folder(D / "jds"), du.load_folder(D / "resumes")
    lab = pd.read_csv(D / "demo_match_labels.csv")
    allp = lab.assign(jd_text=lab.jd_id.map(jds), resume_text=lab.resume_id.map(res))
    train, test = allp.copy(), allp.copy()   # demo only: tiny data, reuse everything
else:
    train = du.load_hf_fit_pairs("train")
    try:
        test = du.load_hf_fit_pairs("test")
    except Exception as e:
        print("no test split, splitting train by JD:", e)
        train, test = du.split_by_jd(train, 0.25)
# avoid JD leakage between train and test
if not args.demo:
    leak = set(train.jd_id) & set(test.jd_id)
    train = train[~train.jd_id.isin(leak)]
train.to_csv(OUT / "train.csv.gz", index=False, compression="gzip")
test.to_csv(OUT / "test.csv.gz", index=False, compression="gzip")
print(f"train {len(train)} pairs / {train.jd_id.nunique()} JDs ; test {len(test)} pairs / {test.jd_id.nunique()} JDs")

stats = []
for name, df in (("train", train), ("test", test)):
    stats.append({"split": name, "pairs": len(df), "unique_jds": df.jd_id.nunique(),
                  "unique_resumes": df.resume_id.nunique(),
                  **{f"label_{k}": int(v) for k, v in df.label.value_counts().sort_index().items()},
                  "avg_jd_words": round(df.jd_text.str.split().str.len().mean(), 1),
                  "avg_resume_words": round(df.resume_text.str.split().str.len().mean(), 1)})
pd.DataFrame(stats).to_csv(RES / "A_hf_dataset_stats.csv", index=False)

# ---------------------------------------------------------------- JD sentences to label (Member C)
rng = np.random.default_rng(args.seed)
pool = test.drop_duplicates("jd_id")
pick = pool.sample(min(args.n_jds_to_label, len(pool)), random_state=args.seed)
jd_dict = dict(zip(pick.jd_id, pick.jd_text))
(OUT / "jds_to_label.json").write_text(json.dumps(jd_dict, ensure_ascii=False), encoding="utf-8")
sheet = du.make_requirement_sheet(jd_dict, str(ROOT / "data" / "annotations" / "requirements_to_label.csv"))

# ---------------------------------------------------------------- SkillSpan (Member B)
ss_info = {}
if not args.demo:
    try:
        from src.skill_extraction import load_skillspan_gold
        docs, gold = load_skillspan_gold("test", group_size=10)
        (OUT / "skillspan_test.json").write_text(json.dumps({"docs": docs, "gold": gold}, ensure_ascii=False),
                                                 encoding="utf-8")
        ss_info = {"skillspan_docs": len(docs), "skillspan_gold_skills": sum(len(v) for v in gold.values())}
    except Exception as e:
        ss_info = {"skillspan_error": repr(e)}
print(ss_info)

# ---------------------------------------------------------------- package outputs
sample = pd.concat([g.sample(min(len(g), 5), random_state=1) for _, g in test.groupby("label")]).copy()
sample["jd_text"] = sample.jd_text.str[:1500]
sample["resume_text"] = sample.resume_text.str[:1500]
sample.to_csv(RES / "A_sample_pairs.csv", index=False)
(RES / "A_step1_info.json").write_text(json.dumps(ss_info, indent=1), encoding="utf-8")
zp = ROOT / "results_step1.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT / "data" / "annotations" / "requirements_to_label.csv", "requirements_to_label.csv")
    for f in ("A_hf_dataset_stats.csv", "A_sample_pairs.csv", "A_step1_info.json"):
        z.write(RES / f, f)
print(f"\nDONE -> {zp}  ({zp.stat().st_size // 1024} KB). All outputs are packed into this zip.")
