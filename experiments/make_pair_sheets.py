"""
Build one Excel labelling sheet per team member for the 100 JD–resume pairs in
data/annotations/pairs_for_human_labelling.csv. Each pair goes to exactly 2 of the 6 members.
Needs data/hf/test.csv.gz (run experiments/step1_prepare.py first).
Output: pair_sheets.zip  (pair_labels_A.xlsx … pair_labels_F.xlsx + assignment.csv)
"""
import itertools
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import clean_text  # noqa: E402

MEMBERS = list("ABCDEF")
OUT = ROOT / "pair_sheets"
OUT.mkdir(exist_ok=True)

pairs = pd.read_csv(ROOT / "data" / "annotations" / "pairs_for_human_labelling.csv")
test = pd.read_csv(ROOT / "data" / "hf" / "test.csv.gz").drop_duplicates(["jd_id", "resume_id"])
df = pairs.merge(test[["jd_id", "resume_id", "jd_text", "resume_text"]], on=["jd_id", "resume_id"], how="left")
missing = df.jd_text.isna().sum()
assert missing == 0, f"{missing} pairs not found in test split — re-run step1_prepare.py"

# each pair -> 2 members, cycling through all 15 member pairs so the workload is balanced
combos = list(itertools.combinations(MEMBERS, 2))
assign = [{"pair_id": pid, "annotator": m} for i, pid in enumerate(df.pair_id) for m in combos[i % len(combos)]]
assign = pd.DataFrame(assign)
assign.to_csv(OUT / "assignment.csv", index=False)
print(assign.annotator.value_counts().sort_index().to_dict())

GUIDE = [
    ["How to label (read before starting)"],
    [""],
    ["Read the job description, then the resume. Judge ONLY from the text. Do not discuss with other members."],
    ["label: 2 = Good fit (worth interviewing: meets almost all core requirements)"],
    ["       1 = Potential fit (related background, meets about half, or short on years / degree)"],
    ["       0 = No fit (different field, or misses most core requirements)"],
    ["score_0_100: your overall impression, 0 = no fit at all, 100 = perfect fit"],
    ["rationale: one short sentence, e.g. 'Data engineer with 6 yrs SQL/Spark; JD wants exactly that'"],
    ["Ignore names, gender, school prestige. About 3 minutes per pair is enough."],
]

for m in MEMBERS:
    ids = assign[assign.annotator == m].pair_id
    sub = df[df.pair_id.isin(ids)].copy()
    sheet = pd.DataFrame({
        "pair_id": sub.pair_id,
        "job_description": sub.jd_text.map(lambda t: clean_text(str(t))[:30000]),
        "resume": sub.resume_text.map(lambda t: clean_text(str(t))[:30000]),
        "label (0/1/2)": "", "score_0_100": "", "rationale": "",
    })
    path = OUT / f"pair_labels_{m}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(GUIDE).to_excel(xw, sheet_name="instructions", index=False, header=False)
        sheet.to_excel(xw, sheet_name="labels", index=False)
        ws = xw.sheets["labels"]
        from openpyxl.styles import Alignment
        for col, width in zip("ABCDEF", (10, 80, 80, 12, 12, 50)):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[row[0].row].height = 300
        ws.freeze_panes = "B2"
        xw.sheets["instructions"].column_dimensions["A"].width = 110
    print("wrote", path.name, len(sheet), "pairs")

zp = ROOT / "pair_sheets.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob("*")):
        z.write(f, f.name)
print(f"\nDONE -> {zp.name}: send one xlsx to each member (A–F); keep assignment.csv.")
