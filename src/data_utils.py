"""
Member A — Data collection, annotation sheets and inter-annotator agreement.

Data sources (download yourself, then point the loaders to the files)
  * Own JDs:   data/raw/jds/*.txt|pdf|docx        (MyCareersFuture, LinkedIn, NodeFlair, company sites)
  * Own resumes: data/raw/resumes/*.pdf|docx|txt  (team members' / volunteers' resumes, ANONYMISED)
  * Kaggle "Resume Dataset" (snehaanbhawal):  Resume.csv  (ID, Resume_str, Resume_html, Category)
  * HuggingFace "cnamuangtoun/resume-job-description-fit": resume_text, job_description_text, label
    (labels: "No Fit" / "Potential Fit" / "Good Fit") -> large labelled pair set for Member D / E.
  * HuggingFace "jjzha/skillspan" (SkillSpan, NAACL 2022): skill span annotations -> Member B.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import CONTACT_RE, clean_text, read_document
from .requirement_clf import RuleClassifier, split_with_sections


def _hid(text: str, prefix: str) -> str:
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_folder(folder: str, exts=(".txt", ".pdf", ".docx")) -> dict[str, str]:
    out = {}
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in exts:
            try:
                out[p.stem] = clean_text(read_document(p))
            except Exception as e:  # pragma: no cover
                print(f"[skip] {p.name}: {e}")
    print(f"Loaded {len(out)} documents from {folder}")
    return out


def load_kaggle_resumes(csv_path: str, categories=None, n_per_category: int | None = None,
                        seed: int = 42) -> pd.DataFrame:
    """Kaggle Resume.csv  ->  DataFrame(resume_id, category, resume_text)."""
    df = pd.read_csv(csv_path)
    text_col = "Resume_str" if "Resume_str" in df.columns else "Resume"
    df = df.rename(columns={text_col: "resume_text", "Category": "category"})
    df["resume_id"] = df["ID"].astype(str) if "ID" in df.columns else [f"K{i}" for i in range(len(df))]
    if categories:
        df = df[df.category.isin(categories)]
    if n_per_category:
        df = pd.concat([g.sample(min(len(g), n_per_category), random_state=seed)
                        for _, g in df.groupby("category")])
    df["resume_text"] = df["resume_text"].astype(str).map(clean_text)
    return df[["resume_id", "category", "resume_text"]].reset_index(drop=True)


def load_hf_fit_pairs(split: str = "train", n: int | None = None, seed: int = 42,
                      local_path: str | None = None) -> pd.DataFrame:
    """
    Resume–JD fit pairs -> DataFrame(jd_id, resume_id, jd_text, resume_text, label[0/1/2]).
    Either from HuggingFace hub (needs `datasets` + internet) or a local CSV/parquet export.
    """
    if local_path:
        df = pd.read_parquet(local_path) if local_path.endswith(".parquet") else pd.read_csv(local_path)
    else:
        from datasets import load_dataset
        df = load_dataset("cnamuangtoun/resume-job-description-fit", split=split).to_pandas()
    df = df.rename(columns={"job_description_text": "jd_text"})
    lab = {"no fit": 0, "potential fit": 1, "good fit": 2}
    df["label"] = df["label"].astype(str).str.lower().str.strip().map(lab)
    df = df.dropna(subset=["label"])
    if n:
        df = pd.concat([g.sample(min(len(g), n // 3), random_state=seed) for _, g in df.groupby("label")])
    df["jd_id"] = df.jd_text.map(lambda t: _hid(t, "J"))
    df["resume_id"] = df.resume_text.map(lambda t: _hid(t, "R"))
    df["label"] = df["label"].astype(int)
    return df[["jd_id", "resume_id", "jd_text", "resume_text", "label"]].reset_index(drop=True)


def split_by_jd(pairs: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    """Train/test split where all pairs of one JD stay together (avoids leakage)."""
    rng = np.random.default_rng(seed)
    jds = pairs.jd_id.unique()
    rng.shuffle(jds)
    test = set(jds[: max(1, int(len(jds) * test_size))])
    m = pairs.jd_id.isin(test)
    return pairs[~m].reset_index(drop=True), pairs[m].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anonymisation (privacy + fairness)
# ---------------------------------------------------------------------------
def anonymise_resume(text: str, name: str | None = None) -> str:
    """Remove contact lines and (optionally) the candidate's name."""
    lines = []
    for ln in text.splitlines():
        if CONTACT_RE.search(ln) and len(ln) < 120:
            continue
        lines.append(ln)
    out = "\n".join(lines)
    if name:
        for part in [name] + name.split():
            if len(part) > 1:
                out = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", out)
    return out


# ---------------------------------------------------------------------------
# Annotation sheets
# ---------------------------------------------------------------------------
def make_requirement_sheet(jds: dict[str, str], out_csv: str, prefill_rules: bool = True) -> pd.DataFrame:
    """One row per JD sentence. Annotators fill `gold_label` (REQUIRED/PREFERRED/MANDATORY/UNSPECIFIED)."""
    rule = RuleClassifier(use_section_headers=True)
    rows = []
    for jid, text in jds.items():
        for i, (sent, hdr) in enumerate(split_with_sections(clean_text(text))):
            rows.append({"jd_id": jid, "sent_id": f"{jid}_{i:03d}", "section_header": hdr, "sentence": sent,
                         "rule_suggestion": rule.predict_one(sent, hdr) if prefill_rules else "",
                         "gold_label": "", "annotator": "", "notes": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} sentences -> {out_csv}")
    return df


def make_skill_sheet(docs: dict[str, str], out_csv: str, doc_type: str = "JD", suggestions=None) -> pd.DataFrame:
    """Long format: one row per (doc, skill). Optional pre-filled suggestions that annotators keep/delete/add."""
    rows = []
    for did, text in docs.items():
        sug = suggestions.extract(text) if suggestions else []
        for s in sug:
            rows.append({"doc_id": did, "doc_type": doc_type, "skill": s, "keep(1/0)": "", "annotator": ""})
        rows.append({"doc_id": did, "doc_type": doc_type, "skill": "<add missing skills as new rows>",
                     "keep(1/0)": "", "annotator": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def make_match_sheet(jds: dict[str, str], resumes: dict[str, str], out_csv: str,
                     pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Annotators read JD + resume and give label 0 (No fit) / 1 (Potential) / 2 (Good) + a 0-100 score."""
    pairs = pairs or [(j, r) for j in jds for r in resumes]
    df = pd.DataFrame([{"jd_id": j, "resume_id": r, "label": "", "score_0_100": "", "annotator": "", "rationale": ""}
                       for j, r in pairs])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def load_skill_gold(csv_path: str, annotator: str | None = None) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    if "keep(1/0)" in df.columns:
        df = df[df["keep(1/0)"].astype(str).isin(["1", "1.0"])]
    df = df[~df.skill.astype(str).str.startswith("<")]
    if annotator:
        df = df[df.annotator == annotator]
    return df.groupby("doc_id")["skill"].apply(lambda s: list(dict.fromkeys(s.astype(str).str.strip()))).to_dict()


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def fleiss_kappa(table: pd.DataFrame) -> float:
    """table: rows = items, columns = annotators, values = category labels."""
    cats = sorted(pd.unique(table.values.ravel()))
    counts = np.array([[(row == c).sum() for c in cats] for row in table.values], dtype=float)
    n = counts.sum(1)[0]
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def agreement_report(long_df: pd.DataFrame, item_col: str, label_col: str = "gold_label",
                     annotator_col: str = "annotator") -> pd.DataFrame:
    """Pairwise Cohen's kappa + Fleiss' kappa for items labelled by >=2 annotators."""
    wide = long_df.pivot_table(index=item_col, columns=annotator_col, values=label_col, aggfunc="first")
    anns = list(wide.columns)
    rows = []
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            sub = wide[[anns[i], anns[j]]].dropna()
            if len(sub) >= 5:
                rows.append({"pair": f"{anns[i]} vs {anns[j]}", "n_items": len(sub),
                             "raw_agreement": round(float((sub.iloc[:, 0] == sub.iloc[:, 1]).mean()), 3),
                             "cohen_kappa": round(cohen_kappa(sub.iloc[:, 0], sub.iloc[:, 1]), 3)})
    full = wide.dropna()
    if len(anns) >= 3 and len(full) >= 5:
        rows.append({"pair": "ALL (Fleiss)", "n_items": len(full), "cohen_kappa": round(fleiss_kappa(full), 3)})
    return pd.DataFrame(rows)


def skill_agreement(gold_a: dict, gold_b: dict) -> float:
    """Inter-annotator agreement for skill lists = mean soft-F1 between two annotators."""
    from .skill_extraction import prf
    common = set(gold_a) & set(gold_b)
    return float(np.mean([prf(gold_a[d], gold_b[d], "soft")["f1"] for d in common])) if common else np.nan


def adjudicate(long_df: pd.DataFrame, item_col: str, label_col: str = "gold_label") -> pd.DataFrame:
    """Majority vote (ties -> first annotator's label, flagged for discussion)."""
    def vote(g):
        vc = g[label_col].value_counts()
        tie = len(vc) > 1 and vc.iloc[0] == vc.iloc[1]
        return pd.Series({label_col: vc.index[0], "tie": tie, "n_annotators": len(g)})
    return long_df.dropna(subset=[label_col]).groupby(item_col).apply(vote).reset_index()


def dataset_statistics(jds: dict, resumes: dict, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    def stats(d, name):
        lens = [len(t.split()) for t in d.values()]
        return {"collection": name, "n_docs": len(d), "avg_words": round(np.mean(lens), 1) if lens else 0,
                "min_words": min(lens) if lens else 0, "max_words": max(lens) if lens else 0}
    rows = [stats(jds, "JDs"), stats(resumes, "Resumes")]
    if pairs is not None and "label" in pairs:
        for l, c in pairs.label.value_counts().sort_index().items():
            rows.append({"collection": f"pairs label={l}", "n_docs": int(c)})
    return pd.DataFrame(rows)
