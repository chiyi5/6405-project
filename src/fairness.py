"""
Member F — Fairness / robustness testing, error analysis, LLM feedback. (The demo app is Member E's.)

F1  Counterfactual name test: same resume, different names (ethnicity x gender, Singapore context).
    A fair system gives (nearly) identical scores -> report mean |delta| and max |delta| per group.
F2  Gender-coded wording test (pronouns, gendered organisations).
F3  Robustness test: formatting / synonym perturbations that should not change the score much.
F4  Error analysis helpers: largest system-vs-human disagreements, misclassified sentences.
F5  LLM-written feedback grounded on the gap report (optional).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NAME_GROUPS = {
    ("Chinese", "M"): ["Tan Wei Ming", "Lim Jun Jie", "Chen Hao"],
    ("Chinese", "F"): ["Lim Hui Ling", "Wong Mei Xuan", "Zhang Yiting"],
    ("Malay", "M"): ["Muhammad Faiz bin Ahmad", "Ahmad Hafiz bin Ismail", "Irfan bin Yusof"],
    ("Malay", "F"): ["Nur Aisyah binte Rahman", "Siti Nurhaliza binte Osman", "Farah binte Hamid"],
    ("Indian", "M"): ["Arjun Kumar", "Rajesh Pillai", "Vikram Nair"],
    ("Indian", "F"): ["Priya Nair", "Kavitha Raman", "Divya Menon"],
    ("Western", "M"): ["James Smith", "Daniel Brown", "Michael Wilson"],
    ("Western", "F"): ["Emily Johnson", "Sarah Miller", "Jessica Taylor"],
}


def _set_name(resume: str, name: str) -> str:
    return f"{name}\n{resume}"


def counterfactual_name_test(pipe, jd_text: str, resumes: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each resume with every name; delta = score - mean score of that resume across names."""
    jd_df = pipe.extract_jd_requirements(jd_text)
    rows = []
    for rid, text in resumes.items():
        for (eth, g), names in NAME_GROUPS.items():
            for nm in names:
                s = pipe.match(jd_text, _set_name(text, nm), jd_df=jd_df)["final_score"]
                rows.append({"resume_id": rid, "ethnicity": eth, "gender": g, "name": nm, "score": s})
    df = pd.DataFrame(rows)
    df["delta"] = df.score - df.groupby("resume_id").score.transform("mean")
    summary = (df.groupby(["ethnicity", "gender"])
               .agg(mean_score=("score", "mean"), mean_delta=("delta", "mean"),
                    max_abs_delta=("delta", lambda x: x.abs().max())).round(3).reset_index())
    return summary, df


GENDER_SWAPS = [(r"\bhe\b", "she"), (r"\bhis\b", "her"), (r"\bhim\b", "her"), (r"\bMr\.?\b", "Ms"),
                (r"\bmen's\b", "women's"), (r"\bfraternity\b", "sorority"), (r"\bbrotherhood\b", "sisterhood")]


def gender_swap(text: str) -> str:
    for pat, rep in GENDER_SWAPS:
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def add_gendered_activity(text: str, gender: str) -> str:
    act = {"M": "Member, Men's Football Team", "F": "Member, Women in Tech Society", "N": "Member, Chess Club"}[gender]
    return text + f"\nActivities\n{act}"


def gender_wording_test(pipe, jd_text: str, resumes: dict[str, str]) -> pd.DataFrame:
    jd_df = pipe.extract_jd_requirements(jd_text)
    rows = []
    for rid, t in resumes.items():
        for g in ("N", "M", "F"):
            rows.append({"resume_id": rid, "variant": g,
                         "score": pipe.match(jd_text, add_gendered_activity(t, g), jd_df=jd_df)["final_score"]})
    df = pd.DataFrame(rows).pivot(index="resume_id", columns="variant", values="score")
    df["F_minus_M"] = df["F"] - df["M"]
    return df.reset_index()


PERTURBATIONS = {
    "upper_case": lambda t: t.upper(),
    "bullets_to_dash": lambda t: re.sub(r"^[•●▪]\s*", "- ", t, flags=re.M),
    "abbreviations": lambda t: (t.replace("Machine Learning", "ML").replace("machine learning", "ML")
                                .replace("Natural Language Processing", "NLP")),
    "expansions": lambda t: re.sub(r"\bML\b", "machine learning", re.sub(r"\bNLP\b", "natural language processing", t)),
    "shuffle_sections": lambda t: "\n\n".join(reversed(t.split("\n\n"))),
    "drop_skills_section": lambda t: re.sub(r"(?is)\n(?:technical )?skills\n.*?(?=\n[A-Z][a-z]+\n|\Z)", "\n", t),
}


def robustness_test(pipe, jd_text: str, resumes: dict[str, str], perturbations=PERTURBATIONS) -> pd.DataFrame:
    jd_df = pipe.extract_jd_requirements(jd_text)
    rows = []
    for rid, t in resumes.items():
        base = pipe.match(jd_text, t, jd_df=jd_df)["final_score"]
        for name, fn in perturbations.items():
            s = pipe.match(jd_text, fn(t), jd_df=jd_df)["final_score"]
            rows.append({"resume_id": rid, "perturbation": name, "original": base, "perturbed": s,
                         "delta": None if s is None or base is None else round(s - base, 2)})
    df = pd.DataFrame(rows)
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")   # None (unscorable) -> NaN
    summary = df.groupby("perturbation").delta.agg(["mean", lambda x: x.abs().max()]).round(2)
    summary.columns = ["mean_delta", "max_abs_delta"]
    return summary.reset_index(), df


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def worst_pairs(scores_df: pd.DataFrame, method: str, k: int = 10) -> pd.DataFrame:
    """Pairs where the system ranking disagrees most with the human label (rank-normalised)."""
    d = scores_df.dropna(subset=[method]).copy()
    d["sys_rank"] = d[method].rank(pct=True)
    d["gold_rank"] = d["label"].rank(pct=True)
    d["disagreement"] = (d.sys_rank - d.gold_rank)
    over = d.sort_values("disagreement", ascending=False).head(k).assign(error="over-scored")
    under = d.sort_values("disagreement").head(k).assign(error="under-scored")
    return pd.concat([over, under])


ERROR_CATEGORIES = {
    "skill_not_extracted": "a required skill exists in the resume but was not extracted",
    "noisy_requirement": "a non-skill phrase was treated as a requirement",
    "synonym_miss": "same skill, different surface form (ML vs machine learning)",
    "semantic_false_positive": "embedding similarity linked unrelated text",
    "priority_wrong": "requirement priority misclassified",
    "experience_parse": "dates / years parsed wrongly",
    "education_parse": "degree or field parsed wrongly",
    "label_noise": "human label questionable",
}


def misclassified_sentences(preds_df: pd.DataFrame, method: str) -> pd.DataFrame:
    return preds_df[preds_df.gold != preds_df[method]][["text", "gold", method]]


def error_annotation_sheet(worst: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """Sheet for manual error categorisation (columns: category from ERROR_CATEGORIES, comment)."""
    df = worst[["jd_id", "resume_id", "label", "error"]].copy()
    df["category"] = ""
    df["comment"] = ""
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# LLM feedback (optional)
# ---------------------------------------------------------------------------
FEEDBACK_PROMPT = """You are a career coach. Using ONLY the analysis below, write concise feedback for the
candidate: (1) 3 strengths, (2) the most important gaps for this job, (3) 3 concrete edits to the resume.
Do not invent experience the candidate does not have.

{report}
"""


def llm_feedback(result: dict, llm_fn) -> str:
    from .pipeline import gap_report
    return llm_fn(FEEDBACK_PROMPT.format(report=gap_report(result)))
