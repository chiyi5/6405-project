"""
Member E — Demo UI (hybrid score, gap report, multi-resume ranking).   Run:  python app.py   (then open the printed local URL)
Upload/paste a JD and a resume -> score, requirement table, gap report.
"""
import tempfile
from pathlib import Path

import gradio as gr

from src.pipeline import MatchingPipeline, PipelineConfig, gap_report, read_document

# Final system (v3): JD skill lists + requirement-level matching + 0.5 x whole-document similarity (hybrid).
# Set SKILL_MODEL to a fine-tuned JobBERT folder (e.g. models/skill_ner) to add JobBERT skills.
import os
from src.skill_extraction import build_skill_extractor

SKILL_MODEL = os.environ.get("SKILL_MODEL", "")
PIPE = MatchingPipeline(
    PipelineConfig(embedder_name="sentence-transformers/all-MiniLM-L6-v2", doc_sim_weight=0.5),
    skill_extractor=build_skill_extractor(f"jobbert:{SKILL_MODEL}" if SKILL_MODEL else "dictionary"))


def _text(file, pasted):
    if file is not None:
        return read_document(file.name if hasattr(file, "name") else file)
    return pasted or ""


def run(jd_file, jd_text, cv_file, cv_text):
    jd, cv = _text(jd_file, jd_text), _text(cv_file, cv_text)
    if not jd.strip() or not cv.strip():
        return "Please provide both a JD and a resume.", None, ""
    res = PIPE.match(jd, cv)
    headline = (f"## Match score: {res.get('hybrid_score', res['final_score'])}\n"
                f"requirement coverage {res['final_score']} (base {res['base_score']} + bonus {res['bonus_score']}) · "
                f"document similarity {res.get('doc_similarity')} · eligibility {res['eligibility']}")
    table = res["details"][["category", "requirement", "priority", "match_score", "match_type", "resume_evidence"]].copy()
    # for unmatched rows the pipeline keeps the closest (too weak) sentence; label it so it is not read as evidence
    weak = table.match_type.isin(["NOT_EVIDENCED", "NOT_MATCHED"])
    table.loc[weak, "resume_evidence"] = "(no evidence; closest: " + table.loc[weak, "resume_evidence"].astype(str) + ")"
    out_csv = Path(tempfile.gettempdir()) / "match_details.csv"
    table.to_csv(out_csv, index=False)
    return headline, table, gap_report(res)


def run_many(jd_file, jd_text, cv_files):
    """Rank several resumes for one JD (the use case the system is built for)."""
    jd = _text(jd_file, jd_text)
    if not jd.strip() or not cv_files:
        return None
    resumes = {Path(f.name if hasattr(f, "name") else f).name: read_document(f.name if hasattr(f, "name") else f)
               for f in cv_files}
    rows = []
    jd_df = PIPE.extract_jd_requirements(jd)
    for name, text in resumes.items():
        r = PIPE.match(jd, text, jd_df=jd_df)
        rows.append({"resume": name, "score": r.get("hybrid_score", r["final_score"]),
                     "coverage": r["final_score"], "eligibility": r["eligibility"]})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def run_learning(jd_file, jd_text, cv_file, cv_text, method="hybrid", top_k=5):
    """Consume the team's matcher output; learning recommendations are a separate task."""
    from src.learning.integration import recommend_from_match
    from src.learning.workflow import plan_markdown
    jd, cv = _text(jd_file, jd_text), _text(cv_file, cv_text)
    if not jd.strip() or not cv.strip():
        return "Please provide both a JD and a resume.", {}
    plan = recommend_from_match(PIPE.match(jd, cv), method=method, top_k=int(top_k))
    return plan_markdown(plan), plan


with gr.Blocks(title="Group32 JD–Resume Matcher") as demo:
    gr.Markdown("# JD–Resume Matcher (Group 32)\nUpload files (PDF/DOCX/TXT) or paste text.")
    with gr.Row():
        with gr.Column():
            jd_file = gr.File(label="Job description file")
            jd_text = gr.Textbox(label="...or paste JD", lines=10)
        with gr.Column():
            cv_file = gr.File(label="Resume file")
            cv_text = gr.Textbox(label="...or paste resume", lines=10)
    btn = gr.Button("Match", variant="primary")
    headline = gr.Markdown()
    table = gr.Dataframe(wrap=True)
    report = gr.Textbox(label="Gap report", lines=14)
    btn.click(run, [jd_file, jd_text, cv_file, cv_text], [headline, table, report])
    gr.Markdown("### Rank several resumes for this JD")
    cv_many = gr.File(label="Upload several resumes", file_count="multiple")
    btn2 = gr.Button("Rank resumes")
    ranking = gr.Dataframe()
    btn2.click(run_many, [jd_file, jd_text, cv_many], ranking)
    gr.Markdown("### SkillBridge: evidence-based learning recommendations\n"
                "A separate contribution: skill gaps → retrieval → coverage-aware learning path. "
                "For the visual experiment workspace, run `python learning_app.py` and open http://127.0.0.1:8765.")
    with gr.Row():
        learn_method = gr.Dropdown(["bm25", "tfidf", "hybrid"], value="hybrid", label="Learning retrieval method")
        learn_k = gr.Slider(1, 12, value=5, step=1, label="Maximum learning units")
    learn_btn = gr.Button("Build learning path from these documents")
    learn_report = gr.Markdown()
    with gr.Accordion("Recommendation evidence and algorithm trace", open=False):
        learn_json = gr.JSON()
    learn_btn.click(run_learning, [jd_file, jd_text, cv_file, cv_text, learn_method, learn_k], [learn_report, learn_json])

if __name__ == "__main__":
    demo.launch()
