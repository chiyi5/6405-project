"""Append the learning study to the team's notebook without altering existing cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Group32_NLPproject_extended.ipynb"


def cell(kind, source, identity):
    result = {"cell_type": kind, "id": identity, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CELLS = [
    cell("markdown", """---
## 9. Independent extension — Evidence-based learning resource recommendation

The team pipeline answers **how the resume matches the JD**. This extension consumes its
requirement-level results and asks **which sourced learning units cover the skill evidence gaps**.
It does not modify the team's extraction, matching or scoring algorithms.

Research question: does coverage-aware selection reduce redundant topic assignments compared
with BM25, TF-IDF and lexical RRF ranking, and what prerequisite trade-offs arise?

**Run this section directly:** cells below are self-contained and do not require running Part I,
a browser, a local HTTP server, ESCO, or downloading a transformer model. If Part II's `r` is
available, it is consumed directly. Otherwise the section runs the same `MatchingPipeline`
on a documented synthetic pair already shipped in `data/demo/` using an explicit offline backend.
The resulting numbers check implementation behavior, not formal recommendation accuracy.
""", "learning-introduction"),
    cell("code", """from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown

# Open the notebook from the project root, or from its immediate parent directory.
learning_root = Path.cwd()
if not (learning_root / "src" / "pipeline.py").exists():
    learning_root = learning_root / "6405-project"
if not (learning_root / "src" / "pipeline.py").exists():
    raise FileNotFoundError("Set the notebook working directory to 6405-project.")
sys.path.insert(0, str(learning_root))
from src.learning import LearningRecommender, gaps_from_match
from src.learning.integration import recommend_from_match
from src.learning.workflow import plan_markdown

learning_output = learning_root / "results" / "learning"
learning_output.mkdir(parents=True, exist_ok=True)
learning_engine = LearningRecommender()
print("Learning corpus:", len(learning_engine.courses), "curated starter units")
""", "learning-setup"),
    cell("code", """# Reuse the team's real result when available. Keep all original variables unchanged.
LEARNING_USE_EXISTING_RESULT = True
if LEARNING_USE_EXISTING_RESULT and isinstance(globals().get("r"), dict) and "details" in r:
    learning_match_result = r
    learning_input_source = "Part II matching output r (same extraction/matching configuration)"
else:
    from src.pipeline import MatchingPipeline, PipelineConfig
    from src.skill_extraction import DictionarySkillExtractor
    learning_pipeline = MatchingPipeline(
        PipelineConfig(embedder_name="tfidf-char", doc_sim_weight=0.5),
        skill_extractor=DictionarySkillExtractor())
    learning_jd = (learning_root / "data/demo/jds/JD_backend_engineer.txt").read_text(encoding="utf-8")
    learning_resume = (learning_root / "data/demo/resumes/R02_analyst_junior.txt").read_text(encoding="utf-8")
    learning_match_result = learning_pipeline.match(learning_jd, learning_resume)
    learning_input_source = "Synthetic demo pair: backend engineer / junior analyst; offline character n-gram backend"

print("Input:", learning_input_source)
display(learning_match_result["details"])
learning_gaps = gaps_from_match(learning_match_result)
display(pd.DataFrame(learning_gaps))
""", "learning-matcher-input"),
    cell("markdown", """### 9.1 Retrieval and coverage-aware selection

The adapter only uses `Skill` rows with score below 50 or unknown score. Education, employment
duration and citizenship constraints are not treated as skills that a course can resolve.
Missing resume evidence is not evidence of missing ability.

For each gap, retrieve by BM25 and TF-IDF, fuse lexical rankings using RRF (`k=60`), then select
resources using priority-weighted new coverage, relevance, redundancy and prerequisite penalties:

`new coverage + 0.15 × normalized relevance − 0.25 × repeated coverage − 0.5 × missing prerequisites`

Priority weights: mandatory=3, required=2, unspecified=1, preferred=0.5.
Only curated source-evidence mappings can count as covered skills. The selected units are ordered
by editorial prerequisite hints. This is a heuristic, not a guarantee of an optimal study plan.
""", "learning-method"),
    cell("code", """learning_plan = recommend_from_match(learning_match_result, method="hybrid", top_k=5)
learning_table = pd.DataFrame([
    {"step": c["step"], "resource": c["title"], "covers": ", ".join(c["covered_gaps"]),
     "new_topics": ", ".join(c["newly_covered_gaps"]),
     "prerequisites_to_confirm": ", ".join(c["unmet_prerequisites"]), "source": c["url"]}
    for c in learning_plan["recommendations"]])
display(learning_table)
print("Uncovered:", learning_plan["uncovered_skills"])
print("Weighted syllabus-topic coverage:", learning_plan["weighted_coverage"])
display(pd.DataFrame(learning_plan["selection_trace"]))

learning_evidence = pd.DataFrame([
    {"resource_id": c["id"], "skill": skill, "source_span": c["evidence"][skill], "source_url": c["url"]}
    for c in learning_plan["recommendations"] for skill in c["covered_gaps"]])
display(learning_evidence)
learning_table.to_csv(learning_output / "L_recommendations.csv", index=False, encoding="utf-8-sig")
learning_evidence.to_csv(learning_output / "L_source_evidence.csv", index=False, encoding="utf-8-sig")
(learning_output / "L_plan.json").write_text(json.dumps(learning_plan, ensure_ascii=False, indent=2), encoding="utf-8")
(learning_output / "L_plan.md").write_text(plan_markdown(learning_plan), encoding="utf-8")
""", "learning-recommendation-results"),
    cell("code", """# Same gaps and candidate catalog for every configuration; no human accuracy claims.
learning_comparison_rows = []
for learning_method, learning_optimize in [("bm25", False), ("tfidf", False), ("hybrid", False), ("hybrid", True)]:
    learning_comparison_result = learning_engine.recommend(
        learning_gaps, method=learning_method, optimize=learning_optimize, top_k=5)
    learning_comparison_rows.append({
        "method": learning_method, "coverage_optimization": learning_optimize,
        "weighted_topic_coverage": learning_comparison_result["weighted_coverage"],
        "redundancy_ratio": learning_comparison_result["redundancy_ratio"],
        "learning_units": len(learning_comparison_result["recommendations"]),
        "prerequisite_warnings": sum(bool(c["unmet_prerequisites"]) for c in learning_comparison_result["recommendations"])})
learning_comparison = pd.DataFrame(learning_comparison_rows)
display(learning_comparison)
learning_comparison.to_csv(learning_output / "L_pair_ablation.csv", index=False, encoding="utf-8-sig")
""", "learning-ablation"),
    cell("markdown", """### 9.2 Reproducibility, limitations and contribution evidence

Run `python experiments/evaluate_learning.py --smoke` for the 12 synthetic implementation cases
(48 case/configuration rows). They include aliases, unknown skills, known skills, prerequisites,
priority weighting and redundant resources. These are **not independent judged data**.

Formal retrieval evaluation is prepared in `experiments/evaluate_learning.py`: independently judge
the resource relevance, retain annotator identities, split source profiles by `group_id`, then report
NDCG@K, Recall@K and MRR@K on the held-out split. Unlabelled relevance must not become zero.

Current limits: 12 starter units from 6 providers; manual topic mappings; unknown durations;
lexical retrieval; prerequisite planning can omit useful foundation units. A tiny catalog may give
identical coverage across retrieval methods. Lower redundancy alone does not prove better learning.

The independent contribution is the gap adapter, resource schema, retrieval/fusion, coverage selection,
evidence trace, experimental design and notebook integration. Team matching work remains credited
to the original members. AI assistance and the student's actual validation/modifications should be
declared in the group reflection; this implementation is not evidence of independently hand-written code.
""", "learning-limitations"),
]


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # Idempotent: replace only cells belonging to this extension.
    notebook["cells"] = [c for c in notebook["cells"] if not c.get("id", "").startswith("learning-")]
    notebook["cells"].extend(CELLS)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Integrated {len(CELLS)} cells into {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
