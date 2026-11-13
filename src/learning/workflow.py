"""Run the team's actual matcher and adapt its output, without changing its scores."""
from functools import lru_cache
import json

from .integration import recommend_from_match


@lru_cache(maxsize=1)
def get_pipeline():
    from src.pipeline import MatchingPipeline, PipelineConfig
    from src.skill_extraction import DictionarySkillExtractor
    # Explicit offline mode: never report a downloaded embedding model here.
    return MatchingPipeline(PipelineConfig(embedder_name="tfidf-char", doc_sim_weight=0.5),
                            skill_extractor=DictionarySkillExtractor())


def match_documents(jd, resume):
    if not isinstance(jd, str) or not isinstance(resume, str) or not jd.strip() or not resume.strip():
        raise ValueError("Please provide both a job description and a resume.")
    if len(jd) > 40000 or len(resume) > 40000:
        raise ValueError("Each document must be at most 40000 characters.")
    pipe = get_pipeline()
    result = pipe.match(jd, resume)
    # DataFrame.to_json converts numpy values and NaN to standard JSON/null.
    details = json.loads(result["details"].to_json(orient="records"))
    return {"details": details, "profile": {"skills": result["profile"]["skills"]},
            "scores": {k: result.get(k) for k in ("hybrid_score", "final_score", "eligibility")},
            "backend": "Team MatchingPipeline / dictionary skills / hashed character n-grams (offline)",
            "notice": "Offline demonstration backend; not the team's MiniLM/JobBERT benchmark configuration."}


def build_workflow(jd, resume, **options):
    match = match_documents(jd, resume)
    return {"match_result": match, "plan": recommend_from_match(match, **options)}


def plan_markdown(plan):
    """Portable report for the group reflection and demonstration."""
    lines = ["# SkillBridge learning plan", "", f"Backend: {plan['backend']}",
             f"Weighted syllabus-topic coverage: {plan['weighted_coverage']:.1%}",
             f"Redundant topic assignments: {plan['redundancy_ratio']:.1%}", "",
             "Coverage is not mastery; absent resume evidence is not proof of absent ability.", ""]
    for c in plan["recommendations"]:
        lines += [f"## {c['step']}. {c['title']}", "", c["description"], "",
                  "Targets: " + ", ".join(c["covered_gaps"]), "",
                  f"Source: {c['url']}", ""]
        for skill in c["covered_gaps"]:
            lines += [f"- {skill}: {c['evidence'][skill]}"]
        lines += ["", "Prerequisites to confirm: " + (", ".join(c["unmet_prerequisites"]) or "none"), ""]
    lines += ["## Uncovered topics", "", ", ".join(plan["uncovered_skills"]) or "None in this catalog mapping."]
    return "\n".join(lines)
