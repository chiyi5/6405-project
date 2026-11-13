"""Thin adapter for the existing Gradio app; no changes to matching/scoring."""
from functools import lru_cache

from .engine import LearningRecommender, gaps_from_match


@lru_cache(maxsize=1)
def get_recommender():
    return LearningRecommender()


def recommend_from_match(result, **options):
    known = result.get("profile", {}).get("skills", [])
    # Existing skill entries may be strings or structured dictionaries.
    skills = [s if isinstance(s, str) else s.get("skill", s.get("text", "")) for s in known]
    # Keep the matcher's gap decisions; do not erase a gap merely because the same
    # keyword appears in its profile dictionary (mention != sufficient evidence).
    gaps = gaps_from_match(result)
    missing = {g["skill"] for g in gaps}
    from .engine import canonical
    skills = [s for s in skills if canonical(s) not in missing]
    return get_recommender().recommend(gaps, known_skills=skills, **options)
