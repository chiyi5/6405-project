"""Independent, evidence-grounded learning recommendation module."""
from .engine import LearningRecommender, gaps_from_match, parse_gaps

__all__ = ["LearningRecommender", "gaps_from_match", "parse_gaps"]
