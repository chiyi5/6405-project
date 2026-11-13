"""Production entry point for the large-catalogue recommender (3,428 courses).

Uses the configuration that won on the benchmark (results/learning/bench):
dev-tuned BM25 per-gap retrieval + xQuAD diversification (λ=0.7). Every
recommendation carries a verbatim evidence sentence from the course description,
or is marked "title match only" — the recommender never invents claims.

    from src.learning.course_search import CourseSearch
    cs = CourseSearch()
    cs.plan([{"skill": "SQL", "priority": "REQUIRED"}, {"skill": "Docker", "priority": "PREFERRED"}], k=5)
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from .benchmark import BENCH, load_benchmark, save_benchmark
from .planner import PRIORITY_WEIGHT, plan
from .retrievers import BM25Retriever

_SENT = re.compile(r"(?<=[.!?])\s+")


class CourseSearch:
    def __init__(self, k1: float = 2.0, b: float = 0.3, lam: float = 0.7):
        if not (BENCH / "corpus.jsonl").exists():
            save_benchmark()
        self.corpus, *_ = load_benchmark()
        self.lam = lam
        t = time.perf_counter()
        self.retriever = BM25Retriever(k1, b).fit(list(self.corpus.text))
        self.index_seconds = time.perf_counter() - t

    def evidence(self, idx: int, skill: str) -> str:
        row = self.corpus.iloc[idx]
        pat = re.compile(r"(?<![\w+#])" + re.escape(skill.lower()) + r"(?![\w+#])")
        for s in _SENT.split(row.description):
            if pat.search(s.lower()):
                return s.strip()[:300]
        return "title match only" if pat.search(row.title.lower()) else ""

    def plan(self, gaps: list[dict], k: int = 5, strategy: str = "xquad") -> dict:
        t = time.perf_counter()
        gaps = [g for g in gaps if str(g.get("skill", "")).strip()]
        chosen = plan(gaps, self.retriever, k=k, strategy=strategy, lam=self.lam)
        scores = self.retriever.score([g["skill"] for g in gaps]) if gaps else None
        recs, covered = [], set()
        for rank, i in enumerate(chosen, 1):
            row = self.corpus.iloc[i]
            ev = {g["skill"]: self.evidence(i, g["skill"]) for g in gaps}
            ev = {s: e for s, e in ev.items() if e}
            new = sorted(set(ev) - covered)
            covered |= set(ev)
            best_gap = max(range(len(gaps)), key=lambda j: scores[j, i]) if gaps else None
            recs.append({"rank": rank, "id": row.id, "title": row.title, "provider": row.provider,
                         "level": row.level, "rating": None if row.rating != row.rating else float(row.rating),
                         "url": row.url, "source": row.source,
                         "best_matching_gap": gaps[best_gap]["skill"] if best_gap is not None else None,
                         "evidence": ev, "newly_covered": new})
        w = {g["skill"]: PRIORITY_WEIGHT.get(g.get("priority", "REQUIRED"), 1.0) for g in gaps}
        return {"recommendations": recs, "strategy": strategy, "lambda": self.lam,
                "evidence_coverage": round(sum(w[s] for s in covered) / sum(w.values()), 4) if w else 0.0,
                "uncovered": sorted(set(w) - covered), "latency_ms": round(1000 * (time.perf_counter() - t), 1),
                "catalogue_size": len(self.corpus),
                "notice": "Evidence = sentence of the course description naming the skill. "
                          "Coverage is a text-evidence estimate, not a guarantee of learning outcomes."}


@lru_cache(maxsize=1)
def get_course_search() -> CourseSearch:
    return CourseSearch()
