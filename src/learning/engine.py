"""Retrieval, RRF and weighted set coverage; no mandatory third-party packages.

Skill evidence is curator-labelled source text, not a model's claim of competence.
The optional dense backend fails explicitly instead of silently becoming lexical.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "learning" / "catalog.json"
WEIGHTS = {"MANDATORY": 3.0, "REQUIRED": 2.0, "UNSPECIFIED": 1.0, "PREFERRED": 0.5}
ALIASES = {
    "python": ["python programming", "python语言", "python编程"],
    "sql": ["postgresql", "postgres", "mysql", "数据库查询"],
    "sql joins": ["join", "joins", "table joins", "多表连接"],
    "sql aggregation": ["group by", "aggregate functions", "聚合查询"],
    "docker": ["containerization", "containers", "容器化", "容器"],
    "fastapi": ["fast api"],
    "api development": ["rest api", "restful", "web api", "接口开发"],
    "testing": ["unit testing", "pytest", "单元测试", "测试"],
    "machine learning": ["ml", "机器学习"],
    "model evaluation": ["cross validation", "cross-validation", "模型评估", "交叉验证"],
    "git": ["version control", "版本控制"],
}
STOP = set("a an the and or of to in for with is are on by this that course learn learning".split())


def canonical(skill):
    value = re.sub(r"\s+", " ", str(skill).strip().lower())
    for name, variants in ALIASES.items():
        if value == name or value in variants:
            return name
    return value


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9+#]+|[\u4e00-\u9fff]+", text.lower()) if t not in STOP]


def normalise_gaps(items):
    if not isinstance(items, list) or len(items) > 50:
        raise ValueError("Provide a list of at most 50 skill gaps.")
    merged = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each gap must be an object containing skill and priority.")
        skill = canonical(item.get("skill", ""))
        if not skill:
            continue
        if len(skill) > 150:
            raise ValueError("Skill name exceeds 150 characters.")
        priority = str(item.get("priority", "REQUIRED")).upper()
        if priority not in WEIGHTS:
            raise ValueError(f"Unsupported priority: {priority}")
        gap = {"skill": skill, "priority": priority, "weight": WEIGHTS[priority]}
        if skill not in merged or gap["weight"] > merged[skill]["weight"]:
            merged[skill] = gap
    return list(merged.values())


def parse_gaps(text):
    """One skill per line; optional `skill | PRIORITY` suffix."""
    rows = []
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            parts = line.split("|", 1)
            rows.append({"skill": parts[0], "priority": parts[1].strip() if len(parts) > 1 else "REQUIRED"})
    return normalise_gaps(rows)


def gaps_from_match(result, threshold=50.0):
    """Accept pipeline.match() (DataFrame) or JSON {details: [records]}.

    Only skill rows below threshold/unknown are returned. Education and years of
    experience are deliberately not converted to learnable skills.
    """
    details = result.get("details", [])
    if hasattr(details, "to_dict"):
        details = details.to_dict("records")
    rows = []
    for row in details:
        if str(row.get("category", "")).lower() != "skill":
            continue
        if row.get("match_type") == "DISABLED":
            continue
        raw = row.get("match_score")
        try:
            score = float(raw)
        except (ValueError, TypeError):
            score = float("nan")
        if not math.isfinite(score) or score < threshold:
            rows.append({"skill": row.get("requirement", ""),
                         "priority": row.get("priority") or "UNSPECIFIED"})
    return normalise_gaps(rows)


def load_catalog(path=CATALOG):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    courses = raw["resources"]
    if not courses:
        raise ValueError("Catalog is empty.")
    ids = set()
    for course in courses:
        cid = course["id"]
        if cid in ids:
            raise ValueError(f"Duplicate resource ID: {cid}")
        ids.add(cid)
        if not course["url"].startswith("https://"):
            raise ValueError(f"Resource requires an HTTPS source: {cid}")
        course["skills"] = [canonical(s) for s in course["skills"]]
        evidence = {}
        for skill, quote in course["evidence"].items():
            name = canonical(skill)
            if not quote or quote not in course["source_excerpt"]:
                raise ValueError(f"Evidence must be an exact source excerpt span: {cid}/{skill}")
            evidence[name] = quote
        course["evidence"] = evidence
        if set(course["skills"]) != set(evidence):
            raise ValueError(f"Every skill needs a curated evidence span: {cid}")
        duration = course.get("duration_hours")
        if duration is not None and (not isinstance(duration, (float, int)) or duration <= 0):
            raise ValueError(f"Invalid duration: {cid}")
    return raw


class LearningRecommender:
    def __init__(self, catalog_path=CATALOG, dense_model=None):
        catalog = load_catalog(catalog_path)
        self.metadata = {k: v for k, v in catalog.items() if k != "resources"}
        self.courses = catalog["resources"]
        self.dense_model = dense_model
        self._encoder = None
        self._vectors = None
        self.texts = [" ".join([c["title"], c["description"], c["source_excerpt"],
                                    " ".join(c["skills"])]) for c in self.courses]
        self.docs = [Counter(tokens(t)) for t in self.texts]
        self.df = Counter(t for d in self.docs for t in d)
        self.avgdl = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        n = len(self.docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in self.df.items()}
        self.tfidf = [self._tfidf(d) for d in self.docs]

    def _tfidf(self, counts):
        vec = {t: (1 + math.log(f)) * self.idf[t] for t, f in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def _bm25(self, query):
        n = len(self.docs)
        scores = []
        for doc in self.docs:
            dl, score = sum(doc.values()), 0.0
            for t in set(tokens(query)):
                f = doc[t]
                if f:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
            scores.append(score)
        return scores

    def _dense(self, query):
        if not self.dense_model:
            raise ValueError("Dense retrieval requires --dense-model with an installed SentenceTransformer model.")
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ValueError("Install sentence-transformers for dense/hybrid_dense; no lexical fallback is used.") from exc
            self._encoder = SentenceTransformer(self.dense_model)
            self._vectors = self._encoder.encode(self.texts, normalize_embeddings=True)
        q = self._encoder.encode([query], normalize_embeddings=True)[0]
        return [float(v @ q) for v in self._vectors]

    def _retrieve_one(self, query, method):
        if method == "bm25":
            scores = self._bm25(query)
        elif method == "tfidf":
            q = self._tfidf(Counter(tokens(query)))
            scores = [sum(v * d.get(t, 0) for t, v in q.items()) for d in self.tfidf]
        elif method == "dense":
            scores = self._dense(query)
        else:
            second = "dense" if method == "hybrid_dense" else "tfidf"
            scores = [0.0] * len(self.courses)
            for backend in ("bm25", second):
                ranking = self._retrieve_one(query, backend)
                for rank, (idx, _) in enumerate(ranking, start=1):
                    scores[idx] += 1 / (60 + rank)
        # Zero lexical matches must not enter RRF and acquire arbitrary rank credit.
        return sorted(((i, s) for i, s in enumerate(scores) if s > 0),
                      key=lambda x: (-x[1], self.courses[x[0]]["id"]))

    def recommend(self, gaps, *, method="hybrid", top_k=5, pool_size=30,
                  optimize=True, known_skills=(), budget_hours=None):
        start = time.perf_counter()
        gaps = normalise_gaps(gaps)
        if method not in {"bm25", "tfidf", "dense", "hybrid", "hybrid_dense"}:
            raise ValueError("Unknown retrieval method.")
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20.")
        if not isinstance(pool_size, int) or not top_k <= pool_size <= 200:
            raise ValueError("pool_size must be >= top_k and <= 200.")
        if budget_hours is not None and (not math.isfinite(float(budget_hours)) or float(budget_hours) <= 0):
            raise ValueError("budget_hours must be positive and finite.")
        if budget_hours is not None:
            budget_hours = float(budget_hours)
        known = {canonical(s) for s in known_skills}
        gaps = [g for g in gaps if g["skill"] not in known]
        weights = {g["skill"]: g["weight"] for g in gaps}
        scores = {}
        # Per-gap retrieval prevents a long multi-skill query suppressing a niche requirement.
        for gap in gaps:
            query = gap["skill"] + " " + " ".join(ALIASES.get(gap["skill"], []))
            for idx, score in self._retrieve_one(query, method):
                scores[idx] = scores.get(idx, 0) + score * gap["weight"]
        maxscore = max(scores.values(), default=1)
        ranking = sorted(scores, key=lambda i: (-scores[i], self.courses[i]["id"]))[:pool_size]
        candidates = []
        for idx in ranking:
            course = self.courses[idx]
            covered = set(course["evidence"]) & set(weights)
            # Semantic relevance alone is not evidence that a course teaches a skill.
            if not covered:
                continue
            candidates.append({"course": course, "covered": covered, "retrieval_score": scores[idx] / maxscore})
        selected, covered, used = [], set(), 0.0
        trace = []
        excluded_unknown_duration = 0
        while candidates and len(selected) < top_k:
            eligible = []
            planned = known | {s for c in selected for s in c["skills"]}
            for item in candidates:
                hours = item["course"].get("duration_hours")
                if budget_hours is not None and (hours is None or used + hours > budget_hours):
                    continue
                gain = sum(weights[s] for s in item["covered"] - covered)
                repeat = sum(weights[s] for s in item["covered"] & covered)
                missing_prereqs = len(set(item["course"].get("prerequisites", [])) - planned)
                item["selection_score"] = gain + 0.15 * item["retrieval_score"] - 0.25 * repeat - 0.5 * missing_prereqs
                item["score_components"] = {"new_coverage": gain, "relevance": round(0.15 * item["retrieval_score"], 5),
                                            "redundancy_penalty": 0.25 * repeat,
                                            "prerequisite_penalty": 0.5 * missing_prereqs}
                if optimize and not item["covered"] - covered:
                    continue
                eligible.append(item)
            if not eligible:
                break
            if optimize:
                best = max(eligible, key=lambda x: (x["selection_score"], x["retrieval_score"]))
                if not best["covered"] - covered:
                    break  # Do not fill the list with redundant resources.
            else:
                best = eligible[0]
            new = sorted(best["covered"] - covered)
            course = best["course"]
            trace.append({"selection_step": len(selected) + 1, "resource_id": course["id"],
                          "new_skills": new, "components": best["score_components"],
                          "objective": round(best["selection_score"], 5),
                          "eligible_candidates": len(eligible)})
            selected.append({**course, "covered_gaps": sorted(best["covered"]),
                             "newly_covered_gaps": new, "retrieval_score": round(best["retrieval_score"], 5),
                             "score_components": best["score_components"],
                             "selection_score": round(best["selection_score"], 5)})
            covered |= best["covered"]
            used += course.get("duration_hours") or 0
            candidates.remove(best)
        # Order chosen resources using curator prerequisites; do not add unrequested courses.
        pending, ordered, available = selected[:], [], known.copy()
        presentation_covered = set()
        while pending:
            ready = next((c for c in pending if set(c.get("prerequisites", [])) <= available), None)
            item = ready or pending[0]
            missing = sorted(set(item.get("prerequisites", [])) - available)
            ordered.append({**item, "step": len(ordered) + 1, "unmet_prerequisites": missing,
                            "newly_covered_gaps": sorted(set(item["covered_gaps"]) - presentation_covered)})
            presentation_covered.update(item["covered_gaps"])
            # Completing a listed resource is an assumed planning step, not verified mastery.
            available.update(item["skills"])
            pending.remove(item)
        if budget_hours is not None:
            excluded_unknown_duration = sum(c.get("duration_hours") is None for c in self.courses)
        denominator = sum(weights.values())
        return {"gaps": gaps, "method": method,
                "backend": {"hybrid": "BM25 + TF-IDF (lexical RRF)",
                            "hybrid_dense": "BM25 + SentenceTransformer (RRF)"}.get(method, method),
                "dense_model": self.dense_model if method in {"dense", "hybrid_dense"} else None,
                "optimized": optimize, "recommendations": ordered,
                "selection_trace": trace,
                "redundancy_ratio": round(1 - len(covered) / sum(len(c["covered_gaps"]) for c in ordered), 4) if ordered else 0.0,
                "candidate_count": len(ranking),
                "covered_skills": sorted(covered), "uncovered_skills": sorted(set(weights) - covered),
                "weighted_coverage": round(sum(weights[s] for s in covered) / denominator, 4) if denominator else 0.0,
                "budget_hours": budget_hours,
                "known_duration_hours": used,
                "total_duration_hours": used if ordered and all(c.get("duration_hours") is not None for c in ordered) else None,
                "excluded_unknown_duration": excluded_unknown_duration,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "catalog": self.metadata,
                "notice": "Missing resume evidence is not proof of missing ability. Coverage means curated syllabus-topic coverage, not skill mastery. Prerequisites are editorial planning hints. No LLM-generated claims."}
