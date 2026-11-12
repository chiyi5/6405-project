"""Independent module correctness tests, including failure/abstention cases."""
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.learning.engine import LearningRecommender, gaps_from_match, load_catalog, parse_gaps
from learning_app import make_handler
from experiments.evaluate_learning import ndcg


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LearningRecommender()

    def test_aliases_and_strongest_priority_are_merged(self):
        gaps = parse_gaps("PostgreSQL | PREFERRED\nSQL | MANDATORY\nDocker")
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["weight"], 3)

    def test_adapter_excludes_non_skills_and_matched_items(self):
        details = [
            {"category": "Skill", "requirement": "Docker", "priority": "REQUIRED", "match_score": None},
            {"category": "Skill", "requirement": "SQL", "priority": "PREFERRED", "match_score": float("nan")},
            {"category": "Skill", "requirement": "Python", "match_score": 95},
            {"category": "Education", "requirement": "Masters", "match_score": 0},
            {"category": "Experience", "requirement": "5 years", "match_score": 0},
            {"category": "Skill", "requirement": "Git", "match_score": 0, "match_type": "DISABLED"},
        ]
        self.assertEqual({g["skill"] for g in gaps_from_match({"details": details})}, {"docker", "sql"})

    def test_unsupported_skill_abstains(self):
        result = self.engine.recommend(parse_gaps("Kubernetes\nRust"))
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["weighted_coverage"], 0)

    def test_known_skills_removed(self):
        result = self.engine.recommend(parse_gaps("Python\nDocker"), known_skills=["python programming"])
        self.assertEqual([g["skill"] for g in result["gaps"]], ["docker"])
        self.assertEqual(result["weighted_coverage"], 1)

    def test_multiple_distinct_skills_have_grounded_evidence(self):
        result = self.engine.recommend(parse_gaps("SQL\nDocker\nFastAPI"), known_skills=["python"])
        self.assertEqual(result["weighted_coverage"], 1)
        self.assertEqual(len(result["recommendations"]), 3)
        for r in result["recommendations"]:
            self.assertTrue(r["newly_covered_gaps"])
            for s in r["covered_gaps"]:
                self.assertIn(r["evidence"][s], r["source_excerpt"])

    def test_optimizer_stops_before_redundant_fill(self):
        gaps = parse_gaps("Python")
        plain = self.engine.recommend(gaps, optimize=False)
        diverse = self.engine.recommend(gaps, optimize=True)
        self.assertGreater(len(plain["recommendations"]), len(diverse["recommendations"]))
        self.assertEqual(len(diverse["recommendations"]), 1)

    def test_missing_sql_prerequisite_prefers_intro(self):
        result = self.engine.recommend(parse_gaps("SQL"))
        self.assertEqual(result["recommendations"][0]["id"], "sql-intro")
        self.assertEqual(result["recommendations"][0]["unmet_prerequisites"], [])

    def test_unknown_duration_is_not_zero(self):
        result = self.engine.recommend(parse_gaps("Docker"), budget_hours=5)
        self.assertEqual(result["recommendations"], [])
        self.assertIsNone(result["total_duration_hours"])
        self.assertGreater(result["excluded_unknown_duration"], 0)

    def test_budget_and_prerequisite_order_with_controlled_catalog(self):
        catalog = copy.deepcopy(load_catalog())
        for c in catalog["resources"]:
            c["duration_hours"] = 2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            engine = LearningRecommender(path)
            result = engine.recommend(parse_gaps("Python\nFastAPI\nDocker"), budget_hours=4)
            self.assertLessEqual(result["total_duration_hours"], 4)
            result = engine.recommend(parse_gaps("Python\nFastAPI"))
            steps = result["recommendations"]
            self.assertIn("python", steps[0]["skills"])
            self.assertEqual(steps[1]["unmet_prerequisites"], [])

    def test_no_silent_dense_fallback(self):
        with self.assertRaisesRegex(ValueError, "dense-model"):
            self.engine.recommend(parse_gaps("Docker"), method="dense")

    def test_hybrid_backend_is_honestly_labelled(self):
        r = self.engine.recommend(parse_gaps("SQL"), method="hybrid")
        self.assertIn("lexical", r["backend"])
        self.assertIsNone(r["dense_model"])

    def test_catalog_rejects_untraceable_evidence(self):
        catalog = copy.deepcopy(load_catalog())
        catalog["resources"][0]["evidence"]["python"] = "invented evidence"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact source"):
                LearningRecommender(path)

    def test_ndcg_ideal_uses_all_judged_resources(self):
        self.assertLess(ndcg(["weak"], {"strong": 3, "weak": 1}, 1), 1)
        self.assertEqual(ndcg(["strong"], {"strong": 3, "weak": 1}, 1), 1)
        self.assertIsNone(ndcg([], {"a": 0}, 5))

    def test_invalid_inputs(self):
        for budget in [0, -1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                self.engine.recommend(parse_gaps("SQL"), budget_hours=budget)
        with self.assertRaises(ValueError):
            parse_gaps("SQL | URGENT")
        with self.assertRaises(ValueError):
            self.engine.recommend(parse_gaps("SQL"), top_k=0)

    def test_json_roundtrip(self):
        r = self.engine.recommend(parse_gaps("SQL"))
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False))["covered_skills"], ["sql"])


class HttpTests(unittest.TestCase):
    """Exercise real HTTP parsing/handlers with an in-memory socket transport.

    This does not assert that a browser or a real network socket can connect.
    """
    @classmethod
    def setUpClass(cls):
        cls.handler = make_handler(LearningRecommender())

    def request(self, method, path, data=None, headers=None):
        body = json.dumps(data).encode() if data is not None else b""
        fields = {"Host": "127.0.0.1:8765", "Content-Length": str(len(body)), **(headers or {})}
        request = (f"{method} {path} HTTP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n").encode() + body

        class Socket:
            def __init__(self):
                self.sent = bytearray()
            def makefile(self, *_):
                return io.BytesIO(request)
            def sendall(self, payload):
                self.sent.extend(payload)

        sock = Socket()
        self.handler(sock, ("127.0.0.1", 12345), None)
        head, body = bytes(sock.sent).split(b"\r\n\r\n", 1)
        return int(head.split()[1]), body

    def post(self, path, data, headers=None):
        status, body = self.request("POST", path, data, headers)
        return status, json.loads(body)

    def test_page_and_catalog(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"SkillBridge", body)
        status, body = self.request("GET", "/api/catalog")
        self.assertEqual(len(json.loads(body)["resources"]), 12)

    def test_recommend_and_compare(self):
        status, result = self.post("/api/recommend", {"gaps": "SQL\nDocker"})
        self.assertEqual(status, 200)
        self.assertEqual(result["weighted_coverage"], 1)
        status, result = self.post("/api/compare", {"gaps": "SQL"})
        self.assertEqual(status, 200)
        self.assertEqual(len(result), 4)
        self.assertEqual([r["optimized"] for r in result], [False, False, False, True])

    def test_live_team_pipeline_and_report(self):
        status, match = self.post("/api/match", {
            "jd": "Requirements:\nPython, SQL, Docker, FastAPI, Git.",
            "resume": "Skills:\nPython\nProjects:\nBuilt a Python text processing script."})
        self.assertEqual(status, 200)
        self.assertIn("offline", match["backend"])
        status, plan = self.post("/api/recommend", {"match_result": match})
        self.assertEqual(status, 200)
        self.assertIn("docker", plan["covered_skills"])
        self.assertNotIn("python", [g["skill"] for g in plan["gaps"]])
        status, report = self.post("/api/export", {"plan": plan})
        self.assertEqual(status, 200)
        self.assertIn("https://docs.docker.com", report["markdown"])

    def test_empty_document_request_rejected(self):
        status, _ = self.post("/api/match", {"jd": "", "resume": "Python"})
        self.assertEqual(status, 400)

    def test_bad_input_and_cross_origin(self):
        for payload in [{"gaps": ""}, {"gaps": "SQL", "top_k": 100}, [1]]:
            status, _ = self.post("/api/recommend", payload)
            self.assertEqual(status, 400)
        status, _ = self.post("/api/recommend", {"gaps": "SQL"}, {"Origin": "https://example.com"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
