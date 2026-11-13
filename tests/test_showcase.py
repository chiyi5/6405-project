"""Exercise upload, ranking and showcase routes through the real HTTP parser.

Uses the existing in-memory transport; a passing test does not prove socket reachability.
"""
import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_learning as learning_tests
from showcase_app import make_showcase_handler
from src.learning import LearningRecommender


class ShowcaseTests(learning_tests.HttpTests):
    @classmethod
    def setUpClass(cls):
        cls.handler = make_showcase_handler(LearningRecommender())

    def test_bootstrap_and_static_whitelist(self):
        status, body = self.request("GET", "/api/showcase")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertNotIn("jobs", data)
        self.assertNotIn("resumes", data)
        self.assertTrue(data["experiments"]["skills"]["rows"])
        for key in ("classic", "courses", "plans"):
            self.assertTrue(data["experiments"][key]["rows"], key)
        for route in ("/", "/app.js", "/style.css", "/locales.js", "/i18n.js"):
            self.assertEqual(self.request("GET", route)[0], 200)
        self.assertEqual(self.request("GET", "/../README.md")[0], 404)

    def test_upload_decodes_without_persisting_user_filename(self):
        status, data = self.post("/api/read-document", {"name": "../../resume.txt", "content": base64.b64encode(b"Skills: Python").decode()})
        self.assertEqual(status, 200)
        self.assertEqual(data["text"], "Skills: Python")
        for payload in [{"name": "code.exe", "content": ""}, {"name": "cv.txt", "content": "***"}, {"name": "cv.txt", "content": ""}]:
            self.assertEqual(self.post("/api/read-document", payload)[0], 400)
        self.assertEqual(self.post("/api/rank", {}, {"Origin": "https://other.example"})[0], 403)

    def test_real_pipeline_ranking_and_validation(self):
        status, data = self.post("/api/rank", {"jd": "Requirements:\nPython, Docker.", "resumes": [
            {"name": "weak", "text": "Skills:\nExcel"},
            {"name": "strong", "text": "Skills:\nPython, Docker"}]})
        self.assertEqual(status, 200)
        self.assertEqual(data["ranking"][0]["name"], "strong")
        self.assertIn("details", data["ranking"][0])
        for resumes in ([], [{}] * 13, ["invalid"]):
            self.assertEqual(self.post("/api/rank", {"jd": "Python", "resumes": resumes})[0], 400)


if __name__ == "__main__":
    unittest.main()
