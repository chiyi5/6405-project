"""Unit tests for the course benchmark, retrievers, metrics, planner and classic matchers.

    python -m unittest tests/test_benchmark.py -v      (~30 s, no model downloads)
"""
import math
import sys
import unittest
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_GENSIM = importlib.util.find_spec("gensim") is not None  # no gensim wheel on Python 3.14

from src.learning import ir_metrics as M  # noqa: E402
from src.learning import retrievers as R  # noqa: E402
from src.learning.benchmark import parse_coursera_tags, split_skills, tag_matches  # noqa: E402
from src.learning.planner import plan, plan_metrics  # noqa: E402

DOCS = ["Learn SQL joins and aggregate queries in PostgreSQL",
        "Docker containers: build images and run containers",
        "Kubernetes orchestration for Docker containers",
        "Python programming for beginners: lists, loops, functions",
        "Data visualisation with Tableau dashboards",
        "Advanced SQL window functions for analysts"]


class MetricTests(unittest.TestCase):
    def test_perfect_and_empty_rankings(self):
        rel = {"a": 1, "b": 1}
        self.assertAlmostEqual(M.ndcg_at(["a", "b", "c"], rel, 10), 1.0)
        self.assertEqual(M.ndcg_at(["c", "d"], rel, 10), 0.0)
        self.assertTrue(math.isnan(M.ndcg_at(["a"], {}, 10)))
        self.assertEqual(M.mrr_at(["c", "a"], rel), 0.5)
        self.assertAlmostEqual(M.ap_at(["a", "c", "b"], rel), (1 + 2 / 3) / 2)
        self.assertEqual(M.recall_at(["a", "c"], rel, 2), 0.5)

    def test_randomisation_test_and_holm(self):
        a = np.linspace(0.5, 1, 30)
        self.assertLess(M.paired_randomization(a, a - 0.2, n=2000), 0.01)
        self.assertEqual(M.paired_randomization(a, a), 1.0)
        adj = M.holm({"x": 0.01, "y": 0.04, "z": 0.5})
        self.assertAlmostEqual(adj["x"], 0.03)
        self.assertAlmostEqual(adj["y"], 0.08)
        self.assertGreaterEqual(adj["z"], adj["y"])

    def test_bootstrap_ci_contains_mean(self):
        v = np.random.default_rng(0).random(50)
        lo, hi = M.bootstrap_ci(v)
        self.assertLess(lo, v.mean())
        self.assertGreater(hi, v.mean())


class BenchmarkTests(unittest.TestCase):
    def test_tag_parsing_separates_domain_slugs(self):
        tags, dom = parse_coursera_tags("Drama  Comedy  writing  unix shells arts-and-humanities music-and-art")
        self.assertEqual(tags, ["Drama", "Comedy", "writing", "unix shells"])
        self.assertEqual(dom, ["arts-and-humanities", "music-and-art"])

    def test_tag_matching_is_token_based(self):
        self.assertTrue(tag_matches("python", "python programming"))
        self.assertFalse(tag_matches("sql", "nosql"))
        self.assertFalse(tag_matches("c", "c++"))
        self.assertFalse(tag_matches("machine learning", "applied machine learning in python"))

    def test_split_is_deterministic_and_disjoint(self):
        skills = [f"s{i}" for i in range(50)]
        a, b = split_skills(skills, 13), split_skills(skills, 13)
        self.assertEqual(a, b)
        self.assertEqual(sorted(set(a.values())), ["dev", "test", "train"])


class RetrieverTests(unittest.TestCase):
    def test_tokenizer_keeps_technical_tokens(self):
        self.assertEqual(R.tokenize("C++, C# and Node.js"), ["c++", "c#", "node.js"])

    def test_lexical_retrievers_rank_the_obvious_document_first(self):
        for ret in (R.BM25Retriever(), R.TfidfRetriever(), R.TfidfRetriever("char_wb", (3, 5))):
            ret.fit(DOCS)
            idx, _ = ret.rank(["docker images"], 3)
            self.assertEqual(idx[0][0], 1, ret.name)

    def test_lsa_and_fusion_shapes(self):
        lsa = R.LSARetriever(k=3).fit(DOCS * 3)
        bm = R.BM25Retriever().fit(DOCS * 3)
        for ret in (lsa, R.RRFFusion([bm, lsa]), R.LinearFusion([bm, lsa], [1, 1])):
            self.assertEqual(ret.score(["sql", "python"]).shape, (2, 18))

    @unittest.skipUnless(HAS_GENSIM, "gensim not installed")
    def test_word2vec_is_deterministic(self):
        a = R.EmbeddingRetriever("word2vec", w2v_kwargs={"min_count": 1, "epochs": 2}).fit(DOCS * 5).score(["sql"])
        b = R.EmbeddingRetriever("word2vec", w2v_kwargs={"min_count": 1, "epochs": 2}).fit(DOCS * 5).score(["sql"])
        np.testing.assert_allclose(a, b)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ret = R.BM25Retriever().fit(DOCS)

    def test_xquad_covers_low_priority_gap_instead_of_second_sql_course(self):
        gaps = [{"skill": "SQL", "priority": "REQUIRED"}, {"skill": "Docker", "priority": "PREFERRED"}]
        chosen = plan(gaps, self.ret, k=2, strategy="xquad")
        self.assertTrue({0, 5} & set(chosen) and {1, 2} & set(chosen), chosen)  # one SQL + one Docker course
        pergap = plan(gaps, self.ret, k=2, strategy="pergap")
        self.assertEqual(set(pergap), {0, 5})  # both SQL courses: the redundancy xQuAD removes

    def test_plan_metrics(self):
        gaps = [{"skill": "SQL", "priority": "REQUIRED"}, {"skill": "Docker", "priority": "PREFERRED"}]
        rel = {"SQL": {"d0", "d5"}, "Docker": {"d1", "d2"}}
        m = plan_metrics(["d0", "d5"], gaps, rel, 2)
        self.assertAlmostEqual(m["weighted_coverage"], 2 / 2.5)
        self.assertEqual(m["redundancy"], 0.5)
        self.assertEqual(m["mandatory_required_covered"], 1.0)
        self.assertTrue(math.isnan(plan_metrics([], [{"skill": "x", "priority": "PREFERRED"}], {}, 2)["mandatory_required_covered"]))

    def test_empty_gaps(self):
        self.assertEqual(plan([], self.ret), [])


class ClassicMatcherTests(unittest.TestCase):
    def test_pair_classifier_and_unsupervised_matchers(self):
        from src.classic_nlp import LDAMatcher, LSAMatcher, PairClassifier, WordVectorMatcher
        rng = np.random.default_rng(0)
        jds = [DOCS[i % 6] for i in range(60)]
        res = [DOCS[i % 6] if i % 3 else DOCS[(i + 1) % 6] for i in range(60)]
        lab = [2 if i % 3 else 0 for i in range(60)]
        df = pd.DataFrame({"jd_text": jds, "resume_text": res, "label": lab})
        for model in PairClassifier.MODELS:
            clf = PairClassifier(model).train(df)
            s = clf.score_pairs(jds, res)
            self.assertEqual(len(s), 60)
            self.assertGreater(np.corrcoef(s, lab)[0, 1], 0.5, model)
        self.assertEqual(len(PairClassifier("logreg").train(df).top_terms(3)), 6)
        matchers = [LSAMatcher(3).fit(DOCS * 4), LDAMatcher(3, max_iter=5).fit(DOCS * 4)]
        if HAS_GENSIM:
            matchers.append(WordVectorMatcher("word2vec", {"min_count": 1}).fit(DOCS * 4))
        for m in matchers:
            s = m.score_pairs(jds[:5], res[:5])
            self.assertEqual(len(s), 5)
            self.assertFalse(np.isnan(s).any())


if __name__ == "__main__":
    unittest.main()
