"""
Member D — JD–Resume matching models and evaluation.

Methods compared in the report
  D1  TfidfMatcher                cosine of TF-IDF vectors (lexical baseline)
  D2  BM25Matcher                 Okapi BM25, JD as query (lexical baseline)
  D3  EmbeddingMatcher(doc)       one embedding per document (MiniLM / mpnet / bge / e5 ...)
  D4  EmbeddingMatcher(maxsim)    JD sentences x resume sentences, mean of per-requirement max similarity
  D5  CrossEncoderMatcher         cross-encoder relevance score (ms-marco MiniLM)
  D6  PipelineMatcher             our full requirement-level pipeline (pipeline.MatchingPipeline)
  D7  fine_tune_biencoder()       bi-encoder fine-tuned on labelled fit pairs

Metrics: Spearman rho vs. ordinal gold label (No=0 / Potential=1 / Good=2), ROC-AUC (Good vs rest),
macro-F1 after learning cut-points with cross-validation, and NDCG@k for ranking resumes per JD.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .nlp_backend import Embedder, split_sentences

LABEL_MAP = {"no fit": 0, "potential fit": 1, "good fit": 2, "0": 0, "1": 1, "2": 2}


def _tok(text):
    return re.findall(r"[a-z0-9+#]+", text.lower())


def chunk_text(text: str, min_len: int = 15) -> list[str]:
    chunks = []
    for line in text.splitlines():
        for s in split_sentences(line):
            if len(s) >= min_len:
                chunks.append(s[:500])
    return chunks or [text[:500]]


class TfidfMatcher:
    name = "D1_tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._fitted = False

    def fit(self, corpus):
        self.vec.fit(corpus)
        self._fitted = True
        return self

    def score_pairs(self, jds, resumes) -> np.ndarray:
        if not self._fitted:
            self.fit(list(jds) + list(resumes))
        a, b = self.vec.transform(jds), self.vec.transform(resumes)
        return np.asarray(a.multiply(b).sum(1)).ravel()


class BM25Matcher:
    name = "D2_bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.idf, self.avgdl = {}, 1.0

    def fit(self, corpus):
        docs = [_tok(d) for d in corpus]
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.avgdl = np.mean([len(d) for d in docs]) if docs else 1.0
        return self

    def _score(self, query, doc):
        q, d = _tok(query), _tok(doc)
        tf, dl = Counter(d), len(d)
        s = 0.0
        for t in set(q):
            if t in tf:
                f = tf[t]
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def score_pairs(self, jds, resumes) -> np.ndarray:
        if not self.idf:
            self.fit(list(resumes))
        return np.array([self._score(j, r) for j, r in zip(jds, resumes)])


class EmbeddingMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", pooling: str = "maxsim",
                 embedder: Embedder | None = None):
        self.emb = embedder or Embedder(model_name)
        self.pooling = pooling
        short = model_name.split("/")[-1]
        self.name = f"D{'4' if pooling == 'maxsim' else '3'}_{short}_{pooling}"
        self._cache = {}

    def _enc(self, texts):
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.emb.encode(missing)):
                self._cache[t] = v
        vs = [self._cache[t] for t in texts]
        if hasattr(vs[0], "toarray"):
            from scipy.sparse import vstack
            return vstack(vs)
        return np.vstack(vs)

    def score_pairs(self, jds, resumes) -> np.ndarray:
        out = []
        for j, r in zip(jds, resumes):
            if self.pooling == "doc":
                out.append(float(Embedder.cosine(self._enc([j[:4000]]), self._enc([r[:4000]]))[0, 0]))
            else:
                S = Embedder.cosine(self._enc(chunk_text(j)), self._enc(chunk_text(r)))
                out.append(float(S.max(axis=1).mean()))
        return np.array(out)


class CrossEncoderMatcher:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_chars: int = 2000):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)
        self.max_chars = max_chars
        self.name = "D5_cross_encoder"

    def score_pairs(self, jds, resumes) -> np.ndarray:
        pairs = [(j[: self.max_chars], r[: self.max_chars]) for j, r in zip(jds, resumes)]
        return np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=float)


class PipelineMatcher:
    """Wrap our requirement-level pipeline as a pair scorer (JD parsing cached per JD)."""

    def __init__(self, pipeline, name: str = "D6_our_pipeline"):
        self.pipe, self.name = pipeline, name
        self._jd_cache = {}
        self.results = []          # full pipeline outputs, re-used by scoring (Member E)

    def score_pairs(self, jds, resumes) -> np.ndarray:
        out = []
        for i, (j, r) in enumerate(zip(jds, resumes)):
            if i and i % 100 == 0:
                print(f"    pipeline {i}/{len(jds)}")
            if j not in self._jd_cache:
                self._jd_cache[j] = self.pipe.extract_jd_requirements(j)
            res = self.pipe.match(j, r, jd_df=self._jd_cache[j])
            self.results.append(res)
            if "hybrid_score" in res:
                out.append(res["hybrid_score"])
            else:
                out.append(res["final_score"] if res["final_score"] is not None else np.nan)
        return np.array(out, dtype=float)


def fine_tune_biencoder(train_df: pd.DataFrame, base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                        output_dir: str = "models/biencoder_fit", epochs: int = 1, batch_size: int = 16,
                        max_chars: int = 2000):
    """
    D7: fine-tune a bi-encoder with CosineSimilarityLoss on (jd_text, resume_text, label) pairs.
    Gold label 0/1/2 is mapped to target similarity 0.0 / 0.5 / 1.0.
    Uses the classic sentence-transformers fit() API (works in v2.x and v3.x).
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader
    model = SentenceTransformer(base_model)
    model.max_seq_length = 512
    ex = [InputExample(texts=[j[:max_chars], r[:max_chars]], label=float(l) / 2.0)
          for j, r, l in zip(train_df.jd_text, train_df.resume_text, train_df.label)]
    dl = DataLoader(ex, shuffle=True, batch_size=batch_size)
    model.fit(train_objectives=[(dl, losses.CosineSimilarityLoss(model))], epochs=epochs, warmup_steps=50,
              show_progress_bar=True)
    model.save(output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def ndcg_at_k(rel: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    order = np.argsort(-scores)[:k]
    gains = (2 ** rel[order] - 1) / np.log2(np.arange(2, len(order) + 2))
    ideal = np.sort(rel)[::-1][:k]
    idcg = ((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return float(gains.sum() / idcg) if idcg > 0 else np.nan


def cv_macro_f1(scores: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    """Learn a 1-D score -> label mapping with cross-validated logistic regression, report macro-F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    k = int(min(5, pd.Series(y).value_counts().min()))
    if k < 2:
        return np.nan
    X = scores.reshape(-1, 1)
    X = (X - np.nanmean(X)) / (np.nanstd(X) + 1e-9)
    pred = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=1000), X, y,
                             cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    return float(f1_score(y, pred, average="macro"))


def evaluate_matchers(matchers: list, pairs: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    pairs: DataFrame with columns jd_id, resume_id, jd_text, resume_text, label (0/1/2)
    Returns (summary, per-pair scores).
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    pairs = pairs.reset_index(drop=True).copy()
    y = pairs["label"].astype(int).values
    scores_df = pairs[["jd_id", "resume_id", "label"]].copy()
    rows = []
    for m in matchers:
        if hasattr(m, "fit") and not getattr(m, "_fitted", True):
            m.fit(pd.concat([pairs.jd_text, pairs.resume_text]).drop_duplicates().tolist())
        import time as _t
        t0 = _t.time()
        try:
            s = np.asarray(m.score_pairs(pairs.jd_text.tolist(), pairs.resume_text.tolist()), dtype=float)
        except Exception as e:  # one failing matcher must not kill the comparison
            print(f"[matcher failed] {m.name}: {e!r}")
            continue
        print(f"  {m.name}: {_t.time() - t0:.0f}s")
        scores_df[m.name] = s
        ok = ~np.isnan(s)
        rho = spearmanr(s[ok], y[ok]).statistic if ok.sum() > 2 else np.nan
        try:
            auc = roc_auc_score((y[ok] == y.max()).astype(int), s[ok])
        except ValueError:
            auc = np.nan
        ndcgs = [ndcg_at_k(g.label.values.astype(float), g[m.name].values, k)
                 for _, g in scores_df[ok].groupby("jd_id") if len(g) >= 2 and g.label.nunique() > 1]
        rows.append({"method": m.name, "spearman": round(float(rho), 3), "auc_goodfit": round(float(auc), 3),
                     "macro_f1_cv": round(cv_macro_f1(s[ok], y[ok]), 3),
                     f"ndcg@{k}": round(float(np.nanmean(ndcgs)), 3) if ndcgs else np.nan,
                     "n_pairs": int(ok.sum())})
    return pd.DataFrame(rows), scores_df
