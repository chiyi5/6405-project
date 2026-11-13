"""
Member D/E extension — traditional-NLP matchers and supervised pair classifiers.

Fills the gap between the lexical baselines (D1 TF-IDF, D2 BM25) and the transformer
matchers (D3–D7) with the rest of the EE6405 "traditional" toolbox, so the report can
compare every family on the same 600 test pairs:

  D9   LSAMatcher            TF-IDF -> truncated SVD (latent semantic analysis), cosine
  D10  LDAMatcher            LDA topic mixtures, 1 - Hellinger distance
  D11  WordVectorMatcher     IDF-weighted mean word vectors: GloVe (pre-trained) or
                             Word2Vec skip-gram trained on the 3,000 training pairs
  D12  PairClassifier        supervised: pair features -> Logistic Regression /
                             Linear SVM / Multinomial NB / Random Forest, trained on
                             the training pairs (3-class fit label); the score is
                             E[label] = P(potential) + 2 P(good)

All matchers follow the src.matching interface: ``name``, ``fit(corpus)``,
``score_pairs(jds, resumes)``; unsupervised ones are fitted on the *training* texts
(``_fitted=True`` stops evaluate_matchers from refitting on test texts).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

_TOK = re.compile(r"[a-z0-9][a-z0-9+#.-]*[a-z0-9+#]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(str(text).lower()) if t not in ENGLISH_STOP_WORDS]


def _rowcos(A, B):
    A, B = normalize(A), normalize(B)
    return np.asarray(A.multiply(B).sum(1)).ravel() if hasattr(A, "multiply") else (A * B).sum(1)


class LSAMatcher:
    def __init__(self, k: int = 200, seed: int = 0):
        self.k, self.seed, self._fitted = k, seed, False
        self.name = f"D9_lsa_k{k}"

    def fit(self, corpus):
        from sklearn.decomposition import TruncatedSVD
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     sublinear_tf=True, min_df=2, max_df=0.5)
        X = self.tfidf.fit_transform(corpus)
        self.svd = TruncatedSVD(min(self.k, X.shape[1] - 1), random_state=self.seed).fit(X)
        self._fitted = True
        return self

    def embed(self, texts):
        return self.svd.transform(self.tfidf.transform(texts))

    def score_pairs(self, jds, resumes):
        return _rowcos(self.embed(jds), self.embed(resumes))


class LDAMatcher:
    def __init__(self, k: int = 50, seed: int = 0, max_iter: int = 20):
        self.k, self.seed, self.max_iter, self._fitted = k, seed, max_iter, False
        self.name = f"D10_lda_k{k}"

    def fit(self, corpus):
        from sklearn.decomposition import LatentDirichletAllocation
        self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=3, max_df=0.4)
        X = self.cv.fit_transform(corpus)
        self.lda = LatentDirichletAllocation(self.k, learning_method="online", max_iter=self.max_iter,
                                             random_state=self.seed, n_jobs=-1).fit(X)
        self._fitted = True
        return self

    def topics(self, texts):
        return self.lda.transform(self.cv.transform(texts))

    def score_pairs(self, jds, resumes):
        a, b = np.sqrt(self.topics(jds)), np.sqrt(self.topics(resumes))
        return (a * b).sum(1)  # Bhattacharyya coefficient = 1 - Hellinger^2

    def top_words(self, n: int = 10):
        vocab = np.array(self.cv.get_feature_names_out())
        return [" ".join(vocab[c.argsort()[::-1][:n]]) for c in self.lda.components_]


class WordVectorMatcher:
    """IDF-weighted average word embedding. ``vectors``: gensim model name, "word2vec", or KeyedVectors."""

    def __init__(self, vectors="glove-wiki-gigaword-100", w2v_kwargs: dict | None = None):
        self.vectors, self.w2v_kwargs, self._fitted = vectors, w2v_kwargs or {}, False
        label = "word2vec_trained" if vectors == "word2vec" else (vectors if isinstance(vectors, str) else "custom")
        self.name = f"D11_{label}"

    def fit(self, corpus):
        if self.vectors == "word2vec":
            from gensim.models import Word2Vec
            kw = dict(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=1, seed=0)  # workers=1: deterministic
            kw.update(self.w2v_kwargs)
            self.kv = Word2Vec([tokenize(t) for t in corpus], **kw).wv
        elif isinstance(self.vectors, str):
            from src.learning.retrievers import load_pretrained_vectors
            self.kv = load_pretrained_vectors(self.vectors)
        else:
            self.kv = self.vectors
        tf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit(corpus)
        self.idf = dict(zip(tf.get_feature_names_out(), tf.idf_))
        self.max_idf = max(self.idf.values())
        self._fitted = True
        return self

    def embed(self, texts):
        out = np.zeros((len(texts), self.kv.vector_size), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in tokenize(t) if w in self.kv.key_to_index]
            if toks:
                w = np.array([self.idf.get(x, self.max_idf) for x in toks], dtype=np.float32)
                out[i] = (self.kv[toks] * w[:, None]).sum(0) / w.sum()
        return out

    def score_pairs(self, jds, resumes):
        return _rowcos(self.embed(jds), self.embed(resumes))


class PairClassifier:
    """Supervised fit classifier on pair features (course topic: ML classifiers).

    Features per (JD, resume):
      * TF-IDF of the JD, TF-IDF of the resume, and their element-wise product
        (shared-term evidence), all from one vocabulary fitted on training texts;
      * 5 dense similarity features: TF-IDF cosine, Jaccard of token sets, coverage of JD
        tokens by the resume, and log lengths.
    """

    MODELS = ("logreg", "linear_svm", "naive_bayes", "random_forest")

    def __init__(self, model: str = "logreg", C: float = 1.0, max_features: int = 20000, seed: int = 0):
        assert model in self.MODELS, model
        self.model, self.C, self.max_features, self.seed = model, C, max_features, seed
        self.name = f"D12_{model}" + (f"_C{C:g}" if model in ("logreg", "linear_svm") else "")
        self._fitted = True  # trained via .train(), not via evaluate_matchers' fit()

    def fit(self, corpus):
        return self

    def _dense(self, J, R, jds, resumes):
        cos = _rowcos(J, R)
        js = [set(tokenize(t)) for t in jds]
        rs = [set(tokenize(t)) for t in resumes]
        jac = np.array([len(a & b) / max(1, len(a | b)) for a, b in zip(js, rs)])
        cov = np.array([len(a & b) / max(1, len(a)) for a, b in zip(js, rs)])
        return np.c_[cos, jac, cov, np.log1p([len(a) for a in js]), np.log1p([len(b) for b in rs])]

    def _features(self, jds, resumes):
        from scipy import sparse
        J, R = self.vec.transform(jds), self.vec.transform(resumes)
        dense = self._dense(J, R, jds, resumes)
        if self.model == "naive_bayes":  # NB needs non-negative counts-like input
            return sparse.hstack([J, R, J.multiply(R)]).tocsr()
        return sparse.hstack([J, R, J.multiply(R), sparse.csr_matrix(self.scaler.transform(dense))]).tocsr()

    def train(self, train_df: pd.DataFrame):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        jds, res = train_df.jd_text.tolist(), train_df.resume_text.tolist()
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True,
                                   min_df=2, max_features=self.max_features).fit(jds + res)
        J, R = self.vec.transform(jds), self.vec.transform(res)
        self.scaler = StandardScaler().fit(self._dense(J, R, jds, res))
        X, y = self._features(jds, res), train_df.label.astype(int).values
        clf = {"logreg": lambda: LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced"),
               "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=self.C, class_weight="balanced"), cv=3),
               "naive_bayes": lambda: MultinomialNB(alpha=0.5),
               "random_forest": lambda: RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1,
                                                               class_weight="balanced", random_state=self.seed)}
        self.clf = clf[self.model]().fit(X, y)
        return self

    def predict_proba(self, jds, resumes):
        return self.clf.predict_proba(self._features(jds, resumes))

    def predict(self, jds, resumes):
        return self.clf.predict(self._features(jds, resumes))

    def score_pairs(self, jds, resumes):
        p = self.predict_proba(jds, resumes)
        return p @ np.array(self.clf.classes_, dtype=float)

    def top_terms(self, n: int = 15) -> pd.DataFrame:
        """Most positive / negative weights of the 'good fit' class (logreg only): interpretation."""
        if self.model != "logreg":
            raise ValueError("coefficients only for logreg")
        vocab = list(self.vec.get_feature_names_out())
        names = ([f"JD:{v}" for v in vocab] + [f"CV:{v}" for v in vocab] + [f"BOTH:{v}" for v in vocab]
                 + ["sim:tfidf_cos", "sim:jaccard", "sim:jd_coverage", "len:jd", "len:cv"])
        coefs = self.clf.coef_
        coef = coefs[0] if coefs.shape[0] == 1 else coefs[list(self.clf.classes_).index(max(self.clf.classes_))]
        order = np.argsort(coef)
        rows = [{"feature": names[i], "weight": coef[i], "direction": "good-fit"} for i in order[::-1][:n]]
        rows += [{"feature": names[i], "weight": coef[i], "direction": "no-fit"} for i in order[:n]]
        return pd.DataFrame(rows)


def sentence_occlusion(score_fn, jd: str, resume: str, max_sentences: int = 40) -> pd.DataFrame:
    """Model-agnostic explanation: drop each resume sentence and record the score change.

    ``score_fn(jd, resume) -> float``. Positive delta = the sentence supported the match.
    Works for every matcher (lexical, topic, embedding, pipeline), so explanations can be
    compared across model families.
    """
    from .nlp_backend import split_sentences
    sents = [s for line in resume.splitlines() for s in split_sentences(line) if len(s) > 15][:max_sentences]
    base = score_fn(jd, resume)
    rows = []
    for s in sents:
        rows.append({"sentence": s[:200], "delta": base - score_fn(jd, resume.replace(s, " "))})
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
