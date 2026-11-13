"""Interchangeable retrievers for course recommendation (one class per course topic).

Every retriever implements ``fit(texts)`` and ``score(queries) -> (n_queries, n_docs)``.
They map onto the EE6405 syllabus:

=====================  ==========================================  ==================
Retriever              Idea                                        Course topic
=====================  ==========================================  ==================
BM25Retriever          probabilistic term weighting (k1, b)         term weighting
TfidfRetriever         TF-IDF cosine, word or char n-grams          BoW / TF-IDF
LSARetriever           truncated SVD of TF-IDF (latent semantics)   feature extraction
LDARetriever           topic distributions, Hellinger similarity    topic modelling
EmbeddingRetriever     IDF-weighted mean of word vectors            word embeddings
                       (pre-trained GloVe or Word2Vec trained here)
DenseRetriever         Sentence-BERT bi-encoder (optional)          transformers / PLMs
CrossEncoderReranker   joint query-doc transformer (optional)       attention
RRFFusion / Linear     rank fusion of several retrievers            ensembles
LTRReranker            learned combination of retriever features    supervised ML
=====================  ==========================================  ==================

Heavy dependencies (sentence-transformers, gensim models) are imported lazily, so the
lexical baselines run with scikit-learn only.
"""
from __future__ import annotations

import re
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

TOKEN = re.compile(r"[\w#][\w+#.-]*[\w+#]|\w")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(str(text).lower()) if t not in ENGLISH_STOP_WORDS]


class Retriever:
    name = "base"

    def fit(self, texts):
        raise NotImplementedError

    def score(self, queries) -> np.ndarray:
        raise NotImplementedError

    def rank(self, queries, k=100):
        s = self.score(queries)
        k = min(k, s.shape[1])
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(s, top, 1).argsort(1)[:, ::-1]
        return np.take_along_axis(top, order, 1), s


# ---------------------------------------------------------------------------
# Lexical
# ---------------------------------------------------------------------------
class BM25Retriever(Retriever):
    def __init__(self, k1=1.2, b=0.75, ngram_range=(1, 1)):
        self.k1, self.b, self.ngram_range = k1, b, ngram_range
        self.name = f"BM25(k1={k1},b={b})"

    def fit(self, texts):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=self.ngram_range)
        tf = self.vec.fit_transform(texts).tocsr().astype(float)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(1)).ravel()
        denom_doc = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + denom_doc[tf.row])
        self.W = sparse.csr_matrix((w * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        return self

    def score(self, queries):
        q = self.vec.transform(queries)
        q.data[:] = 1.0  # binary query terms
        return np.asarray((q @ self.W.T).todense())


class TfidfRetriever(Retriever):
    def __init__(self, analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1):
        self.kw = dict(analyzer=analyzer, ngram_range=ngram_range, sublinear_tf=sublinear_tf, min_df=min_df)
        if analyzer == "word":
            self.kw.update(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.name = f"TF-IDF({analyzer},{ngram_range[0]}-{ngram_range[1]})"

    def fit(self, texts):
        self.vec = TfidfVectorizer(**self.kw)
        self.D = self.vec.fit_transform(texts)
        return self

    def score(self, queries):
        return np.asarray((self.vec.transform(queries) @ self.D.T).todense())


class LSARetriever(Retriever):
    """Latent Semantic Analysis: rank-k SVD of the TF-IDF matrix, cosine in latent space."""

    def __init__(self, k=300, seed=0):
        self.k, self.seed = k, seed
        self.name = f"LSA(k={k})"

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, min_df=2)
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(self.k, random_state=self.seed).fit(X)
        self.D = normalize(self.svd.transform(X))
        return self

    def score(self, queries):
        return normalize(self.svd.transform(self.tfidf.transform(queries))) @ self.D.T


class LDARetriever(Retriever):
    """Topic-model retrieval: similarity of LDA topic mixtures (1 - Hellinger distance)."""

    def __init__(self, k=100, seed=0, max_iter=20):
        self.k, self.seed, self.max_iter = k, seed, max_iter
        self.name = f"LDA(k={k})"

    def fit(self, texts):
        from sklearn.decomposition import LatentDirichletAllocation
        self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=3, max_df=0.3)
        X = self.cv.fit_transform(texts)
        self.lda = LatentDirichletAllocation(self.k, learning_method="online", random_state=self.seed,
                                             max_iter=self.max_iter, n_jobs=1).fit(X)  # n_jobs=-1 is very slow on Windows (process spawn)
        self.D = np.sqrt(self.lda.transform(X))
        return self

    def score(self, queries):
        Q = np.sqrt(self.lda.transform(self.cv.transform(queries)))
        # Bhattacharyya coefficient = 1 - Hellinger^2 for unit-sum distributions
        return Q @ self.D.T

    def top_words(self, n=8):
        vocab = np.array(self.cv.get_feature_names_out())
        return [list(vocab[c.argsort()[::-1][:n]]) for c in self.lda.components_]


class _Vectors:
    """Minimal stand-in for gensim KeyedVectors (key_to_index, vector_size, kv[list])."""

    def __init__(self, words, matrix):
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors, self.vector_size = matrix, matrix.shape[1]

    def __getitem__(self, keys):
        return self.vectors[[self.key_to_index[k] for k in keys]]


def load_pretrained_vectors(name="glove-wiki-gigaword-100", cache=None):
    """Load a gensim-data model without gensim: download the word2vec-text .gz once and parse it."""
    import gzip
    import urllib.request
    from pathlib import Path
    cache = Path(cache or Path.home() / ".cache" / "skillbridge-vectors")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.gz"
    if not path.exists():
        url = f"https://github.com/RaRe-Technologies/gensim-data/releases/download/{name}/{name}.gz"
        urllib.request.urlretrieve(url, path)
    words, rows = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) != 2:  # plain GloVe text without a header line
            words.append(first[0]); rows.append(np.asarray(first[1:], dtype=np.float32))
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0]); rows.append(np.asarray(parts[1:], dtype=np.float32))
    return _Vectors(words, np.vstack(rows))


class EmbeddingRetriever(Retriever):
    """Bag-of-embeddings: IDF-weighted average of word vectors, cosine similarity.

    ``vectors`` is a gensim KeyedVectors (e.g. GloVe), or the string ``"word2vec"`` to
    train a skip-gram Word2Vec model on the course corpus itself.
    """

    def __init__(self, vectors="glove-wiki-gigaword-100", idf_weight=True, w2v_kwargs=None):
        self.vectors, self.idf_weight = vectors, idf_weight
        self.w2v_kwargs = w2v_kwargs or {}
        label = vectors if isinstance(vectors, str) else "custom"
        self.name = f"Emb({label}{',idf' if idf_weight else ''})"

    def _embed(self, texts):
        out = np.zeros((len(texts), self.kv.vector_size), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in tokenize(t) if w in self.kv.key_to_index]
            if not toks:
                continue
            w = np.array([self.idf.get(x, self.max_idf) if self.idf_weight else 1.0 for x in toks], dtype=np.float32)
            out[i] = (self.kv[toks] * w[:, None]).sum(0) / w.sum()
        return normalize(out)

    def fit(self, texts):
        if self.vectors == "word2vec":
            from gensim.models import Word2Vec
            kw = dict(vector_size=200, window=5, min_count=3, sg=1, epochs=10, workers=1, seed=0)  # workers=1: deterministic
            kw.update(self.w2v_kwargs)
            self.kv = Word2Vec([tokenize(t) for t in texts], **kw).wv
        elif isinstance(self.vectors, str):
            try:
                import gensim.downloader as api
                self.kv = api.load(self.vectors)
            except ImportError:  # e.g. no gensim wheel for this Python version
                self.kv = load_pretrained_vectors(self.vectors)
        else:
            self.kv = self.vectors
        tf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit(texts)
        self.idf = dict(zip(tf.get_feature_names_out(), tf.idf_))
        self.max_idf = max(self.idf.values())
        self.D = self._embed(texts)
        return self

    def score(self, queries):
        return self._embed(queries) @ self.D.T


# ---------------------------------------------------------------------------
# Neural (optional: needs sentence-transformers + model download)
# ---------------------------------------------------------------------------
class DenseRetriever(Retriever):
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2", batch_size=64, max_seq_length=256,
                 query_prefix="", doc_prefix=""):
        self.model_name, self.batch_size, self.max_seq_length = model, batch_size, max_seq_length
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.name = f"Dense({str(model).split('/')[-1]})"

    def fit(self, texts):
        from sentence_transformers import SentenceTransformer
        self.model = self.model_name if not isinstance(self.model_name, str) else SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length
        self.D = self.model.encode([self.doc_prefix + t for t in texts], batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
        return self

    def score(self, queries):
        Q = self.model.encode([self.query_prefix + q for q in queries], batch_size=self.batch_size,
                              normalize_embeddings=True, show_progress_bar=False)
        return Q @ self.D.T


class CrossEncoderReranker(Retriever):
    """Re-scores the top ``depth`` candidates of a first-stage retriever with a cross-encoder."""

    def __init__(self, first_stage: Retriever, model="cross-encoder/ms-marco-MiniLM-L-6-v2", depth=50):
        self.first, self.model_name, self.depth = first_stage, model, depth
        self.name = f"{first_stage.name}>>CE({model.split('/')[-1]},{depth})"

    def fit(self, texts):
        from sentence_transformers import CrossEncoder
        self.texts = list(texts)
        self.ce = CrossEncoder(self.model_name, max_length=256)
        return self

    def score(self, queries):
        base = self.first.score(queries)
        out = np.full_like(base, -1e9, dtype=float)
        for i, q in enumerate(queries):
            top = np.argsort(-base[i])[: self.depth]
            s = self.ce.predict([(q, self.texts[j]) for j in top], show_progress_bar=False)
            out[i, top] = s
        return out


# ---------------------------------------------------------------------------
# Fusion and learning to rank
# ---------------------------------------------------------------------------
class RRFFusion(Retriever):
    """Reciprocal rank fusion (Cormack et al., 2009): sum 1/(k + rank)."""

    def __init__(self, members, k=60, weights=None):
        self.members, self.k = members, k
        self.weights = weights or [1.0] * len(members)
        self.name = "RRF(" + "+".join(m.name.split("(")[0] for m in members) + ")"

    def fit(self, texts):
        return self  # members are fitted by the caller (shared across systems)

    def score(self, queries):
        total = None
        for m, w in zip(self.members, self.weights):
            s = m.score(queries)
            ranks = (-s).argsort(1).argsort(1) + 1
            part = w / (self.k + ranks)
            total = part if total is None else total + part
        return total


def _zscore(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


class LinearFusion(Retriever):
    def __init__(self, members, weights):
        self.members, self.weights = members, weights
        self.name = "Lin(" + "+".join(f"{w:g}*{m.name.split('(')[0]}" for m, w in zip(members, weights)) + ")"

    def fit(self, texts):
        return self

    def score(self, queries):
        return sum(w * _zscore(m.score(queries)) for m, w in zip(self.members, self.weights))


class LTRReranker(Retriever):
    """Pointwise learning-to-rank over per-query z-scored retriever scores + doc priors.

    Trained on the *train* split queries with silver labels; the candidate pool is the
    union of each member's top ``depth``. Model: logistic regression or gradient boosting.
    """

    def __init__(self, members, doc_features=None, model="logreg", depth=100, C=1.0, seed=0):
        self.members, self.depth, self.model_kind, self.C, self.seed = members, depth, model, C, seed
        self.doc_features = doc_features  # (n_docs, f) array of query-independent features
        self.name = f"LTR({model},C={C})" if model == "logreg" else f"LTR({model})"

    def fit(self, texts):
        return self

    def _features(self, queries):
        zs = [_zscore(m.score(queries)) for m in self.members]
        pools = []
        for i in range(len(queries)):
            pool = set()
            for z in zs:
                pool |= set(np.argsort(-z[i])[: self.depth])
            pools.append(np.array(sorted(pool)))
        feats = []
        for i, pool in enumerate(pools):
            f = np.stack([z[i, pool] for z in zs], 1)
            if self.doc_features is not None:
                f = np.hstack([f, self.doc_features[pool]])
            feats.append(f)
        return pools, feats, zs[0].shape[1]

    def train(self, queries, labels):
        """``labels``: list of sets of relevant doc indices, aligned with ``queries``."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        pools, feats, _ = self._features(queries)
        X = np.vstack(feats)
        y = np.concatenate([[int(j in lab) for j in pool] for pool, lab in zip(pools, labels)])
        if self.model_kind == "logreg":
            self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
        else:
            self.clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def score(self, queries):
        pools, feats, n = self._features(queries)
        out = np.full((len(queries), n), -1e9)
        probs = self.clf.predict_proba(np.vstack(feats))[:, 1]  # one batched call (GBT overhead is per call)
        start = 0
        for i, pool in enumerate(pools):
            out[i, pool] = probs[start:start + len(pool)]
            start += len(pool)
        return out


def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t
