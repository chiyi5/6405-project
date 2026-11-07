"""
Lightweight NLP back-end with graceful fallbacks.

Every heavy dependency (spaCy, sentence-transformers, transformers) is optional.
If it is installed, it is used; otherwise a pure-python / scikit-learn fallback
is used so that the whole project still runs end-to-end (useful for unit tests
and for machines without a GPU).
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import spacy  # noqa: F401
    from spacy.matcher import PhraseMatcher
    HAS_SPACY = True
except Exception:  # pragma: no cover
    HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F401
    HAS_ST = True
except Exception:  # pragma: no cover
    HAS_ST = False


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@lru_cache(maxsize=1)
def _spacy_sentencizer():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str) -> list[str]:
    """Split one line of text into sentences (spaCy sentencizer if available)."""
    text = text.strip()
    if not text:
        return []
    if HAS_SPACY:
        return [s.text.strip() for s in _spacy_sentencizer()(text).sents if s.text.strip()]
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Phrase matching (case-insensitive, longest match wins, no overlaps)
# ---------------------------------------------------------------------------
class PhraseFinder:
    """Find dictionary phrases in text. Uses spaCy PhraseMatcher when available."""

    def __init__(self, phrases, label="PHRASE"):
        self.phrases = [p.strip() for p in phrases if str(p).strip()]
        self.label = label
        if HAS_SPACY:
            self._nlp = spacy.blank("en")
            self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
            self._matcher.add(label, [self._nlp.make_doc(p) for p in self.phrases])
        else:
            # Build one big alternation regex, longest phrases first
            ordered = sorted(set(self.phrases), key=len, reverse=True)
            body = "|".join(re.escape(p) for p in ordered)
            self._regex = re.compile(rf"(?<![\w+#]){'(?:' + body + ')' if body else '(?!x)x'}(?![\w+#])",
                                     re.IGNORECASE)

    def find(self, text: str) -> list[dict]:
        found = []
        if HAS_SPACY:
            doc = self._nlp(text)
            for _, s, e in self._matcher(doc):
                span = doc[s:e]
                found.append({"text": span.text, "start": span.start_char, "end": span.end_char})
        else:
            for m in self._regex.finditer(text):
                found.append({"text": m.group(0), "start": m.start(), "end": m.end()})
        return remove_overlaps(found)


def remove_overlaps(items: list[dict]) -> list[dict]:
    """Keep the longest non-overlapping spans, returned in text order."""
    items = sorted(items, key=lambda x: (-(x["end"] - x["start"]), x["start"]))
    selected = []
    for it in items:
        if not any(it["start"] < s["end"] and it["end"] > s["start"] for s in selected):
            selected.append(it)
    return sorted(selected, key=lambda x: x["start"])


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embedder:
    """
    Unified text encoder.

    name = "tfidf-char"              -> stateless hashed char n-gram vectors (no download)
    name = any sentence-transformers  -> e.g. "sentence-transformers/all-MiniLM-L6-v2",
                                         "BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2",
                                         "sentence-transformers/all-mpnet-base-v2"
    All outputs are L2-normalised, so cosine similarity == dot product.
    """

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.requested_name = name
        if name != "tfidf-char" and HAS_ST:
            self.name = name
            self._model = SentenceTransformer(name, device=device)
            self.kind = "sbert"
        else:
            if name != "tfidf-char":
                print(f"[Embedder] sentence-transformers not available -> falling back to tfidf-char "
                      f"(requested {name})")
            from sklearn.feature_extraction.text import HashingVectorizer
            self.name = "tfidf-char"
            self._model = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                            n_features=2 ** 18, norm="l2", alternate_sign=False)
            self.kind = "hash"
        # e5 models expect a prefix
        self._prefix = "query: " if "e5" in name.lower() else ""

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False):
        texts = [self._prefix + str(t) for t in texts]
        if not texts:
            return np.empty((0, 384))
        if self.kind == "sbert":
            return self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=show_progress_bar)
        return self._model.transform(texts)

    @staticmethod
    def cosine(a, b) -> np.ndarray:
        """Cosine similarity matrix for normalised vectors (dense or sparse)."""
        out = a @ b.T
        if hasattr(out, "toarray"):
            out = out.toarray()
        return np.asarray(out)
