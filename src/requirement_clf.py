"""
Member C — Requirement priority classification (MANDATORY / REQUIRED / PREFERRED / UNSPECIFIED).

Methods compared in the report
  C1  RuleClassifier(use_section_headers=False)   original regex baseline
  C2  RuleClassifier(use_section_headers=True)    + priority inherited from section header
                                                   ("Nice to have:", "Requirements:" ...)
  C3  TfidfLRClassifier                            TF-IDF (word+char) + Logistic Regression
  C4  ZeroShotNLIClassifier                        facebook/bart-large-mnli zero-shot
  C5  BertClassifier                               fine-tuned DistilBERT / BERT

Evaluation: accuracy, macro-F1, per-class report, confusion matrix; trainable models via
stratified k-fold cross-validation (the dataset is small).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .nlp_backend import split_sentences

LABELS_4 = ["MANDATORY", "REQUIRED", "PREFERRED", "UNSPECIFIED"]
LABELS_3 = ["REQUIRED", "PREFERRED", "UNSPECIFIED"]  # MANDATORY merged into REQUIRED

PREFERRED_PATTERNS = [r"\bpreferred\b", r"\bpreferably\b", r"\bwould be (?:preferred|advantageous|a plus|an advantage)\b",
                      r"\badvantageous\b", r"\ban (?:added )?advantage\b", r"\ba plus\b", r"\bbonus\b",
                      r"\bnice to have\b", r"\bgood to have\b", r"\bdesirable\b", r"\bnot mandatory\b",
                      r"\bnot required\b", r"\boptional\b", r"\bnot essential\b", r"\bhighly regarded\b",
                      r"\bwill be given preference\b", r"\bideally\b", r"\bis a strong plus\b", r"\bdesired\b"]
REQUIRED_PATTERNS = [r"\brequired\b", r"\bmust\b", r"\bmandatory\b", r"\bessential\b", r"\bminimum\b", r"\bat least\b",
                     r"\bneed(?:s)? to\b", r"\bnecessary\b", r"\bexpected to have\b", r"\bproficiency in\b",
                     r"\bproficient in\b", r"\brequires?\b", r"\bshould (?:have|possess|be able)\b",
                     r"\bcandidates? should\b", r"\byou (?:have|will have|bring)\b", r"\bstrong (?:knowledge|experience)\b"]
MANDATORY_PATTERNS = [r"\bwithout .*? (?:not eligible|ineligible)\b", r"\bnot eligible\b", r"\bineligible\b",
                      r"\bwill not be considered\b", r"\bapplications? will be rejected\b", r"\bnon[- ]negotiable\b",
                      r"\bstrict eligibility requirement\b", r"\bonly (?:singaporeans?|citizens|PRs?)\b",
                      r"\bmust (?:be|hold) (?:a )?(?:singapore|citizen|permanent resident)\b", r"\bsecurity clearance\b"]

HEADER_HINTS = [
    (r"nice to have|preferred|bonus|good to have|desirable|plus points?|advantageous|optional|ideal(?:ly)?|"
     r"would be great", "PREFERRED"),
    (r"must[- ]haves?|minimum qualifications?|basic qualifications?|requirements?|required|what you need|"
     r"what we(?:'| a)re looking for|qualifications?|who you are|you have|skills and experience|eligibility|"
     r"skills?|competenc",
     "REQUIRED"),
    (r"responsibilit|what you(?:'| wi)ll do|duties|the role|about (?:us|the)|benefits|what we offer|perks|"
     r"job description|overview", "UNSPECIFIED"),
]


def contains_pattern(text, patterns) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def header_priority(header: str | None) -> str | None:
    if not isinstance(header, str) or not header:
        return None
    for pat, lab in HEADER_HINTS:
        if re.search(pat, header, re.I):
            return lab
    return None


def split_requirement_clauses(text: str) -> list[str]:
    parts = re.split(r"\s*(?:;|\bwhile\b|\bwhereas\b)\s*", text, flags=re.I)
    return [p.strip(" ,.;") for p in parts if p.strip(" ,.;")]


_HEADER_LINE = re.compile(r"^[A-Z][^.!?]{1,60}:$|^(?:[A-Z][a-z]+\s?){1,5}$")


_HEADER_WORDS = re.compile(
    r"\b(about|role|responsibilit\w*|requirements?|qualifications?|skills?|competenc\w*|offer|apply|benefits?|"
    r"perks|statement|overview|introduction|summary|duties|what|who|why|company|team|position|job description|"
    r"preferred|nice to have|bonus|education|experience|must[- ]haves?)\b", re.I)


def is_header_line(line: str) -> bool:
    """A section heading: short, no sentence punctuation, and either ends with ':' or uses a heading word."""
    words = line.split()
    if not words or len(words) > 8 or re.search(r"[.!?;,]$", line) or re.search(r"\d", line):
        return False
    if line.endswith(":"):
        return True
    return len(words) <= 6 and bool(_HEADER_WORDS.search(line)) and line[0].isupper()


def split_with_sections(text: str) -> list[tuple[str, str | None]]:
    """Return [(sentence, current_section_header)] — header lines themselves are not returned.

    Inline headers "Strong ML capability: Proficient in Python ..." are kept in the sentence.
    """
    out, header = [], None
    for raw in text.splitlines():
        raw = raw.strip()
        is_bullet = bool(re.match(r"^[•●▪◦‣\-–*]\s*", raw))
        line = re.sub(r"^[•●▪◦‣\-–*]\s*", "", raw)
        if not line:
            continue
        if not is_bullet and is_header_line(line):
            header = line.rstrip(":")
            continue
        for s in split_sentences(line):
            out.append((s, header))
    return out


class RuleClassifier:
    name = "C1_rules"

    def __init__(self, use_section_headers: bool = True, merge_mandatory: bool = False):
        self.use_section_headers = use_section_headers
        self.merge_mandatory = merge_mandatory
        if use_section_headers:
            self.name = "C2_rules+sections"

    def fit(self, *a, **k):
        return self

    def predict_one(self, text: str, header: str | None = None) -> str:
        if contains_pattern(text, PREFERRED_PATTERNS):
            lab = "PREFERRED"
        elif contains_pattern(text, MANDATORY_PATTERNS):
            lab = "MANDATORY"
        elif contains_pattern(text, REQUIRED_PATTERNS):
            lab = "REQUIRED"
        else:
            hint = header_priority(header) if self.use_section_headers else None
            lab = hint or "UNSPECIFIED"
        if self.merge_mandatory and lab == "MANDATORY":
            lab = "REQUIRED"
        return lab

    def predict(self, texts, headers=None) -> list[str]:
        headers = headers if headers is not None else [None] * len(texts)
        return [self.predict_one(t, h) for t, h in zip(texts, headers)]


def _with_header(texts, headers):
    headers = headers if headers is not None else [None] * len(texts)
    return [f"[SEC {h}] {t}" if isinstance(h, str) and h else t for t, h in zip(texts, headers)]


class TfidfLRClassifier:
    name = "C3_tfidf_lr"

    def __init__(self, use_headers: bool = True, C: float = 4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        self.use_headers = use_headers
        self.model = Pipeline([
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True, lowercase=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
        ])

    def fit(self, texts, labels, headers=None):
        X = _with_header(texts, headers) if self.use_headers else list(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts, headers=None):
        X = _with_header(texts, headers) if self.use_headers else list(texts)
        return list(self.model.predict(X))

    def top_features(self, k: int = 15) -> pd.DataFrame:
        """Most indicative n-grams per class (nice for the report)."""
        feats = self.model.named_steps["feats"]
        names = np.concatenate([feats.transformer_list[0][1].get_feature_names_out(),
                                feats.transformer_list[1][1].get_feature_names_out()])
        clf = self.model.named_steps["clf"]
        rows = []
        for ci, c in enumerate(clf.classes_):
            coefs = clf.coef_[ci] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if ci else -1)
            for j in np.argsort(coefs)[::-1][:k]:
                rows.append({"class": c, "feature": names[j], "weight": round(float(coefs[j]), 3)})
        return pd.DataFrame(rows)


class ZeroShotNLIClassifier:
    name = "C4_zero_shot_nli"
    HYPOTHESES = {
        "MANDATORY": "This is a strict eligibility condition; applicants without it are rejected.",
        "REQUIRED": "This is a required qualification for the job.",
        "PREFERRED": "This qualification is optional, a bonus or nice to have.",
        "UNSPECIFIED": "This sentence describes the company or job duties, not a qualification.",
    }

    def __init__(self, model: str = "facebook/bart-large-mnli", labels=LABELS_4, device: int = -1):
        from transformers import pipeline
        self.labels = list(labels)
        self.pipe = pipeline("zero-shot-classification", model=model, device=device)

    def fit(self, *a, **k):
        return self

    def predict(self, texts, headers=None):
        hyps = [self.HYPOTHESES[l] for l in self.labels]
        out = []
        for t in _with_header(texts, headers):
            r = self.pipe(t, candidate_labels=hyps, hypothesis_template="{}", multi_label=False)
            out.append(self.labels[hyps.index(r["labels"][0])])
        return out


class BertClassifier:
    """Fine-tune a transformer sequence classifier. Needs torch + transformers (GPU recommended)."""
    name = "C5_distilbert"

    def __init__(self, base_model: str = "distilbert-base-uncased", epochs: int = 5, lr: float = 3e-5,
                 batch_size: int = 16, use_headers: bool = True, output_dir: str = "models/req_clf"):
        self.base_model, self.epochs, self.lr, self.bs = base_model, epochs, lr, batch_size
        self.use_headers, self.output_dir = use_headers, output_dir

    def fit(self, texts, labels, headers=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        self.classes_ = sorted(set(labels))
        l2i = {l: i for i, l in enumerate(self.classes_)}
        self.tok = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=len(self.classes_), id2label=dict(enumerate(self.classes_)), label2id=l2i)
        X = _with_header(texts, headers) if self.use_headers else list(texts)
        enc = self.tok(X, truncation=True, padding=True, max_length=128)

        class DS(torch.utils.data.Dataset):
            def __len__(s):
                return len(X)

            def __getitem__(s, i):
                d = {k: torch.tensor(v[i]) for k, v in enc.items()}
                d["labels"] = torch.tensor(l2i[labels[i]])
                return d

        args = TrainingArguments(self.output_dir, num_train_epochs=self.epochs, learning_rate=self.lr,
                                 per_device_train_batch_size=self.bs, save_strategy="no", report_to=[],
                                 logging_steps=10)
        Trainer(model=self.model, args=args, train_dataset=DS()).train()
        return self

    def predict(self, texts, headers=None):
        import torch
        X = _with_header(texts, headers) if self.use_headers else list(texts)
        self.model.eval()
        preds = []
        for i in range(0, len(X), 32):
            enc = self.tok(X[i:i + 32], truncation=True, padding=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                preds.extend(self.model(**enc).logits.argmax(-1).tolist())
        return [self.classes_[p] for p in preds]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_classifiers(classifiers: dict, df: pd.DataFrame, text_col="sentence", label_col="gold_label",
                         header_col="section_header", n_splits: int = 5, merge_mandatory: bool = True,
                         seed: int = 42, group_col: str | None = "jd_id"):
    """
    classifiers: {"name": factory_fn}  — a *factory* returning a fresh classifier (needed for CV).
    Non-trainable classifiers are evaluated on all data; trainable ones with (Group)KFold so that
    sentences of the same JD never appear in both train and test.
    Returns (summary_df, predictions_df).
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    df = df.dropna(subset=[label_col]).reset_index(drop=True).copy()
    y = df[label_col].str.upper().str.strip()
    if merge_mandatory:
        y = y.replace({"MANDATORY": "REQUIRED"})
    texts = df[text_col].astype(str).tolist()
    headers = df[header_col].where(df[header_col].notna(), None).tolist() if header_col in df else None
    preds_all = pd.DataFrame({"text": texts, "gold": y})
    rows = []
    for name, factory in classifiers.items():
        clf = factory()
        pred = np.empty(len(df), dtype=object)
        trainable = not isinstance(clf, (RuleClassifier, ZeroShotNLIClassifier))
        if not trainable:
            pred[:] = clf.predict(texts, headers)
        else:
            if group_col and group_col in df and df[group_col].nunique() >= n_splits:
                splits = GroupKFold(n_splits=n_splits).split(texts, y, groups=df[group_col])
            else:
                k = min(n_splits, y.value_counts().min())
                splits = StratifiedKFold(n_splits=max(k, 2), shuffle=True, random_state=seed).split(texts, y)
            for tr, te in splits:
                c = factory().fit([texts[i] for i in tr], [y.iloc[i] for i in tr],
                                  [headers[i] for i in tr] if headers else None)
                pred[te] = c.predict([texts[i] for i in te], [headers[i] for i in te] if headers else None)
        if merge_mandatory:
            pred = np.array(["REQUIRED" if p == "MANDATORY" else p for p in pred], dtype=object)
        preds_all[name] = pred
        rows.append({"method": name, "accuracy": round(accuracy_score(y, pred), 3),
                     "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 3),
                     "weighted_f1": round(f1_score(y, pred, average="weighted", zero_division=0), 3),
                     "n": len(y)})
    return pd.DataFrame(rows), preds_all


def plot_confusion(preds_df: pd.DataFrame, method: str, labels=LABELS_3, ax=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    ConfusionMatrixDisplay.from_predictions(preds_df["gold"], preds_df[method], labels=labels, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(method)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax
