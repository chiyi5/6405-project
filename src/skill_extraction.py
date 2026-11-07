"""
Member B — Skill extraction: methods + evaluation.

Methods compared in the report
  M1  ESCOSkillExtractor          original baseline (ojd_daps_skills, ESCO taxonomy)
  M2  ESCOFilteredExtractor       M1 + post-processing (split long spans, drop generic phrases)
  M3  DictionarySkillExtractor    curated skill dictionary + alias normalisation (PhraseMatcher)
  M4  CombinedExtractor           union of M2 and M3
  M5  LLMSkillExtractor           few-shot prompting of any LLM (pass your own llm_fn)
  M6  TokenClassifierExtractor    JobBERT / BERT fine-tuned on SkillSpan (see train_skill_ner())

Evaluation: set-level precision / recall / F1 per document against gold skill lists
(strict = normalised exact string, soft = token-overlap Jaccard >= 0.5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .nlp_backend import PhraseFinder

# ---------------------------------------------------------------------------
# Skill dictionary (canonical names). Extend data/skills_dictionary.txt freely.
# ---------------------------------------------------------------------------
CANONICAL_SKILLS = """
Python|R|Java|JavaScript|TypeScript|C|C++|C#|Go|Rust|Scala|Kotlin|Swift|MATLAB|Julia|Bash|Shell scripting|PHP|Ruby|Perl|VBA
SQL|MySQL|PostgreSQL|SQLite|Oracle|MongoDB|Redis|Cassandra|Elasticsearch|Neo4j|DynamoDB|Snowflake|BigQuery|Redshift|NoSQL
HTML|CSS|React|Angular|Vue.js|Node.js|Express.js|Django|Flask|FastAPI|Spring Boot|REST API|GraphQL|APIs|Web development
Pandas|NumPy|SciPy|Scikit-learn|Matplotlib|Seaborn|Plotly|Statsmodels|XGBoost|LightGBM|CatBoost|NLTK|spaCy|Gensim
PyTorch|TensorFlow|Keras|JAX|Hugging Face|Transformers|Datasets|Tokenizers|Accelerate|TRL|PEFT|LoRA|DeepSpeed|vLLM|SGLang|TGI|LangChain|LlamaIndex|ONNX|OpenCV
Machine learning|Deep learning|Natural language processing|Computer vision|Reinforcement learning|Large language models|Generative AI|Information retrieval|Recommender systems|Time series analysis|Speech recognition|Statistics|Data mining|Feature engineering|A/B testing|Hypothesis testing|Regression|Classification|Clustering|Optimization|Bayesian statistics
Fine-tuning|Pre-training|Continued pre-training|Instruction tuning|Preference optimisation|RLHF|Prompt engineering|Retrieval-augmented generation|Embeddings|Tokenisation|Model evaluation|Model deployment|Model compression|Quantization|Distributed training|GPU programming|CUDA|Multilingual NLP|Experiment tracking|Checkpointing|Profiling|Data pipelines|ETL|Data cleaning|Data deduplication|Data visualization|Data analysis|Data engineering|Data modeling|Data warehousing
Spark|PySpark|Hadoop|Kafka|Airflow|dbt|Databricks|Hive|Flink
AWS|Azure|Google Cloud|GCP|Docker|Kubernetes|Linux|Unix|Git|GitHub|GitLab|CI/CD|Jenkins|Terraform|Ansible|MLOps|MLflow|Weights & Biases|DVC|Slurm|HPC|Cloud computing|Microservices|Serverless
Power BI|Tableau|Excel|Looker|Google Analytics|SAS|SPSS|Stata|Jupyter|VS Code
Unit testing|Testing|Code review|Documentation|Agile|Scrum|Jira|Software engineering|Object-oriented programming|Data structures|Algorithms|System design|Debugging|Version control
Embedded systems|Arduino|Raspberry Pi|Verilog|VHDL|FPGA|IoT|Networking|Cybersecurity|Computer networks|Operating systems
Communication|Teamwork|Leadership|Problem solving|Project management|Stakeholder management|Presentation skills|Critical thinking|Time management|Research
""".replace("\n", "|")
CANONICAL_SKILLS = [s.strip() for s in CANONICAL_SKILLS.split("|") if s.strip()]

# alias (lower-case) -> canonical
SKILL_ALIASES = {
    "py": "Python", "python3": "Python", "js": "JavaScript", "ts": "TypeScript", "golang": "Go",
    "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn", "scikit-learn": "Scikit-learn", "numpy": "NumPy",
    "num py": "NumPy", "pandas": "Pandas", "tensorflow": "TensorFlow", "tf": "TensorFlow", "pytorch": "PyTorch",
    "torch": "PyTorch", "huggingface": "Hugging Face", "hugging face": "Hugging Face", "hf": "Hugging Face",
    "ml": "Machine learning", "dl": "Deep learning", "nlp": "Natural language processing", "cv": "Computer vision",
    "llm": "Large language models", "llms": "Large language models", "genai": "Generative AI",
    "rag": "Retrieval-augmented generation", "postgres": "PostgreSQL", "mssql": "SQL", "t-sql": "SQL",
    "gcp": "Google Cloud", "google cloud platform": "Google Cloud", "amazon web services": "AWS",
    "microsoft azure": "Azure", "k8s": "Kubernetes", "ci cd": "CI/CD", "ci/cd": "CI/CD", "cicd": "CI/CD",
    "powerbi": "Power BI", "power bi": "Power BI", "ms excel": "Excel", "microsoft excel": "Excel",
    "nodejs": "Node.js", "node": "Node.js", "reactjs": "React", "react.js": "React", "vue": "Vue.js",
    "rest": "REST API", "restful api": "REST API", "restful apis": "REST API", "api": "APIs",
    "oop": "Object-oriented programming", "dsa": "Data structures", "data structures and algorithms": "Data structures",
    "tokenization": "Tokenisation", "tokenisation": "Tokenisation", "preference optimization": "Preference optimisation",
    "fine tuning": "Fine-tuning", "finetuning": "Fine-tuning", "pretraining": "Pre-training", "pre training": "Pre-training",
    "visualisation": "Data visualization", "data visualisation": "Data visualization", "web dev": "Web development",
    "wandb": "Weights & Biases", "w&b": "Weights & Biases", "gpu": "GPU programming", "hpc clusters": "HPC",
    "a/b tests": "A/B testing", "unit tests": "Unit testing", "matlab": "MATLAB", "c plus plus": "C++",
    "multilingual": "Multilingual NLP", "data pipeline": "Data pipelines", "deduplication": "Data deduplication",
    "experiment tracking": "Experiment tracking", "spark": "Spark", "pyspark": "PySpark",
}
for _s in CANONICAL_SKILLS:
    SKILL_ALIASES.setdefault(_s.lower(), _s)

# Very short / ambiguous canonical names that need case-sensitive matching
_CASE_SENSITIVE = {"R", "C", "Go", "Git", "TGI", "TRL", "HPC", "AWS", "Spark", "Datasets", "Transformers",
                   "Tokenizers", "Accelerate", "Testing", "Research", "Documentation", "Communication"}


def normalize_skill(s: str) -> str:
    """Canonical lower-case key for comparison (alias-aware)."""
    k = re.sub(r"\s+", " ", str(s).strip().strip(".,;:()").lower())
    return SKILL_ALIASES.get(k, k).lower()


# ---------------------------------------------------------------------------
# M3 dictionary extractor
# ---------------------------------------------------------------------------
class DictionarySkillExtractor:
    name = "M3_dictionary"

    def __init__(self, extra_dictionary: str | None = None, include_soft_skills: bool = True, caps_fix: bool = True):
        self.caps_fix = caps_fix
        aliases = dict(SKILL_ALIASES)
        if extra_dictionary and Path(extra_dictionary).exists():
            for line in Path(extra_dictionary).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                canon, *al = [x.strip() for x in line.split("|")]
                aliases.setdefault(canon.lower(), canon)
                for a in al:
                    aliases[a.lower()] = canon
        soft = {"communication", "teamwork", "leadership", "problem solving", "project management",
                "stakeholder management", "presentation skills", "critical thinking", "time management", "research"}
        if not include_soft_skills:
            aliases = {k: v for k, v in aliases.items() if v.lower() not in soft}
        self.aliases = aliases
        ci = [a for a, c in aliases.items() if c not in _CASE_SENSITIVE and len(a) > 2]
        self._finder = PhraseFinder(ci)
        cs = sorted({c for c in aliases.values() if c in _CASE_SENSITIVE} | {"C++", "C#"}, key=len, reverse=True)
        self._cs_re = re.compile(r"(?<![\w+#.&])(" + "|".join(re.escape(c) for c in cs) + r")(?![\w+#&])")

    def extract(self, text: str) -> list[str]:
        found = [self.aliases.get(m["text"].lower(), m["text"]) for m in self._finder.find(text)]
        letters = [c for c in text if c.isalpha()]
        mostly_upper = letters and sum(c.isupper() for c in letters) / len(letters) > 0.6
        if mostly_upper and self.caps_fix:   # ALL-CAPS resumes: case-sensitive names (Go, Git, R ...) need a case-insensitive pass
            canon = {c.lower(): c for c in self.aliases.values() if c in _CASE_SENSITIVE}
            pat = re.compile(self._cs_re.pattern, re.I)
            found = [self.aliases.get(f.lower(), f) for f in found]
            found += [canon.get(m.group(1).lower(), m.group(1)) for m in pat.finditer(text)]
        else:
            found += [m.group(1) for m in self._cs_re.finditer(text)]
        return list(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# M1 / M2 ESCO extractor (original baseline)
# ---------------------------------------------------------------------------
class ESCOSkillExtractor:
    name = "M1_esco"

    def __init__(self, taxonomy_name: str = "esco"):
        from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor  # heavy import
        self._ex = SkillsExtractor(taxonomy_name=taxonomy_name)

    def extract(self, text: str) -> list[str]:
        doc = self._ex([text])[0]
        out = []
        for sp in doc._.skill_spans:
            t = sp if isinstance(sp, str) else getattr(sp, "text", "")
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                out.append(t)
        return list(dict.fromkeys(out))


GENERIC_HEADS = {"develop", "support", "collaborate", "contribute", "build", "maintain", "translate", "reproduce",
                 "national", "experience", "ability", "able", "familiarity", "strong", "work", "help", "ensure",
                 "drive", "manage", "lead", "review", "join"}
GENERIC_PHRASES = {"development", "workshops", "publications", "curiosity", "resilience", "high ownership",
                   "self-directed learner", "collaborative", "programming", "concepts", "skills", "cleaning",
                   "filtering", "seminars", "outreach", "knowledge-sharing activities", "rigorous", "engineering-minded",
                   "technical sharing", "stakeholders", "internal teams", "industry partners"}


def postprocess_spans(spans: list[str], dictionary: DictionarySkillExtractor | None = None,
                      max_tokens: int = 5) -> list[str]:
    """M2 filter: split enumerations, drop generic verb-led phrases, normalise aliases."""
    dictionary = dictionary or DictionarySkillExtractor()
    out = []
    for sp in spans:
        parts = re.split(r"\s*(?:,|;|\band\b|\bor\b|/(?!CD))\s*", sp)
        for p in parts:
            p = re.sub(r"^(?:experience (?:with|in)|knowledge of|familiarity with|proficient in)\s+", "", p.strip(),
                       flags=re.I).strip(" .:-")
            if not p:
                continue
            toks = p.split()
            in_dict = dictionary.extract(p)
            if in_dict:
                out.extend(in_dict)
                continue
            if len(toks) > max_tokens or toks[0].lower() in GENERIC_HEADS or p.lower() in GENERIC_PHRASES:
                continue
            # fragments such as "Py", "Num" (PDF line-break artefacts): short, lower-case, not an acronym
            if len(p) < 4 and not p.isupper() and p not in _CASE_SENSITIVE:
                continue
            out.append(p)
    return list(dict.fromkeys(SKILL_ALIASES.get(x.lower(), x) for x in out))


class ESCOFilteredExtractor(ESCOSkillExtractor):
    name = "M2_esco_filtered"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._dict = DictionarySkillExtractor()

    def extract(self, text: str) -> list[str]:
        return postprocess_spans(super().extract(text), self._dict)


class CombinedExtractor:
    """M4: union of extractors (order preserved)."""
    name = "M4_combined"

    def __init__(self, *extractors):
        self.extractors = extractors

    def extract(self, text: str) -> list[str]:
        out = []
        for ex in self.extractors:
            out.extend(ex.extract(text))
        seen, res = set(), []
        for s in out:
            k = normalize_skill(s)
            if k not in seen:
                seen.add(k)
                res.append(s)
        return res


# ---------------------------------------------------------------------------
# M5 LLM extractor (provider-agnostic)
# ---------------------------------------------------------------------------
LLM_PROMPT = """You are an information extraction system for recruitment.
Extract every concrete SKILL (programming languages, libraries, tools, platforms, technical methods,
domain knowledge, and explicitly named soft skills) mentioned in the text.
Rules:
- Return short canonical names (1-4 words), e.g. "PyTorch", "Docker", "instruction tuning".
- Split enumerations: "Linux, Docker and CI/CD" -> ["Linux", "Docker", "CI/CD"].
- Do NOT return job duties, company descriptions or personality adjectives.
- Output ONLY a JSON list of strings.

Example
Text: "Proficient in Python and SQL; experience with AWS or GCP is a plus. Strong communicator."
Output: ["Python", "SQL", "AWS", "GCP", "communication"]

Text: \"\"\"{text}\"\"\"
Output:"""


class LLMSkillExtractor:
    """llm_fn: callable(prompt:str)->str. Example with OpenAI / Anthropic / local model in README."""
    name = "M5_llm"

    def __init__(self, llm_fn, prompt: str = LLM_PROMPT, max_chars: int = 6000):
        self.llm_fn, self.prompt, self.max_chars = llm_fn, prompt, max_chars

    def extract(self, text: str) -> list[str]:
        raw = self.llm_fn(self.prompt.format(text=text[: self.max_chars]))
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


# ---------------------------------------------------------------------------
# M6 fine-tuned token classifier (JobBERT / BERT on SkillSpan)
# ---------------------------------------------------------------------------
class TokenClassifierExtractor:
    """Use a fine-tuned HF token-classification model (BIO tags, label containing 'SKILL')."""
    name = "M6_token_clf"

    def __init__(self, model_dir: str, aggregation_strategy: str = "first"):
        from transformers import pipeline
        self._ner = pipeline("token-classification", model=model_dir, aggregation_strategy=aggregation_strategy)

    def extract(self, text: str) -> list[str]:
        out = []
        lines = []
        for l in text.splitlines():
            l = l.strip()
            # windows of <= 800 characters keep every call under the model's 512-token limit
            while len(l) > 800:
                cut = l.rfind(" ", 0, 800)
                cut = cut if cut > 200 else 800
                lines.append(l[:cut])
                l = l[cut:].strip()
            if l:
                lines.append(l)
        for line in lines:
            for ent in self._ner(line):
                # use character offsets -> exact surface form (no "##ava" word-piece fragments)
                w = line[ent["start"]:ent["end"]].strip(" ,.;:()") if ent.get("start") is not None \
                    else ent["word"].replace(" ##", "").replace("##", "").strip()
                if len(w) > 1 or w in ("R", "C"):
                    out.append(w)
        return list(dict.fromkeys(out))


def merge_bio(tags_skill, tags_knowledge) -> list[str]:
    """Merge SkillSpan skill + knowledge BIO tags into one tag sequence ('B'/'I'/'O')."""
    out = []
    for a, b in zip(tags_skill, tags_knowledge):
        a, b = str(a), str(b)
        out.append(a[0] if a[:1] in ("B", "I") else (b[0] if b[:1] in ("B", "I") else "O"))
    return out


def bio_to_spans(tokens, tags) -> list[str]:
    spans, cur = [], []
    for tok, t in zip(tokens, tags):
        if t == "B" or (t == "I" and not cur):
            if cur:
                spans.append(" ".join(cur))
            cur = [tok]
        elif t == "I":
            cur.append(tok)
        else:
            if cur:
                spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def load_skillspan_gold(split: str = "test", group_size: int = 10, max_docs: int | None = None):
    """
    SkillSpan (human-annotated job-posting sentences) -> (docs, gold) for evaluate_extractors.
    Consecutive sentences are grouped into pseudo-documents of `group_size` sentences.
    """
    from datasets import load_dataset
    ds = load_dataset("jjzha/skillspan", split=split)
    docs, gold = {}, {}
    for i in range(0, len(ds), group_size):
        rows = ds[i:i + group_size]
        did = f"SS{i // group_size:04d}"
        texts, skills = [], []
        for toks, ts, tk in zip(rows["tokens"], rows["tags_skill"], rows["tags_knowledge"]):
            texts.append(" ".join(toks))
            skills.extend(bio_to_spans(toks, merge_bio(ts, tk)))
        docs[did], gold[did] = "\n".join(texts), list(dict.fromkeys(skills))
        if max_docs and len(docs) >= max_docs:
            break
    return docs, gold


def train_skill_ner(output_dir: str = "models/skill_ner", base_model: str = "jjzha/jobbert-base-cased",
                    dataset_name: str = "jjzha/skillspan", epochs: int = 3, lr: float = 5e-5, batch_size: int = 16):
    """
    Fine-tune a token classifier on SkillSpan (Zhang et al., NAACL 2022).
    The HF dataset has columns tokens / tags_skill / tags_knowledge (BIO). We merge skill+knowledge -> SKILL.
    Run on GPU (a T4-class GPU is enough, ~15 min). Check the dataset card if column names differ.
    """
    import numpy as np
    from datasets import load_dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments)

    ds = load_dataset(dataset_name)
    labels = ["O", "B-SKILL", "I-SKILL"]
    l2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(base_model)

    def merge(ex):
        ex["labels_merged"] = [l2i[{"B": "B-SKILL", "I": "I-SKILL"}[t[0]]] if t[0] in "BI" else 0
                               for t in merge_bio(ex["tags_skill"], ex["tags_knowledge"])]
        return ex

    def tokenize(ex):
        enc = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256)
        all_labels = []
        for i, lab in enumerate(ex["labels_merged"]):
            word_ids, prev, row = enc.word_ids(i), None, []
            for w in word_ids:
                row.append(-100 if w is None or w == prev else lab[w])
                prev = w
            all_labels.append(row)
        enc["labels"] = all_labels
        return enc

    ds = ds.map(merge).map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=3, id2label=dict(enumerate(labels)),
                                                            label2id=l2i)

    def metrics(p):
        preds = np.argmax(p.predictions, -1)
        mask = p.label_ids != -100
        tp = ((preds == p.label_ids) & (p.label_ids > 0) & mask).sum()
        pp = ((preds > 0) & mask).sum()
        gp = ((p.label_ids > 0) & mask).sum()
        prec, rec = tp / max(pp, 1), tp / max(gp, 1)
        return {"token_precision": prec, "token_recall": rec, "token_f1": 2 * prec * rec / max(prec + rec, 1e-9)}

    args = TrainingArguments(output_dir, learning_rate=lr, per_device_train_batch_size=batch_size,
                             num_train_epochs=epochs, eval_strategy="epoch", save_strategy="epoch",
                             load_best_model_at_end=True, metric_for_best_model="token_f1", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"] if "validation" in ds else ds["test"],
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return trainer.evaluate(ds["test"]) if "test" in ds else None


GENERIC_SINGLE = {"software", "engineering", "science", "technology", "math", "mathematics", "testing", "design",
                  "designing", "development", "programming", "architecture", "installation", "analysis", "support",
                  "web", "applications", "application", "systems", "system", "data", "code", "coding", "stem",
                  "business", "management", "operations", "technical", "experience", "knowledge", "skills",
                  "implementation", "maintenance", "documentation", "reporting", "research", "solutions"}
VERB_HEADS = {"analyze", "analyse", "design", "develop", "discover", "fix", "implement", "perform", "build", "create",
              "maintain", "manage", "support", "write", "test", "review", "ensure", "provide", "work", "collaborate",
              "lead", "drive", "deliver", "identify", "improve", "participate", "assist", "prepare", "conduct",
              "coordinate", "communicate", "plan", "monitor", "evaluate", "define", "deploy", "troubleshoot", "use"}


def is_specific_skill(span: str, dictionary: "DictionarySkillExtractor") -> bool:
    """Keep JobBERT spans that name something checkable; drop verbs, duties and generic nouns.

    keeps: 'GAAP', 'SAP', 'AutoCAD', 'relay settings', 'relational database', 'AC elementary drawings'
    drops: 'designing', 'Analyze', 'software', 'fix programming bugs', 'performing design', 'scalable code'
    """
    toks = span.split()
    if not toks:
        return False
    # degrees and years are handled by the education / experience extractors, not as skills
    if re.search(r"\b(?:bachelor|master|ph\.?d|degree|diploma|\d+\+?\s*(?:years?|yrs))\b", span, re.I):
        return False
    if dictionary.extract(span):
        return True
    first = toks[0].lower().strip(".,")
    if first in VERB_HEADS or first.endswith(("ing", "ed")) or first.endswith(("ize", "yze", "ise")):
        return False
    if len(toks) == 1:
        t = toks[0]
        # single word: keep acronyms / mixed-case product names (GAAP, SAP, AutoCAD, iOS), drop plain words
        return (t.isupper() and len(t) >= 2) or any(c.isupper() for c in t[1:]) or any(c.isdigit() for c in t)
    if all(t.lower().strip(".,") in GENERIC_SINGLE or t.lower() in {"and", "of", "high-quality", "scalable",
                                                                  "testable", "clean"} for t in toks):
        return False
    if toks[0].lower() in {"scalable", "testable", "high-quality", "clean", "robust", "efficient", "strong", "good"}:
        return False
    return True


class JobBERTFilteredExtractor:
    """M7: fine-tuned JobBERT spans, cleaned by the M2 post-processing filter, unioned with the dictionary.

    The dictionary keeps precise canonical names (Python, Docker); JobBERT adds skills the
    dictionary does not list (accounting, wiring, GAAP ...), which removes empty-requirement JDs.
    """
    name = "M7_jobbert+dictionary"

    def __init__(self, model_dir: str, max_tokens: int = 4, strict: bool = True):
        self.strict = strict
        self._ner = TokenClassifierExtractor(model_dir)
        self._dict = DictionarySkillExtractor()
        self.max_tokens = max_tokens

    def extract(self, text: str) -> list[str]:
        base = self._dict.extract(text)
        seen = {normalize_skill(d) for d in base}
        extra = []
        for sp in postprocess_spans(self._ner.extract(text), self._dict, max_tokens=self.max_tokens):
            k = normalize_skill(sp)
            if k not in seen and len(sp) >= 2 and (not self.strict or is_specific_skill(sp, self._dict)):
                seen.add(k)
                extra.append(sp)
        return base + extra


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_skill_extractor(kind: str = "dictionary+esco", **kw):
    if kind.lower().startswith("jobbert:"):          # "jobbert:models/skill_ner"
        return JobBERTFilteredExtractor(kind.split(":", 1)[1])
    kind = kind.lower()
    if kind == "dictionary":
        return DictionarySkillExtractor(**kw)
    try:
        if kind == "esco":
            return ESCOSkillExtractor()
        if kind == "esco_filtered":
            return ESCOFilteredExtractor()
        if kind == "dictionary+esco":
            return CombinedExtractor(DictionarySkillExtractor(), ESCOFilteredExtractor())
    except ImportError:
        print(f"[skills] ojd_daps_skills not installed -> using dictionary extractor instead of '{kind}'")
        return DictionarySkillExtractor()
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _tokens(s):
    return set(re.findall(r"[a-z0-9+#]+", normalize_skill(s)))


def soft_match(a: str, b: str, threshold: float = 0.5) -> bool:
    if normalize_skill(a) == normalize_skill(b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


def prf(pred: list[str], gold: list[str], mode: str = "strict") -> dict:
    """Set-level P/R/F1 for one document. mode = strict | soft."""
    pred = list(dict.fromkeys(pred))
    gold = list(dict.fromkeys(gold))
    if mode == "strict":
        P = {normalize_skill(x) for x in pred}
        G = {normalize_skill(x) for x in gold}
        tp_p = tp_g = len(P & G)
        np_, ng = len(P), len(G)
    else:
        tp_p = sum(any(soft_match(p, g) for g in gold) for p in pred)
        tp_g = sum(any(soft_match(p, g) for p in pred) for g in gold)
        np_, ng = len(pred), len(gold)
    p = tp_p / np_ if np_ else 0.0
    r = tp_g / ng if ng else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "n_pred": np_, "n_gold": ng}


def evaluate_extractors(extractors: dict, docs: dict[str, str], gold: dict[str, list[str]]) -> pd.DataFrame:
    """
    extractors: {"M1_esco": obj, ...}  (obj.extract(text)->list[str])
    docs:       {doc_id: text}
    gold:       {doc_id: [skills]}   (load with data_utils.load_skill_gold)
    Returns macro-averaged P/R/F1 (strict & soft) per method.
    """
    rows = []
    for name, ex in extractors.items():
        per = []
        for did, text in docs.items():
            if did not in gold:
                continue
            pred = ex.extract(text)
            for mode in ("strict", "soft"):
                m = prf(pred, gold[did], mode)
                per.append({"method": name, "doc_id": did, "mode": mode, **m})
        rows.extend(per)
    df = pd.DataFrame(rows)
    summary = (df.groupby(["method", "mode"])[["precision", "recall", "f1", "n_pred"]].mean()
               .round(3).reset_index().pivot(index="method", columns="mode"))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index(), df


def extraction_errors(extractor, docs, gold, mode="soft") -> pd.DataFrame:
    """List false positives / false negatives for error analysis."""
    rows = []
    for did, text in docs.items():
        if did not in gold:
            continue
        pred = extractor.extract(text)
        for p in pred:
            if not any(soft_match(p, g) if mode == "soft" else normalize_skill(p) == normalize_skill(g) for g in gold[did]):
                rows.append({"doc_id": did, "type": "FP", "skill": p})
        for g in gold[did]:
            if not any(soft_match(p, g) if mode == "soft" else normalize_skill(p) == normalize_skill(g) for p in pred):
                rows.append({"doc_id": did, "type": "FN", "skill": g})
    return pd.DataFrame(rows)
