"""Course-recommendation benchmark: corpus, queries and silver relevance labels.

Data sources
------------
* Coursera course catalogue (3,522 rows, 7 columns incl. ``Course Description`` and
  ``Skills``), public copy of the Kaggle "Coursera Courses Dataset 2021".  The
  ``Skills`` field is Coursera's own tag list and is used **only** to derive silver
  relevance labels; retrievers never see it (no label leakage).
* The 12 curated official-documentation units in ``data/learning/catalog.json``.
* Real job descriptions already in the project: the 30 LLM-labelled JDs in
  ``data/annotations/requirements_labelled.csv`` (requirement sentences with priority)
  and the JD/resume pairs in ``results/A_sample_pairs.csv``.

Query sets
----------
* ``skill``   : one query per skill of the matcher's skill vocabulary that has >=3
                tagged courses (the gap names the matcher actually emits).
* ``sentence``: real JD requirement sentences that mention one of those skills (the
                harder, noisier phrasing a candidate would see in a job ad).
* ``profile`` : multi-skill gap profiles (JD skills missing from a real resume),
                used for the end-to-end set-recommendation evaluation.

Splits are made by *skill* (train/dev/test = 60/20/20, fixed seed), so a sentence
query never shares its target skill with a training query.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "learning" / "external"
BENCH = ROOT / "data" / "learning" / "benchmark"
COURSERA_URL = "https://raw.githubusercontent.com/aryan4codes/course_recommender_system/main/Coursera.csv"

# Skill names that are also ordinary English words; the dictionary extractor matches them
# case-insensitively, which produced false queries such as "Youll excel if you can collaborate".
AMBIGUOUS_WORD_SKILLS = {"Excel", "Agile", "Swift", "Spark", "Go", "R", "C", "Testing", "Research", "Communication"}
_SLUG = re.compile(r"[a-z]+(?:-[a-z]+)+")
_TOK = re.compile(r"[a-z0-9+#]+")


def _toks(s: str) -> list[str]:
    return _TOK.findall(str(s).lower())


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def download_coursera(path: Path = EXTERNAL / "coursera_courses.csv", url: str = COURSERA_URL) -> Path:
    """Download once and cache; the cached file is the frozen version used for results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def parse_coursera_tags(raw: str) -> tuple[list[str], list[str]]:
    """Split Coursera's ``Skills`` string into (skill tags, domain slugs).

    Tags are separated by two or more spaces; the domain/sub-domain slugs
    (e.g. ``computer-science``) are glued to the last tag with single spaces.
    """
    tags, domains = [], []
    for part in re.split(r"\s{2,}", str(raw)):
        words = [w for w in part.strip().split(" ") if w]
        slugs = [w for w in words if _SLUG.fullmatch(w)]
        rest = " ".join(w for w in words if w not in slugs)
        domains += slugs
        if rest:
            tags.append(rest)
    return tags, domains


def clean_text(s: str) -> str:
    s = str(s).replace("�", "'")
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_corpus(include_curated: bool = True) -> pd.DataFrame:
    """Return one row per resource: id, title, provider, text, tags, domains, source, url, level."""
    from src.skill_extraction import normalize_skill

    df = pd.read_csv(download_coursera(), encoding="utf-8", on_bad_lines="skip")
    df = df.drop_duplicates("Course_Name").reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        tags, domains = parse_coursera_tags(r["Skills"])
        cid = "cr-" + hashlib.md5(str(r["Course_Name"]).encode()).hexdigest()[:8]
        rows.append({"id": cid, "title": clean_text(r["Course_Name"]), "provider": str(r["University"]),
                     "description": clean_text(r["Course Description"]),
                     "tags": sorted({normalize_skill(t) for t in tags}),
                     "domains": sorted(set(domains)), "level": str(r["Difficulty Level"]),
                     "rating": pd.to_numeric(r["Course Rating"], errors="coerce"),
                     "url": str(r["Course URL"]), "source": "coursera"})
    if include_curated:
        cat = json.loads((ROOT / "data" / "learning" / "catalog.json").read_text(encoding="utf-8"))
        for c in cat["resources"]:
            rows.append({"id": c["id"], "title": c["title"], "provider": c["provider"],
                         "description": c["description"] + " " + c["source_excerpt"],
                         "tags": sorted({normalize_skill(s) for s in c["skills"]}), "domains": ["official-docs"],
                         "level": "Unknown", "rating": float("nan"), "url": c["url"], "source": "curated"})
    corpus = pd.DataFrame(rows)
    corpus["text"] = corpus.title + ". " + corpus.description
    return corpus


# ---------------------------------------------------------------------------
# Silver relevance
# ---------------------------------------------------------------------------
def tag_matches(skill_key: str, tag: str) -> bool:
    """A tag supports a skill if it is the same skill or the skill plus <=1 extra word.

    ``python`` <- ``python programming`` (yes); ``sql`` <- ``nosql`` (no: different token);
    ``machine learning`` <- ``machine learning algorithms`` (yes).
    """
    if tag == skill_key:
        return True
    kt, tt = _toks(skill_key), _toks(tag)
    return bool(kt) and len(tt) <= len(kt) + 1 and all(t in tt for t in kt)


def relevant_ids(skill: str, corpus: pd.DataFrame) -> set[str]:
    from src.skill_extraction import normalize_skill
    key = normalize_skill(skill)
    return {cid for cid, tags in zip(corpus.id, corpus.tags) if any(tag_matches(key, t) for t in tags)}


def skill_vocabulary(corpus: pd.DataFrame, min_rel: int = 3) -> dict[str, set[str]]:
    """Matcher skill vocabulary restricted to skills with >= ``min_rel`` tagged resources.

    Synonymous canonical names that map to an identical relevant set (e.g. GCP / Google Cloud)
    are collapsed to the first one so a skill is not counted twice.
    """
    from src.skill_extraction import CANONICAL_SKILLS
    out, seen = {}, {}
    for sk in CANONICAL_SKILLS:
        rel = relevant_ids(sk, corpus)
        if len(rel) < min_rel:
            continue
        key = frozenset(rel)
        if key in seen:
            continue
        seen[key] = sk
        out[sk] = rel
    return out


def split_skills(skills, seed: int = 13, frac=(0.6, 0.2, 0.2)) -> dict[str, str]:
    skills = sorted(skills)
    random.Random(seed).shuffle(skills)
    n = len(skills)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return {s: ("train" if i < a else "dev" if i < b else "test") for i, s in enumerate(skills)}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def jd_documents() -> dict[str, pd.DataFrame]:
    """Real JDs as sentence tables (jd_id -> sentence, gold_label)."""
    r = pd.read_csv(ROOT / "data" / "annotations" / "requirements_labelled.csv")
    return {j: g.reset_index(drop=True) for j, g in r.groupby("jd_id")}


def build_queries(corpus: pd.DataFrame, min_rel: int = 3, seed: int = 13):
    """Return (queries DataFrame, qrels DataFrame, vocab dict)."""
    from src.skill_extraction import DictionarySkillExtractor, normalize_skill
    vocab = skill_vocabulary(corpus, min_rel)
    split = split_skills(vocab, seed)
    q, qrels = [], []

    def add(qid, text, kind, skill, extra=None):
        q.append({"qid": qid, "query": text, "kind": kind, "skill": skill, "split": split[skill], **(extra or {})})
        for cid in vocab[skill]:
            qrels.append({"qid": qid, "doc_id": cid, "rel": 1})

    for sk in vocab:
        slug = re.sub(r"\W+", "_", sk.lower()).strip("_")
        add(f"S-{slug}-{hashlib.md5(sk.encode()).hexdigest()[:4]}", sk, "skill", sk)  # hash: "C" vs "C++"

    ex = DictionarySkillExtractor()
    key2skill = {normalize_skill(s): s for s in vocab}
    for jd, g in jd_documents().items():
        for _, row in g.iterrows():
            sent = str(row.sentence)
            # only sentences the requirement classifier's gold labels mark as requirements;
            # company blurbs ("collaborate with leadership") are not skill needs
            if row.gold_label not in ("REQUIRED", "PREFERRED", "MANDATORY") or not 25 <= len(sent) <= 400:
                continue
            found = [key2skill[normalize_skill(s)] for s in ex.extract(sent) if normalize_skill(s) in key2skill]
            # one target skill per sentence query keeps the relevance definition unambiguous
            # common-word skill names ("excel if you can", "swift delivery") must keep their case
            found = [f for f in found if f not in AMBIGUOUS_WORD_SKILLS or re.search(rf"\b{re.escape(f)}\b", sent)]
            if len(set(found)) == 1:
                add(f"J-{row.sent_id}", sent, "sentence", found[0],
                    {"jd_id": jd, "priority": row.gold_label})
    return pd.DataFrame(q), pd.DataFrame(qrels), vocab


def build_profiles(vocab: dict[str, set[str]], seed: int = 13) -> list[dict]:
    """Gap profiles = JD skills (in vocab) the paired real resume does not mention.

    Sources: the 30 labelled JDs (priority from the sentence label) crossed with real
    resumes from ``results/A_sample_pairs.csv``; plus the sample pairs themselves.
    """
    from src.skill_extraction import DictionarySkillExtractor, normalize_skill
    ex = DictionarySkillExtractor()
    key2skill = {normalize_skill(s): s for s in vocab}
    rank = {"MANDATORY": 3, "REQUIRED": 2, "UNSPECIFIED": 1, "PREFERRED": 0}
    pairs = pd.read_csv(ROOT / "results" / "A_sample_pairs.csv")
    resumes = dict(zip(pairs.resume_id, pairs.resume_text))
    rng = random.Random(seed)
    profiles = []

    def gaps_from(jd_rows, resume_text):
        have = {normalize_skill(s) for s in ex.extract(resume_text)}
        best = {}
        for sent, label in jd_rows:
            for s in ex.extract(str(sent)):
                k = normalize_skill(s)
                if k in key2skill and k not in have:
                    sk = key2skill[k]
                    if sk not in best or rank[label] > rank[best[sk]]:
                        best[sk] = label
        return [{"skill": s, "priority": p} for s, p in sorted(best.items())]

    for jd, g in jd_documents().items():
        rid = rng.choice(sorted(resumes))
        gaps = gaps_from(zip(g.sentence, g.gold_label), resumes[rid])
        if len(gaps) >= 2:
            profiles.append({"pid": f"{jd}__{rid}", "jd_id": jd, "resume_id": rid, "source": "labelled_jd", "gaps": gaps})
    for _, r in pairs.iterrows():
        gaps = gaps_from([(s, "UNSPECIFIED") for s in re.split(r"(?<=[.!?\n])\s+", r.jd_text)], r.resume_text)
        if len(gaps) >= 2:
            profiles.append({"pid": f"{r.jd_id}__{r.resume_id}", "jd_id": r.jd_id, "resume_id": r.resume_id,
                             "source": "fit_pair", "gaps": gaps})
    return profiles


def save_benchmark(out: Path = BENCH, seed: int = 13) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    queries, qrels, vocab = build_queries(corpus, seed=seed)
    profiles = build_profiles(vocab, seed)
    corpus.to_json(out / "corpus.jsonl", orient="records", lines=True, force_ascii=False)
    queries.to_csv(out / "queries.csv", index=False, encoding="utf-8")
    qrels.to_csv(out / "qrels_silver.csv", index=False, encoding="utf-8")
    (out / "profiles.json").write_text(json.dumps(profiles, indent=1, ensure_ascii=False), encoding="utf-8")
    stats = {"resources": len(corpus), "coursera": int((corpus.source == "coursera").sum()),
             "curated": int((corpus.source == "curated").sum()), "skills": len(vocab),
             "queries": queries.kind.value_counts().to_dict(),
             "queries_by_split": queries.groupby(["kind", "split"]).size().unstack(fill_value=0).to_dict("index"),
             "qrels": len(qrels), "mean_relevant_per_query": round(qrels.groupby("qid").size().mean(), 2),
             "profiles": len(profiles), "mean_gaps_per_profile": round(sum(len(p["gaps"]) for p in profiles) / max(len(profiles), 1), 2),
             "domains": Counter(d for ds in corpus.domains for d in ds).most_common(15), "seed": seed,
             "coursera_sha256": hashlib.sha256(download_coursera().read_bytes()).hexdigest()}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return stats


def load_benchmark(path: Path = BENCH):
    corpus = pd.read_json(path / "corpus.jsonl", lines=True)
    queries = pd.read_csv(path / "queries.csv")
    qrels = pd.read_csv(path / "qrels_silver.csv")
    profiles = json.loads((path / "profiles.json").read_text(encoding="utf-8"))
    rel = defaultdict(dict)
    for qid, d, r in qrels.itertuples(index=False):
        rel[qid][d] = r
    return corpus, queries, dict(rel), profiles
