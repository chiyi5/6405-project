"""
Core JD–Resume matching pipeline (refactored from Group32_NLPproject.ipynb).

Steps
  1. read_document / clean_text
  2. JD requirement extraction (skills, experience, education) + priority classification
  3. Resume profile extraction (skills, education, dated experience, evidence sentences)
  4. Requirement-level matching (exact / semantic) and weighted scoring

Bug fixes compared with the original notebook
  * words broken by PDF line wrapping are re-joined ("Num\\nPy" -> "NumPy")
  * contact lines (email / phone / URL) are moved to a "contact" section, so they no
    longer pollute "projects" evidence or leak names into matching
  * `json` is imported at the top (safe_list used json before it was imported)
  * sentences inherit the priority of their section header
    ("Nice to have:" / "Preferred qualifications:" ...) -> see requirement_clf.py
  * duplicate education requirements from one sentence are merged
  * batch API: match one JD against many resumes
"""

from __future__ import annotations

import calendar  # noqa: F401  (kept for compatibility with original notebook)
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .nlp_backend import Embedder, PhraseFinder, split_sentences
from .skill_extraction import SKILL_ALIASES, DictionarySkillExtractor, normalize_skill

# ===========================================================================
# Step 1  Reading and cleaning
# ===========================================================================

def read_document(file_path) -> str:
    """Read TXT / PDF / DOCX into plain text."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n".join(t for p in reader.pages if (t := p.extract_text()))
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(file_path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paras.append(cell.text)
        return "\n".join(paras)
    raise ValueError(f"Unsupported file format: {suffix}. Use TXT, PDF or DOCX.")


# original name kept for backwards compatibility
read_resume = read_document

_KNOWN_JOINED = {k.replace(" ", "").lower() for k in SKILL_ALIASES} | {
    v.replace(" ", "").lower() for v in SKILL_ALIASES.values()}


def fix_broken_words(text: str, vocabulary=_KNOWN_JOINED) -> str:
    """Re-join tokens split by a PDF line break when the joined form is a known word.

    "Pandas, Num\\nPy, Scikit-learn" -> "Pandas, NumPy, Scikit-learn"
    """
    def _join(m):
        a, b = m.group(1), m.group(2)
        return a + b if (a + b).lower() in vocabulary else m.group(0)
    return re.sub(r"([A-Za-z]+)\n([A-Za-z]+)", _join, text)


# camel-case words that must NOT be split by normalize_layout
CAMEL_WHITELIST = {w.lower() for w in """JavaScript TypeScript GitHub GitLab LinkedIn PowerPoint YouTube DevOps MLOps
DataOps SecOps DevSecOps SolidWorks TensorFlow DeepSpeed LangChain LlamaIndex PyTorch NumPy SciPy PySpark iPhone iPad
iOS macOS eBay McDonald WordPress BigQuery PowerShell OneDrive SharePoint QuickBooks NetSuite HubSpot SalesForce
JetBrains IntelliJ PyCharm WebSocket WebSockets NodeJS ReactJS VueJS NextJS FastAPI OpenAI ChatGPT HuggingFace
DynamoDB MongoDB PostgreSQL MySQL CloudFormation CloudWatch ElasticSearch LabVIEW AutoCAD ArcGIS SketchUp
PayPal FinTech HealthTech EdTech AdTech MarTech PhD PhDs SaaS PaaS IaaS DataBricks SnowFlake ServiceNow WorkDay
MuleSoft OutSystems CircleCI TravisCI GraphQL NoSQL OpenShift OpenStack VMware PagerDuty DataDog NewRelic
GoLang TeamCity BitBucket StackOverflow LeetCode KubeFlow AirFlow MLflow XGBoost LightGBM CatBoost""".split()}

_INLINE_HEADER = re.compile(r"(^|(?<=[.!?:])\s*|(?<=\s))([A-Z][A-Za-z'’&/ ]{2,45}?):(?=[A-Z])")
_KNOWN_HEADERS = (r"What You(?:'|’)?ll Do|What You(?:'|’)?ll Bring|What We(?:'|’)?re Looking For|What You Need|"
                  r"Who You Are|Requirements|Qualifications|Minimum Qualifications|Basic Qualifications|"
                  r"Preferred Qualifications|Responsibilities|Key Responsibilities|Nice to Have|Bonus Points|"
                  r"Benefits|Perks|About (?:the|this) (?:Role|Job|Position|Team)|About Us|Job Description|"
                  r"Skills|Required Skills|Preferred Skills|Education|Experience")
_GLUED_HEADER = re.compile(rf"(?<![\w])({_KNOWN_HEADERS})(?=[A-Z][a-z])")
_CAMEL_TOKEN = re.compile(r"\b[\w+#.-]*[a-z][A-Z][\w+#.-]*\b")


def _split_inline_headers(text: str) -> str:
    """'...such as PyTorch More about Gatik:With headquarters' -> '...such as PyTorch\nMore about Gatik:\nWith ...'"""
    out, last = [], 0
    for m in re.finditer(r":(?=[A-Z])", text):
        colon = m.start()
        b = max(text.rfind("\n", 0, colon), max(text.rfind(c, 0, colon) for c in ".!?")) + 1
        seg = text[b:colon]
        words = seg.split()
        header = None
        if words and len(words) <= 6 and words[0][:1].isupper():
            header = seg.strip()
        else:
            for k in range(2, min(6, len(words)) + 1):
                if words[-k][:1].isupper():
                    header = " ".join(words[-k:])
                    break
        if not header or b < last:
            continue
        hstart = text.rfind(header, b, colon + 1)
        out.append(text[last:hstart].rstrip() + "\n" + header + ":\n")
        last = colon + 1
    out.append(text[last:])
    return "".join(out)


def _split_camel(m):
    tok = m.group(0)
    if tok.lower().strip(".") in CAMEL_WHITELIST:
        return tok
    # split only lower->Upper+lower boundaries with >=3 lower-case letters on the left ("fusionBuild")
    return re.sub(r"(?<=[a-z]{3})(?=[A-Z][a-z][a-z-])", "\n", tok)


def normalize_layout(text: str) -> str:
    """Restore line structure in JDs scraped without newlines (e.g. HuggingFace resume-JD fit dataset).

    'Requirements:Python3+ years ...fusionBuild software...' ->
    'Requirements:\\nPython\\n3+ years ...fusion\\nBuild software...'
    """
    text = _split_inline_headers(text)
    text = _GLUED_HEADER.sub(lambda m: f"\n{m.group(1)}:\n", text)
    text = _CAMEL_TOKEN.sub(_split_camel, text)
    # "related field7+ years of experience" / "OpenCVExpert knowledge"
    text = re.sub(r"(?<=[a-z]{3})(?=\d+\+?\s*(?:years?|yrs))", "\n", text)
    text = re.sub(r"(?<=[A-Z]{2})(?=(?:Expert|Experience|Experienced|Proficiency|Proficient|Familiar|Familiarity|"
                  r"Strong|Knowledge|Ability|Bachelor|Master|Minimum|Must|Excellent|Solid)\b)", "\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text


def clean_text(text: str, fix_words: bool = True, fix_layout: bool = True) -> str:
    """Clean JD or resume text while preserving line structure."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    if fix_words:
        text = fix_broken_words(text)
    if fix_layout:
        text = normalize_layout(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


BULLET_RE = re.compile(r"^[•●▪◦‣\-–*]\s*")


def split_segments(text: str) -> list[str]:
    """Split text by lines then by sentence boundaries (bullets removed)."""
    segments = []
    for line in text.splitlines():
        line = BULLET_RE.sub("", line.strip())
        if line:
            segments.extend(split_sentences(line))
    return segments


split_jd_segments = split_segments  # original name


# ===========================================================================
# Step 2  JD requirement extraction
# ===========================================================================

EXPERIENCE_PATTERNS = [
    ("minimum", re.compile(r"\b(?:at\s+least|minimum(?:\s+of)?|min\.?)\s*(?P<min>\d+(?:\.\d+)?)\s*\+?\s*"
                           r"(?:years?|yrs?)\b", re.I)),
    ("range", re.compile(r"\b(?P<min>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<max>\d+(?:\.\d+)?)\s*"
                         r"(?:years?|yrs?)\b", re.I)),
    ("plus", re.compile(r"\b(?P<min>\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\b", re.I)),
    ("greater_than", re.compile(r"\b(?:over|more\s+than)\s*(?P<min>\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.I)),
    ("exact", re.compile(r"\b(?P<min>\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+|professional\s+|"
                         r"working\s+|hands-on\s+|practical\s+|industry\s+)*(?:experience|exp\.?)\b", re.I)),
]

# Word numbers: "two years of experience"
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _words_to_digits(text: str) -> str:
    return re.sub(r"\b(" + "|".join(_WORD_NUM) + r")\b(?=\s*(?:\(\d+\)\s*)?\+?\s*(?:years?|yrs?))",
                  lambda m: str(_WORD_NUM[m.group(1).lower()]), text, flags=re.I)


def extract_experience(text: str) -> list[dict]:
    """Extract numeric experience requirements from one segment."""
    text_n = _words_to_digits(text)
    results, occupied = [], []
    for ptype, pattern in EXPERIENCE_PATTERNS:
        for m in pattern.finditer(text_n):
            s, e = m.span()
            if any(s < oe and e > os_ for os_, oe in occupied):
                continue
            mx = m.groupdict().get("max")
            results.append({"category": "Experience", "requirement": m.group(0),
                            "min_years": float(m.group("min")),
                            "max_years": float(mx) if mx else None,
                            "comparison": ptype, "evidence": text.strip(), "_start": s})
            occupied.append((s, e))
    results.sort(key=lambda r: r.pop("_start"))
    return results


EDUCATION_PATTERNS = {
    "PhD": [r"\bPh\.?\s*D\.?(?!\w)", r"\bdoctoral\s+degree\b", r"\bdoctorate\b", r"\bdoctor\s+of\s+philosophy\b"],
    "Master": [r"\bmaster['’]?s?\s+degree\b", r"\bmaster\s+of\s+[A-Z][\w&-]*(?:\s+(?:of|in|and|&)?\s*[A-Z][\w&-]*)*",
               r"\bmaster['’]?s?\b", r"\bM\.?\s*Sc\.?(?!\w)", r"\bM\.?\s*Eng\.?(?!\w)", r"\bMBA\b",
               r"\bMComp\b", r"\bMTech\b"],
    "Bachelor": [r"\bbachelor['’]?s?\s+degree\b",
                 r"\bbachelor\s+of\s+[A-Z][\w&-]*(?:\s+(?:of|in|and|&|with)?\s*[A-Z][\w&-]*)*",
                 r"\bbachelor['’]?s?\b", r"\bB\.?\s*Sc\.?(?!\w)", r"\bB\.?\s*Eng\.?(?!\w)", r"\bBBA\b",
                 r"\bBTech\b", r"\bundergraduate\s+degree\b"],
    "Diploma": [r"\badvanced\s+diploma\b", r"\bgraduate\s+diploma\b", r"\bpostgraduate\s+diploma\b",
                r"\bdiploma\b", r"\bassociate['’]?s?\s+degree\b", r"\bNITEC\b"],
}
# NB: "MA"/"BA" removed from the original list – they produced false positives
# (e.g. "MA" inside upper-case acronyms).

EDUCATION_CONTEXT = re.compile(
    r"\b(degree|bachelor|master|phd|ph\.d|doctorate|diploma|nitec|qualification|qualifications|"
    r"academic background|educational background|field of study|related field|related discipline|"
    r"graduate(?:d)? in|university|polytechnic|b\.?eng|m\.?sc|b\.?sc)\b", re.I)

STRONG_EDU_CONTEXT = re.compile(r"\b(degree|qualifications?|field of study|related (?:field|discipline)|"
                                r"graduate(?:d)? (?:in|of)|major(?:ing)? in|background in)\b", re.I)

EDUCATION_RANK = {"Diploma": 1, "Bachelor": 2, "Master": 3, "PhD": 4}

DEFAULT_FIELDS = [
    "Computer Science", "Computer Engineering", "Information Technology", "Information Systems",
    "Software Engineering", "Data Science", "Artificial Intelligence", "Machine Learning",
    "Computational Linguistics", "Statistics", "Mathematics", "Applied Mathematics", "Physics",
    "Electrical Engineering", "Electronic Engineering", "Mechanical Engineering", "Engineering",
    "Business Administration", "Business Analytics", "Economics", "Finance", "Accounting",
    "Linguistics", "Psychology", "Marketing", "Management", "Operations Research", "Data Analytics",
]


class EducationExtractor:
    """Degree level + field-of-study extraction (field list from Courses.csv)."""

    def __init__(self, courses_csv: str | None = None, extra_fields=("Machine Learning", "Computational Linguistics"),
                 alternatives_as_or: bool = True):
        self.alternatives_as_or = alternatives_as_or
        fields = list(DEFAULT_FIELDS)
        if courses_csv and Path(courses_csv).exists():
            df = pd.read_csv(courses_csv)
            col = "Courses" if "Courses" in df.columns else df.columns[0]
            fields += df[col].dropna().astype(str).str.strip().tolist()
        fields += list(extra_fields)
        # case-insensitive de-duplication, keep first spelling
        seen, uniq = set(), []
        for f in fields:
            if f and f.casefold() not in seen:
                seen.add(f.casefold())
                uniq.append(f)
        self.fields = uniq
        self._finder = PhraseFinder(uniq, label="FIELD_OF_STUDY")

    def fields_of_study(self, text: str) -> list[str]:
        return [m["text"] for m in self._finder.find(text)]

    def extract(self, text: str) -> list[dict]:
        if not EDUCATION_CONTEXT.search(text):
            return []
        matches = []
        for level, patterns in EDUCATION_PATTERNS.items():
            for p in patterns:
                for m in re.finditer(p, text, re.I if "[A-Z]" not in p else 0):
                    matches.append({"level": level, "text": m.group(0).strip(), "start": m.start(), "end": m.end()})
        # case-insensitive pass for the "of X" patterns as well
        for level in ("Master", "Bachelor"):
            for m in re.finditer(rf"\b{level.lower()}\s+of\s+[a-z][\w&-]*(?:\s+(?:of|in|and|&|with)?\s*[a-z][\w&-]*){{0,6}}",
                                 text, re.I):
                matches.append({"level": level, "text": m.group(0).strip(), "start": m.start(), "end": m.end()})
        selected = []
        for d in sorted(matches, key=lambda x: (-(x["end"] - x["start"]), x["start"])):
            if not any(d["start"] < s["end"] and d["end"] > s["start"] for s in selected):
                selected.append(d)
        fields = self.fields_of_study(text)
        if not selected:
            # "...hosted at Nanyang Technological University (Artificial Intelligence office)" is not a requirement
            if not fields or not STRONG_EDU_CONTEXT.search(text):
                return []
            return [{"category": "Education", "requirement": "Unspecified qualification",
                     "education_level": None, "field_of_study": fields, "evidence": text.strip()}]
        selected = sorted(selected, key=lambda x: x["start"])
        levels = list(dict.fromkeys(d["level"] for d in selected))
        if self.alternatives_as_or and len(levels) > 1 and re.search(r"\bor\b|/", text, re.I):
            # "Diploma, Bachelor's or Master's" = alternatives -> the LOWEST level is sufficient
            low = min(levels, key=lambda l: EDUCATION_RANK[l])
            return [{"category": "Education", "requirement": " / ".join(d["text"] for d in selected),
                     "education_level": low, "field_of_study": fields, "evidence": text.strip()}]
        # one row per distinct level (fixes duplicated rows in the original)
        out, seen = [], set()
        for d in selected:
            if d["level"] in seen:
                continue
            seen.add(d["level"])
            out.append({"category": "Education", "requirement": d["text"], "education_level": d["level"],
                        "field_of_study": fields, "evidence": text.strip()})
        return out


# ---------------------------------------------------------------------------
# Priority classification (rule baseline). Learned classifiers: requirement_clf.py
# ---------------------------------------------------------------------------
from .requirement_clf import RuleClassifier, split_with_sections, split_requirement_clauses  # noqa: E402

_RULE = RuleClassifier(use_section_headers=True)


def classify_requirement(text: str, section_hint: str | None = None) -> str:
    return _RULE.predict_one(text, section_hint)


# ===========================================================================
# Step 3  Resume profile
# ===========================================================================

SECTION_ALIASES = {
    "education": ["education", "academic background", "academic qualifications", "educational background",
                  "education and qualifications"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "internships", "internship experience", "employment"],
    "projects": ["projects", "project experience", "academic projects", "personal projects", "selected projects"],
    "skills": ["skills", "technical skills", "core competencies", "skills and tools", "technical skills and tools",
               "tools", "programming skills"],
    "summary": ["summary", "profile", "professional summary", "objective", "about me", "career objective"],
    "certifications": ["certifications", "certificates", "licenses and certifications", "awards",
                       "achievements", "awards and achievements"],
    "languages": ["languages", "language"],
}
_ALIAS_TO_SECTION = {a: s for s, al in SECTION_ALIASES.items() for a in al}

# ALL-CAPS + glued layout ("...MODELS.EXPERIENCERESEARCH ENGINEER...JUN 2021SKILLSPYTHON, ...").
# normalize_layout() finds these boundaries through lower->Upper changes, which ALL-CAPS text does not have.
_CAPS_HEADER_GLUE = re.compile(
    r"(?:^|(?<=[\n.!?;|]))[ \t]*("
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_SECTION, key=len, reverse=True))
    # glued to the next word ("EXPERIENCERESEARCH"), but not a word ending ("EXPERIENCED", "TOOLSET")
    + r")[ \t]*(?::[ \t]*|(?=[A-Z])(?![A-Z]{1,2}\b))", re.I | re.M)
_CAPS_DATE_GLUE = re.compile(r"(?:(?<=\b(?:19|20)\d\d)|(?<=\bPRESENT)|(?<=\bCURRENT))(?=[A-Z]{2})")


def restore_caps_layout(text: str) -> str:
    """Put known section headings of a glued ALL-CAPS resume back on their own lines.
    Only applied when is_mostly_upper(text); normal resumes never reach this function."""
    text = _CAPS_DATE_GLUE.sub("\n", text)
    return _CAPS_HEADER_GLUE.sub(lambda m: f"\n{m.group(1)}\n", text)

CONTACT_RE = re.compile(
    r"([\w.+-]+@[\w-]+\.[\w.]+)|(\+?\d[\d\s-]{6,}\d)|(linkedin\.com|github\.com|www\.|https?://)", re.I)


def split_resume_sections(text: str) -> dict[str, str]:
    """Split resume into sections by header lines; contact lines go to 'contact'."""
    sections: dict[str, list[str]] = {"other": [], "contact": []}
    current = "other"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        norm = re.sub(r"[^a-z& ]", "", line.lower()).strip()
        if norm in _ALIAS_TO_SECTION and len(line) < 45:
            current = _ALIAS_TO_SECTION[norm]
            sections.setdefault(current, [])
            continue
        if CONTACT_RE.search(line) and len(line) < 120:
            sections["contact"].append(line)
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


MONTHS = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")
DATE_TOKEN = rf"(?:{MONTHS})\.?\s+\d{{4}}|\d{{4}}[/-](?:0?[1-9]|1[0-2])|(?:0?[1-9]|1[0-2])/\d{{4}}"
DATE_RANGE_PATTERN = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|\bto\b)\s*(?P<end>{DATE_TOKEN}|Present|Current|Now|Till date|To date)", re.I)


def extract_experience_entries(text: str) -> list[dict]:
    """Extract dated entries; heading context usually holds job title / company."""
    matches = list(DATE_RANGE_PATTERN.finditer(text))
    entries = []
    for i, m in enumerate(matches):
        prev_end = matches[i - 1].end() if i > 0 else 0
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = text[prev_end:m.start()].strip()
        # heading = last 2 lines before the date (rest belongs to previous description)
        heading_lines = heading.splitlines()[-2:]
        entries.append({"start_date": m.group("start"), "end_date": m.group("end"), "date_range": m.group(0),
                        "heading_context": " | ".join(heading_lines),
                        "description": text[m.end():next_start].strip()})
    # description of entry i should not contain heading of entry i+1
    for i in range(len(entries) - 1):
        nxt = entries[i + 1]["heading_context"].split(" | ")
        desc_lines = entries[i]["description"].splitlines()
        while desc_lines and desc_lines[-1].strip() in nxt:
            desc_lines.pop()
        entries[i]["description"] = "\n".join(desc_lines)
    return entries


_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z'’-]+|[A-Z]\.?|bin|binte|bte|al|de|van|von|da)$")
_NAME_TOKEN_CAPS = re.compile(r"^(?:[A-Z][A-Z'’-]+|[A-Z]\.?|BIN|BINTE|BTE|AL|DE|VAN|VON|DA)$")


def is_mostly_upper(text: str, threshold: float = 0.6) -> bool:
    """True for ALL-CAPS documents (same rule as the skill extractor's caps fix)."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > threshold


def is_name_like(line: str, allow_caps: bool = False) -> bool:
    """2–5 capitalised tokens, no digits / skills punctuation -> probably a person's name line.
    allow_caps: also accept "TAN WEI MING" (only used for ALL-CAPS resumes)."""
    toks = line.strip().split()
    if not 2 <= len(toks) <= 5 or any(ch.isdigit() for ch in line):
        return False
    if not all(_NAME_TOKEN.match(t) or (allow_caps and _NAME_TOKEN_CAPS.match(t)) for t in toks):
        return False
    if any(t.lower().strip(".,") in _NOT_NAME_WORDS for t in toks):
        return False
    return not _NAME_GUARD.extract(line)


_NOT_NAME_WORDS = {w.lower() for w in """Analyst Engineer Developer Scientist Manager Intern Accountant Consultant
Specialist Associate Lead Director Officer Assistant Technician Architect Designer Researcher Administrator
Coordinator Executive Representative Supervisor Programmer Tester Trainee Senior Junior Principal Head President
University Institute College School Polytechnic Academy Bank Company Group Ltd Inc Pte Corp Corporation Agency
Ministry Authority Hospital Laboratory Lab Centre Center Singapore Data Software Business Marketing Sales Finance
Summary Profile Objective Education Experience Skills Projects References Awards""".split()}


class _LazyDict:
    _ex = None

    def extract(self, text):
        if _LazyDict._ex is None:
            from .skill_extraction import DictionarySkillExtractor
            _LazyDict._ex = DictionarySkillExtractor()
        return _LazyDict._ex.extract(text)


_NAME_GUARD = _LazyDict()


EDU_DATE_CONTEXT = re.compile(r"\b(bachelor|master|ph\.?d|diploma|degree|university|college|polytechnic|school|"
                              r"gpa|b\.?sc|m\.?sc|b\.?eng|coursework|expected)\b", re.I)


def parse_resume_date(text):
    if not text:
        return None
    text = str(text).strip().replace("Sept", "Sep")
    if text.lower() in ("present", "current", "now", "till date", "to date"):
        return date.today().replace(day=1)
    for fmt in ("%B %Y", "%b %Y", "%b. %Y", "%Y-%m", "%Y/%m", "%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def calculate_experience_years(entries) -> dict:
    """Total months covered by dated entries (overlaps merged, inclusive months)."""
    intervals, unparsed = [], 0
    for it in entries:
        s, e = parse_resume_date(it.get("start_date")), parse_resume_date(it.get("end_date"))
        if s is None or e is None:
            unparsed += 1
            continue
        a, b = s.year * 12 + s.month, e.year * 12 + e.month
        if b < a:
            unparsed += 1
            continue
        intervals.append((a, b))
    if not intervals:
        return {"years": None, "unparsed_entries": unparsed}
    intervals.sort()
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    months = sum(b - a + 1 for a, b in merged)
    return {"years": round(months / 12, 2), "unparsed_entries": unparsed}


# ===========================================================================
# Pipeline object
# ===========================================================================

SKILL_HEADER = re.compile(r"\bskills?\b|\bcompetenc|\btechnical requirements\b|\btools\b", re.I)


def is_skill_bullet(text: str) -> bool:
    """'Cash Handling', 'POS Systems Operation' -> True; a full duty sentence -> False."""
    t = text.strip(" .;")
    words = t.split()
    return 1 <= len(words) <= 6 and not re.search(r"[.!?]\s", t) and not re.search(
        r"\b(you|we|our|will|must|years?)\b", t, re.I)


PRIORITY_WEIGHTS = {"MANDATORY": 1.0, "REQUIRED": 0.9, "UNSPECIFIED": 0.5, "PREFERRED": 0.3}
DEFAULT_THRESHOLDS = [(0.80, 85.0), (0.65, 70.0), (0.50, 50.0), (0.35, 25.0)]
NEGATIVE_EVIDENCE_PATTERNS = [r"\bno experience (?:with|in|using)\b", r"\bwithout experience (?:with|in|using)\b",
                              r"\bnot familiar with\b", r"\bnever used\b"]


@dataclass
class PipelineConfig:
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    skill_extractor: str = "dictionary+esco"   # "esco" | "dictionary" | "dictionary+esco" | "esco_filtered"
    courses_csv: str | None = None
    priority_weights: dict = field(default_factory=lambda: dict(PRIORITY_WEIGHTS))
    sim_thresholds: list = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    bonus_max: float = 10.0
    top_k: int = 10
    # ablation switches
    use_exact_match: bool = True
    use_semantic_match: bool = True
    use_priority_weights: bool = True
    use_education: bool = True
    use_experience: bool = True
    use_section_headers: bool = True
    # JD bullets under a "Skills" heading become skill requirements
    use_skill_lists: bool = True
    # hybrid ranking: (final_score + w * 100 * cosine(JD, resume)) / (1 + w) breaks ties (0 = off)
    doc_sim_weight: float = 0.0
    # robustness: if no "Experience" section is found, look for date ranges in the whole resume
    experience_fallback: bool = True
    # fairness: drop name-like lines (e.g. "Kavitha Raman") from the evidence pool
    filter_name_lines: bool = True
    # robustness: ALL-CAPS resumes lose the case cues used to restore glued layouts and to spot name lines;
    # re-split known section headings / dates case-insensitively and accept upper-case name lines.
    # Only active when most letters are upper-case, so normal resumes are scored exactly as before.
    caps_fix: bool = True
    # strategy for unknown scores: "skip" (ignore row) or "none" (final score None, like original)
    unknown_policy: str = "skip"


class MatchingPipeline:
    """End-to-end JD–Resume matcher.

    >>> pipe = MatchingPipeline(PipelineConfig(embedder_name="tfidf-char"))
    >>> result = pipe.match(jd_text, resume_text)
    >>> result["final_score"], result["details"]
    """

    def __init__(self, config: PipelineConfig | None = None, embedder: Embedder | None = None,
                 skill_extractor=None):
        self.cfg = config or PipelineConfig()
        self.embedder = embedder or Embedder(self.cfg.embedder_name)
        self.edu = EducationExtractor(self.cfg.courses_csv)
        self.rule_clf = RuleClassifier(use_section_headers=self.cfg.use_section_headers)
        self._qcache = {}
        self._profile_cache = {}
        if skill_extractor is not None:
            self.skill_extractor = skill_extractor
        else:
            from .skill_extraction import build_skill_extractor
            self.skill_extractor = build_skill_extractor(self.cfg.skill_extractor)

    # ---------------- JD ----------------
    def extract_jd_requirements(self, jd_text: str) -> pd.DataFrame:
        jd_clean = clean_text(jd_text)
        segs = split_with_sections(jd_clean)  # [(sentence, section_header_or_None)]
        skills = self.skill_extractor.extract(jd_clean)
        rows = []
        if self.cfg.use_skill_lists:
            seen_sk = {normalize_skill(x) for x in skills}
            for sent, header in segs:
                # short bullets under a "Skills" / "Required skills" heading are skills themselves
                if isinstance(header, str) and SKILL_HEADER.search(header) and is_skill_bullet(sent):
                    k = normalize_skill(sent)
                    if k not in seen_sk:
                        seen_sk.add(k)
                        skills.append(sent.strip(" .;"))
        for sent, header in segs:
            for clause in split_requirement_clauses(sent):
                low = clause.lower()
                for sk in skills:
                    if re.search(r"(?<![\w+#])" + re.escape(sk.lower()) + r"(?![\w+#])", low):
                        rows.append({"category": "Skill", "requirement": sk,
                                     "priority": self.rule_clf.predict_one(clause, header), "evidence": clause})
            pr_sent = self.rule_clf.predict_one(sent, header)
            if pr_sent == "MANDATORY" and not any(r["evidence"] in (sent,) or sent.find(r["evidence"]) >= 0
                                                  for r in rows if r["category"] == "Skill"):
                # hard eligibility condition without a skill (citizenship, licence, clearance ...)
                rows.append({"category": "Constraint", "requirement": sent[:120], "priority": "MANDATORY",
                             "evidence": sent})
            for r in extract_experience(sent):
                r["priority"] = self.rule_clf.predict_one(sent, header)
                rows.append(r)
            for r in self.edu.extract(sent):
                r["priority"] = self.rule_clf.predict_one(sent, header)
                rows.append(r)
        cols = ["requirement_id", "category", "requirement", "priority", "min_years", "max_years",
                "education_level", "field_of_study", "evidence"]
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        if df.empty:
            return df[cols]
        # a skill mentioned several times keeps its strongest priority
        order = {"MANDATORY": 0, "REQUIRED": 1, "UNSPECIFIED": 2, "PREFERRED": 3}
        df["_o"] = df["priority"].map(order)
        df["_key"] = df["category"] + "::" + df["requirement"].astype(str).str.lower()
        df = df.sort_values("_o").drop_duplicates("_key").sort_index().drop(columns=["_o", "_key"])
        df = df.reset_index(drop=True)
        df["requirement_id"] = [f"R{i:03d}" for i in range(1, len(df) + 1)]
        return df[cols]

    # ---------------- Resume ----------------
    def build_resume_profile(self, resume_text: str) -> dict:
        key = hash(resume_text)
        if key in self._profile_cache:            # same resume scored against many JDs
            return self._profile_cache[key]
        prof = self._build_resume_profile(resume_text)
        if len(self._profile_cache) < 5000:
            self._profile_cache[key] = prof
        return prof

    def _build_resume_profile(self, resume_text: str) -> dict:
        caps = self.cfg.caps_fix and is_mostly_upper(resume_text)
        clean = clean_text(restore_caps_layout(resume_text) if caps else resume_text)
        sections = split_resume_sections(clean)
        skills = self.skill_extractor.extract(clean)
        education = []
        for seg in split_segments(sections.get("education", "") or clean):
            education.extend(e for e in self.edu.extract(seg) if e["education_level"])
        experience = extract_experience_entries(sections.get("experience", ""))
        if not experience and self.cfg.experience_fallback:
            # no "Experience" header found (glued / ALL-CAPS layouts): scan the whole resume,
            # but drop date ranges that belong to education entries
            experience = [e for e in extract_experience_entries(clean)
                          if not EDU_DATE_CONTEXT.search(e["heading_context"].split(" | ")[-1] + " "
                                                         + e["description"][:80])]
        evidence, name_lines = [], set()
        for sec in ("summary", "experience", "projects", "skills", "certifications", "other"):
            for seg in split_segments(sections.get(sec, "")):
                if self.cfg.filter_name_lines and sec == "other" and is_name_like(seg, allow_caps=caps):
                    name_lines.add(seg.strip())
                    continue
                evidence.append({"text": seg, "source": sec})
        for s in skills:
            evidence.append({"text": s, "source": "skills"})
        uniq = {}
        for e in evidence:
            uniq[(e["text"].casefold(), e["source"])] = e
        # text for whole-document similarity: same name filter, so the hybrid score cannot depend on the name
        doc_text = "\n".join(l for l in clean.splitlines() if l.strip() not in name_lines) if name_lines else clean
        return {"clean_text": clean, "doc_text": doc_text, "sections": sections, "skills": skills, "education": education,
                "experience": experience, "evidence": list(uniq.values()),
                "experience_summary": calculate_experience_years(experience)}

    # ---------------- Matching ----------------
    def _sim_to_score(self, sim: float) -> float:
        for th, sc in self.cfg.sim_thresholds:
            if sim >= th:
                return sc
        return 0.0

    def match_skill(self, skill: str, evidence, ev_emb) -> dict:
        if not evidence:
            return {"score": 0.0, "match_type": "NOT_EVIDENCED", "evidence": None, "similarity": None}
        usable = [i for i, e in enumerate(evidence)
                  if not any(re.search(p, e["text"], re.I) for p in NEGATIVE_EVIDENCE_PATTERNS)]
        if self.cfg.use_exact_match:
            skill_n = normalize_skill(skill)
            exact = [i for i in usable
                     if re.search(r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])", evidence[i]["text"], re.I)
                     or normalize_skill(evidence[i]["text"]) == skill_n]
            if exact:
                best = max(exact, key=lambda i: evidence[i]["source"] in ("experience", "projects"))
                return {"score": 100.0, "match_type": "EXACT", "evidence": evidence[best]["text"], "similarity": 1.0}
        if not self.cfg.use_semantic_match or not usable:
            return {"score": 0.0, "match_type": "NOT_EVIDENCED", "evidence": None, "similarity": None}
        q = self._qcache.get(skill)
        if q is None:
            q = self._qcache[skill] = self.embedder.encode([skill])
        sims = Embedder.cosine(q, ev_emb)[0]
        best = max(usable, key=lambda i: sims[i])
        sim = float(sims[best])
        score = self._sim_to_score(sim)
        mtype = "RELATED" if score >= 70 else "PARTIAL" if score >= 40 else "WEAK" if score > 0 else "NOT_EVIDENCED"
        return {"score": score, "match_type": mtype, "evidence": evidence[best]["text"], "similarity": sim}

    @staticmethod
    def _safe_list(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                try:
                    return [str(x).strip() for x in json.loads(s.replace("'", '"'))]
                except json.JSONDecodeError:
                    pass
            return [s] if s else []
        return []

    def match_education(self, req: dict, resume_edu: list) -> dict:
        if not resume_edu:
            return {"score": None, "match_type": "UNKNOWN", "evidence": None}
        need = req.get("education_level")
        need = None if need is None or (isinstance(need, float) and np.isnan(need)) else need
        need_fields = self._safe_list(req.get("field_of_study"))
        best, best_ev = None, None
        for it in resume_edu:
            have = it.get("education_level")
            have = None if have is None or (isinstance(have, float) and np.isnan(have)) else have
            if need is None:
                lvl = 100.0
            elif have is None:
                continue
            else:
                lvl = 100.0 if EDUCATION_RANK[have] >= EDUCATION_RANK[need] else 0.0
            have_fields = self._safe_list(it.get("field_of_study"))
            if not need_fields:
                fld = 100.0
            elif any(a.casefold() == b.casefold() for a in need_fields for b in have_fields):
                fld = 100.0
            else:
                # partial credit when the field is semantically close (e.g. CS vs Computer Engineering)
                if have_fields:
                    sims = Embedder.cosine(self.embedder.encode(need_fields), self.embedder.encode(have_fields))
                    fld = 50.0 if float(sims.max()) >= 0.6 else 0.0
                else:
                    fld = 0.0
            sc = min(lvl, fld)
            if best is None or sc > best:
                best, best_ev = sc, it.get("requirement")
        if best is None:
            return {"score": None, "match_type": "UNKNOWN", "evidence": None}
        return {"score": best, "match_type": "MATCHED" if best == 100 else "PARTIAL" if best > 0 else "NOT_MATCHED",
                "evidence": best_ev}

    @staticmethod
    def match_experience(req: dict, summary: dict) -> dict:
        need = req.get("min_years")
        if need is None or pd.isna(need):
            return {"score": None, "match_type": "UNKNOWN", "evidence": "No numeric requirement extracted."}
        have = summary.get("years")
        if have is None:
            return {"score": None, "match_type": "UNKNOWN", "evidence": "No parseable employment dates."}
        need = float(need)
        score = 100.0 if have >= need else round(100.0 * have / need, 1)
        return {"score": score, "match_type": "DURATION_MET" if have >= need else "DURATION_BELOW",
                "evidence": f"{have} years of dated experience; requires {need} years."}

    def match_requirements(self, jd_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        evidence = profile["evidence"]
        if "_ev_emb" not in profile:
            profile["_ev_emb"] = self.embedder.encode([e["text"] for e in evidence]) if evidence else None
        ev_emb = profile["_ev_emb"]
        rows = []
        for _, r in jd_df.iterrows():
            req = r.to_dict()
            cat = str(req.get("category", "")).lower()
            if cat == "skill":
                res = self.match_skill(str(req["requirement"]), evidence, ev_emb)
            elif cat == "education":
                res = self.match_education(req, profile["education"]) if self.cfg.use_education else \
                    {"score": None, "match_type": "DISABLED"}
            elif cat == "experience":
                res = self.match_experience(req, profile["experience_summary"]) if self.cfg.use_experience else \
                    {"score": None, "match_type": "DISABLED"}
            elif cat == "constraint":
                res = {"score": None, "match_type": "NEEDS_HUMAN_VERIFICATION", "evidence": None}
            else:
                res = {"score": None, "match_type": "UNSUPPORTED"}
            rows.append({"requirement_id": req.get("requirement_id"), "category": req.get("category"),
                         "requirement": req.get("requirement"), "priority": req.get("priority"),
                         "match_score": res.get("score"), "match_type": res.get("match_type"),
                         "resume_evidence": res.get("evidence"), "semantic_similarity": res.get("similarity")})
        cols = ["requirement_id", "category", "requirement", "priority", "match_score", "match_type",
                "resume_evidence", "semantic_similarity"]
        return pd.DataFrame(rows, columns=cols)

    # ---------------- Scoring ----------------
    def score(self, match_df: pd.DataFrame) -> dict:
        w = self.cfg.priority_weights if self.cfg.use_priority_weights else {k: 1.0 for k in PRIORITY_WEIGHTS}
        df = match_df.copy()
        if df.empty:
            return {"eligibility": "NO_REQUIREMENTS_EXTRACTED", "base_score": None, "bonus_score": None,
                    "final_score": None, "provisional": False}
        if self.cfg.unknown_policy == "skip":
            df = df[df["match_score"].notna()]
        base_df = df[df["priority"].isin(["MANDATORY", "REQUIRED", "UNSPECIFIED"])]
        pref_df = df[df["priority"] == "PREFERRED"]
        if base_df.empty or base_df["match_score"].isna().any():
            base = None
        else:
            ww = base_df["priority"].map(w)
            base = round(float((base_df["match_score"] * ww).sum() / ww.sum()), 2)
        if pref_df.empty:
            bonus = 0.0
        elif pref_df["match_score"].isna().any():
            bonus = None
        else:
            bonus = round(self.cfg.bonus_max * float(pref_df["match_score"].mean()) / 100, 2)
        mand = match_df[match_df["priority"] == "MANDATORY"]
        if mand.empty:
            elig = "NO_MANDATORY_CONSTRAINTS"
        elif (mand["match_score"] == 0).any():
            elig = "NOT_QUALIFIED"
        elif mand["match_score"].isna().any():
            elig = "PENDING_VERIFICATION"
        else:
            elig = "ELIGIBLE"
        if elig == "NOT_QUALIFIED":
            final = 0.0
        elif base is None or bonus is None:
            final = None
        else:
            final = round(min(100.0, base + bonus), 2)
        return {"eligibility": elig, "base_score": base, "bonus_score": bonus, "final_score": final,
                "provisional": elig == "PENDING_VERIFICATION"}

    # ---------------- Public API ----------------
    def match(self, jd_text: str, resume_text: str, jd_df: pd.DataFrame | None = None) -> dict:
        jd_df = self.extract_jd_requirements(jd_text) if jd_df is None else jd_df
        profile = self.build_resume_profile(resume_text)
        details = self.match_requirements(jd_df, profile)
        out = self.score(details)
        if self.cfg.doc_sim_weight > 0:
            sim = float(Embedder.cosine(self.embedder.encode([jd_text[:4000]]),
                                        self.embedder.encode([profile.get("doc_text", profile["clean_text"])[:4000]]))[0, 0])
            base = out["final_score"] if out["final_score"] is not None else 0.0
            out["doc_similarity"] = round(sim, 4)
            # rescaled to 0-100; a monotone rescaling, so every ranking metric is unchanged
            w = self.cfg.doc_sim_weight
            out["hybrid_score"] = round((base + w * 100 * sim) / (1 + w), 3)
        out.update({"details": details, "jd_requirements": jd_df, "profile": profile})
        return out

    def rank_resumes(self, jd_text: str, resumes: dict[str, str]) -> pd.DataFrame:
        """Batch mode: score many resumes against one JD (JD parsed once)."""
        jd_df = self.extract_jd_requirements(jd_text)
        rows = []
        for rid, text in resumes.items():
            r = self.match(jd_text, text, jd_df=jd_df)
            d = r["details"]
            req_skills = d[(d.category == "Skill") & d.priority.isin(["REQUIRED", "MANDATORY"])]
            rows.append({"resume_id": rid, "final_score": r["final_score"], "base_score": r["base_score"],
                         "bonus_score": r["bonus_score"], "eligibility": r["eligibility"],
                         "required_skills_matched": int((req_skills.match_score >= 70).sum()),
                         "required_skills_total": len(req_skills)})
        return pd.DataFrame(rows).sort_values("final_score", ascending=False, na_position="last").reset_index(drop=True)


def gap_report(result: dict, top_n: int = 10) -> str:
    """Human-readable explanation: matched requirements, gaps, suggestions."""
    d = result["details"]
    head = (f"Requirement coverage: {result['final_score']}  (base {result['base_score']}, "
            f"bonus {result['bonus_score']}, eligibility {result['eligibility']})")
    if result.get("hybrid_score") is not None:
        head = (f"Match score: {result['hybrid_score']} = (coverage + 0.5 x 100 x document similarity "
                f"{result.get('doc_similarity')}) / 1.5\n" + head)
    lines = [head, ""]
    strong = d[d.match_score >= 70].sort_values("match_score", ascending=False)
    gaps = d[(d.match_score.fillna(0) < 50) & d.priority.isin(["MANDATORY", "REQUIRED"])]
    nice = d[(d.match_score.fillna(0) < 50) & (d.priority == "PREFERRED")]
    lines.append("Strengths:")
    for _, r in strong.head(top_n).iterrows():
        lines.append(f"  + {r.requirement} [{r.match_type}] <- \"{str(r.resume_evidence)[:80]}\"")
    lines.append("\nMissing REQUIRED / MANDATORY items:")
    for _, r in gaps.head(top_n).iterrows():
        lines.append(f"  - {r.requirement} ({r.category})")
    lines.append("\nMissing PREFERRED items (nice to have):")
    for _, r in nice.head(top_n).iterrows():
        lines.append(f"  ~ {r.requirement}")
    if len(gaps):
        lines.append("\nSuggestion: if you have experience with the missing required items, state them explicitly "
                     "(skills section + a project / work bullet with concrete outcomes).")
    return "\n".join(lines)
