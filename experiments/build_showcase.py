"""Generate a portable HTML demo from real cached synthetic-example results.

The interactive server remains the entry point for arbitrary documents/options.
No benchmark or notebook output is modified.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from showcase_app import bootstrap
from src.learning import LearningRecommender, parse_gaps
from src.learning.workflow import match_documents


def main():
    data = bootstrap()
    engine = LearningRecommender()
    bundle = {"bootstrap": data, "catalog": {"metadata": engine.metadata, "resources": engine.courses},
              "matches": [], "plans": {}}

    def add_plan(text, known):
        key = json.dumps([text, known], ensure_ascii=False, separators=(",", ":"))
        bundle["plans"][key] = [engine.recommend(parse_gaps(text), method=method, optimize=opt,
                                                known_skills=known.split(","), top_k=5)
                                for method, opt in (("bm25", False), ("tfidf", False), ("hybrid", False), ("hybrid", True))]
    add_plan("SQL | REQUIRED\nDocker | REQUIRED\nFastAPI | PREFERRED", "Python")
    for jd in data["jobs"]:
        for cv in data["resumes"]:
            result = match_documents(jd["text"], cv["text"])
            bundle["matches"].append({"jd": jd["text"], "resume": cv["text"], "result": result})
            # Keep the UI's original requirement names in its exact input key.
            gaps = [d for d in result["details"] if str(d.get("category", "")).lower() == "skill"
                    and d.get("match_type") != "DISABLED" and (d.get("match_score") is None or d["match_score"] < 50)]
            if gaps:
                text = "\n".join(d["requirement"] + " | " + (d.get("priority") or "REQUIRED") for d in gaps)
                add_plan(text, ", ".join(result["profile"]["skills"]))
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace('<link rel="stylesheet" href="/style.css">', '<style>' + css + '</style>')
    html = html.replace('<script src="/app.js" defer></script>', '<script>window.SKILLBRIDGE_OFFLINE=' + encoded + ';</script><script>' + js + '</script>')
    output = ROOT / "Showcase.html"
    output.write_text(html, encoding="utf-8")
    print(f"Built {output.name}: {len(bundle['matches'])} computed pairs, {len(bundle['plans'])} learning scenarios.")


if __name__ == "__main__":
    main()
