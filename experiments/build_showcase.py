"""Generate a portable UI preview without sample documents or cached analyses.

The interactive server remains the entry point for arbitrary documents/options.
No benchmark or notebook output is modified.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from showcase_app import bootstrap
from src.learning import LearningRecommender


def main():
    data = bootstrap()
    engine = LearningRecommender()
    bundle = {"bootstrap": data, "catalog": {"metadata": engine.metadata, "resources": engine.courses}}
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace('<link rel="stylesheet" href="/style.css">', '<style>' + css + '</style>')
    for script in ("locales.js", "i18n.js"):
        content = (ROOT / "frontend" / script).read_text(encoding="utf-8")
        html = html.replace(f'<script src="/{script}" defer></script>', '<script>' + content + '</script>')
    html = html.replace('<script src="/app.js" defer></script>', '<script>window.SKILLBRIDGE_OFFLINE=' + encoded + ';</script><script>' + js + '</script>')
    output = ROOT / "Showcase.html"
    output.write_text(html, encoding="utf-8")
    print(f"Built {output.name}: empty document inputs; no sample resumes or cached match results.")


if __name__ == "__main__":
    main()
