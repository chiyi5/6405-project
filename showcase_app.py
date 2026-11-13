"""Unified local presentation UI. Run python showcase_app.py --open-browser."""
import argparse
import base64
import binascii
import csv
import json
import tempfile
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from learning_app import make_handler
from src.learning import LearningRecommender
from src.learning.workflow import match_documents

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "frontend"
EXPERIMENTS = {
    "skills": ("B_skill_extraction.csv", "SkillSpan · human gold", "f1_soft"),
    "classification": ("C_requirement_clf.csv", "749 sentences · LLM labels, not human-verified", "macro_f1"),
    "matching": ("D_matching.csv", "External fit benchmark · 600 pairs", "spearman"),
    "ablation": ("T_talentclef_ablation_minilm.csv", "TalentCLEF dev · 10 queries · MiniLM", "map"),
}


def bootstrap():
    experiments = {}
    for key, (filename, context, metric) in EXPERIMENTS.items():
        path = ROOT / "results" / filename
        rows = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        experiments[key] = {"source": "results/" + filename, "context": context, "metric": metric, "rows": rows}
    return {"experiments": experiments}


def make_showcase_handler(engine):
    parent = make_handler(engine)
    match_lock = threading.Lock()

    class Handler(parent):
        def do_GET(self):
            route = urlparse(self.path).path
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                      "/locales.js": ("locales.js", "text/javascript"), "/i18n.js": ("i18n.js", "text/javascript"),
                      "/style.css": ("style.css", "text/css")}
            if route in assets:
                filename, mime = assets[route]
                self.respond(200, (WEB / filename).read_bytes(), mime + "; charset=utf-8")
            elif route == "/api/showcase":
                self.respond(200, bootstrap())
            elif route == "/favicon.ico":
                self.respond(204, b"", "image/x-icon")
            else:
                super().do_GET()

        def do_POST(self):
            if self.path not in {"/api/rank", "/api/read-document"}:
                return super().do_POST()
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(403, {"error": "Cross-origin requests are not accepted."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("Request too large (maximum 8 MB).")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if self.path == "/api/read-document":
                    suffix = Path(str(data.get("name", ""))).suffix.lower()
                    if suffix not in {".txt", ".pdf", ".docx"}:
                        raise ValueError("Please upload TXT, PDF or DOCX.")
                    raw = base64.b64decode(data.get("content", ""), validate=True)
                    if len(raw) > 5 * 1024 * 1024:
                        raise ValueError("Maximum document size is 5 MB.")
                    if suffix == ".txt":
                        content = raw.decode("utf-8-sig")
                    else:
                        from src.pipeline import read_document
                        with tempfile.TemporaryDirectory(prefix="skillbridge-") as folder:
                            path = Path(folder) / ("document" + suffix)
                            path.write_bytes(raw)
                            content = read_document(path)
                    if not content.strip():
                        raise ValueError("No text found. Scanned PDFs need OCR; paste the extracted text instead.")
                    if len(content) > 40000:
                        raise ValueError("Document exceeds 40000 characters.")
                    self.respond(200, {"text": content})
                    return
                resumes = data.get("resumes")
                if not isinstance(resumes, list) or not 1 <= len(resumes) <= 12:
                    raise ValueError("Provide between 1 and 12 resumes.")
                rows = []
                with match_lock:
                    for index, item in enumerate(resumes):
                        if not isinstance(item, dict):
                            raise ValueError("Invalid resume.")
                        match = match_documents(data.get("jd"), item.get("text"))
                        rows.append({"name": str(item.get("name", "Candidate"))[:180], "input_index": index, **match})
                rows.sort(key=lambda r: (r["scores"]["hybrid_score"] is not None,
                                        r["scores"]["hybrid_score"] or 0), reverse=True)
                self.respond(200, {"ranking": rows})
            except ImportError as exc:
                self.respond(503, {"error": "Missing document dependency. Install pypdf and python-docx, or paste text. " + str(exc)})
            except (ValueError, TypeError, KeyError, binascii.Error) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception:
                self.respond(500, {"error": "Document processing failed. Check the document or paste plain text."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_showcase_handler(LearningRecommender()))
    print(f"SkillBridge project showcase: http://127.0.0.1:{args.port}", flush=True)
    if args.open_browser:
        threading.Thread(target=lambda: webbrowser.open(f"http://127.0.0.1:{args.port}"), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
