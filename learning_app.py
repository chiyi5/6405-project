"""Local learning recommender demo, using Python's standard library only.

Run: python learning_app.py; open http://127.0.0.1:8765
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.learning import LearningRecommender, gaps_from_match, parse_gaps
from src.learning.workflow import match_documents, plan_markdown

PAGE = Path(__file__).parent / "src" / "learning" / "index.html"


def make_handler(engine):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log posted resumes or match results.

        def respond(self, status, data, content_type="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.respond(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/health":
                self.respond(200, {"status": "ok", "service": "SkillBridge"})
            elif path == "/api/catalog":
                self.respond(200, {"metadata": engine.metadata, "resources": engine.courses})
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path not in {"/api/recommend", "/api/compare", "/api/match", "/api/export"}:
                self.respond(404, {"error": "Not found"})
                return
            # Browsers may only POST from this local demo, preventing foreign-origin forms.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(403, {"error": "Cross-origin requests are not accepted."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Request must contain 1 to 131072 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                if self.path == "/api/export":
                    self.respond(200, {"markdown": plan_markdown(payload["plan"])})
                    return
                if self.path == "/api/match":
                    with lock:
                        result = match_documents(payload.get("jd"), payload.get("resume"))
                    self.respond(200, result)
                    return
                gaps = gaps_from_match(payload["match_result"]) if payload.get("match_result") is not None else parse_gaps(payload.get("gaps", ""))
                if not gaps:
                    raise ValueError("No skill gaps found. Enter skills or provide unmatched Skill rows.")
                known = payload.get("known", "")
                if not isinstance(known, str):
                    raise ValueError("known must be comma-separated text.")
                options = {"top_k": int(payload.get("top_k", 5)),
                           "known_skills": [s for s in known.split(",") if s.strip()],
                           "budget_hours": payload.get("budget_hours"), "optimize": bool(payload.get("optimize", True))}
                with lock:
                    if self.path == "/api/compare":
                        result = [engine.recommend(gaps, method=m, **{**options, "optimize": o})
                                  for m, o in (("bm25", False), ("tfidf", False), ("hybrid", False), ("hybrid", True))]
                    else:
                        result = engine.recommend(gaps, method=payload.get("method", "hybrid"), **options)
                self.respond(200, result)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                self.respond(400, {"error": str(exc)})
            except ImportError:
                self.respond(503, {"error": "Document matching requires numpy, pandas and scikit-learn. Use Start-Learning.ps1 with the project environment, or use manual skill input."})
            except Exception:
                self.respond(500, {"error": "Recommendation failed. Check the configured model and catalog."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="Open the local page after the server binds successfully")
    parser.add_argument("--dense-model", help="Optional SentenceTransformer model; dense methods never fall back silently")
    args = parser.parse_args()
    engine = LearningRecommender(dense_model=args.dense_model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"Learning module: http://127.0.0.1:{args.port}", flush=True)
    def check_and_open():
        import sys
        import urllib.request
        print(f"Python: {sys.executable}", flush=True)
        # Ignore HTTP_PROXY and system proxy settings for this local diagnostic.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{args.port}/api/health", timeout=5) as response:
                print(f"LOCAL SELF-CHECK: HTTP {response.status} (proxy bypassed)", flush=True)
        except Exception as exc:
            print(f"LOCAL SELF-CHECK FAILED: {exc}", flush=True)
        if args.open_browser:
            import webbrowser
            # Run browser opening separately so a slow URL handler cannot prevent serving.
            webbrowser.open(f"http://127.0.0.1:{args.port}")

    threading.Thread(target=check_and_open, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
