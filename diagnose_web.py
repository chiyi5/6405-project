"""Compare a standard-library HTTP server and the real app on fresh local ports.

Run on the user's machine: python diagnose_web.py. Does not change network settings.
"""
import socket
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from learning_app import make_handler
from src.learning import LearningRecommender


class MinimalHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")


def probe(name, handler, lines):
    server = None
    try:
        # Fresh ports distinguish a fixed-port problem from HTTP application logic.
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=4) as response:
            response.read()
            lines.append(f"{name}: PASS HTTP {response.status} at {url}")
            return True
    except Exception as exc:
        lines.append(f"{name}: FAIL {type(exc).__name__}: {exc}")
        return False
    finally:
        if server:
            server.shutdown()
            server.server_close()


def main():
    lines = [f"Python: {sys.executable}", f"Version: {sys.version}"]
    minimal = probe("Python standard-library HTTP server", MinimalHandler, lines)
    project = probe("SkillBridge actual page handler", make_handler(LearningRecommender()), lines)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8765/api/health", timeout=4) as response:
            lines.append(f"Existing service on 8765: HTTP {response.status}")
    except Exception as exc:
        lines.append(f"Existing service on 8765: {type(exc).__name__}: {exc}")
    if minimal and project:
        lines.append("RESULT: Both HTTP implementations work on fresh ports. Investigate the existing port/process or browser connection.")
    elif minimal and not project:
        lines.append("RESULT: The project HTTP handler fails while the standard server works. Investigate project request handling.")
    elif not minimal and not project:
        lines.append("RESULT: Even a minimal independent HTTP server is unreachable. Investigate the Python runtime/local network environment.")
    else:
        lines.append("RESULT: Inconsistent local connections; repeat to distinguish transient interference.")
    report = Path(__file__).parent / "results" / "learning" / "connection_diagnosis.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
