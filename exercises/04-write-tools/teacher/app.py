"""Small local chat page; the multi-turn tool flow lives in agent.py."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import answer, reset_data

HERE = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        body = (HERE / "web" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 50000:
                raise ValueError("درخواست نامعتبر است.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("بدنهٔ درخواست باید یک شیء JSON باشد.")
            if self.path == "/api/reset":
                reset_data()
                self.send_json(200, {"ok": True})
                return
            if self.path != "/api/ask":
                self.send_error(404)
                return
            question, history = payload.get("question"), payload.get("history", [])
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise ValueError("یک سؤال کوتاه وارد کنید.")
            self.send_json(200, answer(question.strip(), history))
        except (ValueError, json.JSONDecodeError, NotImplementedError) as exc:
            self.send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(502, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    port = 8780
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
