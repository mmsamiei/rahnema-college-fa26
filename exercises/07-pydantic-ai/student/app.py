"""Student local UI; complete agent.py TODOs before sending a question."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import answer

HERE = Path(__file__).resolve().parent
PORT = 8785
ALLOWED_USERS = {"u-104", "u-208"}


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
        if self.path != "/api/ask":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 10000:
                raise ValueError("درخواست نامعتبر است.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("بدنهٔ درخواست باید یک شیء JSON باشد.")
            user_id, question = payload.get("user_id"), payload.get("question")
            if user_id not in ALLOWED_USERS:
                raise ValueError("کاربر آزمایشی معتبر نیست.")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise ValueError("یک سؤال کوتاه وارد کنید.")
            self.send_json(200, answer(user_id, question.strip()))
        except (ValueError, json.JSONDecodeError, NotImplementedError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(502, {"error": f"اجرای Agent ناموفق بود: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        pass


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
