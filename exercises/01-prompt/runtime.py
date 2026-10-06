"""Small HTTP server and real LLM call for the first classroom exercise."""

from __future__ import annotations

import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = [
    "برای انتخاب یک دورهٔ مناسب برنامه‌نویسی از کجا شروع کنم؟",
    "شرایط بازگشت وجه دوره‌های رهنما دقیقاً چیست؟",
    "من دوره را خریدم؛ دسترسی من چه زمانی فعال می‌شود؟",
    "سفارش من را لغو کن و پولم را برگردان.",
]
MAX_TURNS = 10


def configuration() -> tuple[str, str, str, int]:
    """Read a tiny .env without installing a package; environment wins."""
    settings: dict[str, str] = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            settings[name.strip()] = value.strip().strip('"').strip("'")
    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or settings.get(name, default)

    base = get("LLM_BASE_URL").rstrip("/")
    key = get("LLM_API_KEY")
    model = get("LLM_MODEL")
    if not (base and key and model):
        raise ValueError("LLM_BASE_URL، LLM_API_KEY و LLM_MODEL را در فایل .env تنظیم کنید.")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1") or base.startswith("http://localhost")):
        raise ValueError("برای API بیرونی از HTTPS استفاده کنید.")
    try:
        timeout = int(get("LLM_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS باید عدد باشد.") from exc
    if not 1 <= timeout <= 180:
        raise ValueError("LLM_TIMEOUT_SECONDS باید بین ۱ و ۱۸۰ باشد.")
    return base, key, model, timeout


def load_prompt(path: Path) -> str:
    """Reload on every question so edits appear without restarting the server."""
    spec = importlib.util.spec_from_file_location("classroom_prompt", path)
    if spec is None or spec.loader is None:
        raise ValueError("فایل prompt.py پیدا نشد.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SYSTEM_PROMPT باید یک متن غیرخالی باشد.")
    return prompt


def conversation_messages(question: str, prompt_path: Path, history: object) -> list[dict[str, str]]:
    """Build the model input from this conversation's completed turns."""
    if not isinstance(history, list) or len(history) % 2 or len(history) >= MAX_TURNS * 2:
        raise ValueError("تاریخچه نامعتبر یا گفت‌وگو کامل است؛ گفت‌وگوی تازه را شروع کنید.")
    messages = [{"role": "system", "content": load_prompt(prompt_path)}]
    for index, item in enumerate(history):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(item, dict) or item.get("role") != role:
            raise ValueError("ترتیب پیام‌های تاریخچه نامعتبر است.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 8000:
            raise ValueError("متن یکی از پیام‌های تاریخچه نامعتبر است.")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def ask_model(question: str, prompt_path: Path, history: object) -> str:
    base, key, model, timeout = configuration()
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {"model": model, "messages": conversation_messages(question, prompt_path, history)}
    http_request = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"سرویس مدل خطای HTTP {exc.code} برگرداند؛ URL، کلید و نام مدل را بررسی کنید.") from exc
    except error.URLError as exc:
        raise RuntimeError("ارتباط با سرویس مدل برقرار نشد.") from exc
    except TimeoutError as exc:
        raise RuntimeError("زمان انتظار پاسخ مدل تمام شد.") from exc
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("پاسخ سرویس مدل در قالب Chat Completions نبود.") from exc
    if isinstance(answer, list):
        answer = "\n".join(part.get("text", "") for part in answer if isinstance(part, dict))
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("مدل پاسخ متنی برنگرداند.")
    return answer.strip()


def serve(version_dir: Path, port: int) -> None:
    prompt_path = version_dir / "prompt.py"
    page_path = version_dir / "web" / "index.html"

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
            if self.path == "/":
                body = page_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/questions":
                self.send_json(200, {"questions": QUESTIONS})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 180_000:
                    raise ValueError("درخواست بیش از حد بزرگ است؛ گفت‌وگوی تازه را شروع کنید.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("درخواست باید یک شیء JSON باشد.")
                question = payload.get("question")
                if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                    raise ValueError("یک سؤال کوتاه وارد کنید.")
                answer = ask_model(question.strip(), prompt_path, payload.get("history", []))
                self.send_json(200, {"answer": answer})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.send_json(502, {"error": str(exc)})
            except Exception:
                self.send_json(500, {"error": "خطای داخلی؛ خروجی ترمینال را بررسی کنید."})
                raise

        def log_message(self, format: str, *args: object) -> None:
            # No question, response, or credential is written to the terminal.
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.", flush=True)
