"""Small, inspectable RAG runtime for exercise 02."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_TURNS = 10
WINDOW_SIZE = 350
WINDOW_OVERLAP = 70
QUESTIONS = [
    "پیش از فعال شدن دسترسی، تا چه زمانی می‌توانم بازگشت وجه بگیرم؟",
    "اگر ویدئوهای دوره خراب باشند و من دوره را فعال کرده باشم، چه استثنایی وجود دارد؟",
    "پرداخت من موفق بوده ولی دسترسی دوره فعال نشده؛ چه زمانی باید فعال شود؟",
    "اگر دسترسی دوره هنوز فعال نشده باشد، می‌توانم ثبت‌نام را لغو کنم؟",
    "اگر ویدئو خراب است و دسترسی هم فعال نشده، چه کار کنم؟",
    "پرداخت من موفق بوده؛ آیا دسترسی من الان فعال شده است؟",
]
STOP_WORDS = {"و", "یا", "را", "که", "در", "از", "به", "با", "برای", "من", "این", "آن", "است", "شد", "شده", "اگر", "چه", "تا", "می", "شود", "بوده", "اما"}


def configuration() -> tuple[str, str, str, int]:
    settings: dict[str, str] = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                name, value = stripped.split("=", 1)
                settings[name.strip()] = value.strip().strip('"').strip("'")
    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or settings.get(name, default)

    base, key, model = get("LLM_BASE_URL").rstrip("/"), get("LLM_API_KEY"), get("LLM_MODEL")
    if not (base and key and model):
        raise ValueError("LLM_BASE_URL، LLM_API_KEY و LLM_MODEL را در فایل .env تنظیم کنید.")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1") or base.startswith("http://localhost")):
        raise ValueError("برای API بیرونی از HTTPS استفاده کنید.")
    try:
        timeout = int(get("LLM_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS باید عدد باشد.") from exc
    return base, key, model, timeout


def load_prompt(path: Path) -> str:
    spec = importlib.util.spec_from_file_location("rag_prompt", path)
    if spec is None or spec.loader is None:
        raise ValueError("فایل prompt.py پیدا نشد.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SYSTEM_PROMPT باید یک متن غیرخالی باشد.")
    return prompt


def normalize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه"}))
    return [word for word in re.findall(r"[\wآ-ی]+", text) if len(word) > 1 and word not in STOP_WORDS]


def load_documents(directory: Path) -> list[dict[str, str]]:
    return [{"source": item.stem, "title": item.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip(),
             "text": item.read_text(encoding="utf-8")} for item in sorted(directory.glob("*.md"))]


def split_document(document: dict[str, str], mode: str) -> list[dict[str, str]]:
    text = document["text"].replace("<!-- chunk -->", "").strip()
    if mode == "full":
        parts = [text]
    elif mode == "window":
        step = WINDOW_SIZE - WINDOW_OVERLAP
        parts = [text[index:index + WINDOW_SIZE] for index in range(0, len(text), step)]
    elif mode == "manual":
        parts = [part.strip() for part in document["text"].split("<!-- chunk -->") if part.strip()]
    else:
        raise ValueError("حالت chunking نامعتبر است.")
    return [{"id": f"{document['source']}#{number}", "source": document["source"],
             "title": document["title"], "text": part} for number, part in enumerate(parts, 1)]


def retrieve(question: str, documents: list[dict[str, str]], mode: str) -> list[dict[str, object]]:
    query_words = normalize(question)
    chunks = [chunk for document in documents for chunk in split_document(document, mode)]
    chunk_terms = [normalize(chunk["title"] + " " + chunk["text"]) for chunk in chunks]
    document_frequency = {word: sum(word in terms for terms in chunk_terms) for word in set(query_words)}
    scored = []
    for chunk, terms in zip(chunks, chunk_terms):
        score = sum(terms.count(word) * (1 + math.log((len(chunks) + 1) / (document_frequency[word] + 1)))
                    for word in query_words)
        if score:
            scored.append({**chunk, "score": round(score, 2), "characters": len(chunk["text"])})
    return sorted(scored, key=lambda item: (-item["score"], item["id"]))[:2]


def conversation_messages(question: str, prompt_path: Path, history: object, chunks: list[dict[str, object]]) -> list[dict[str, str]]:
    if not isinstance(history, list) or len(history) % 2 or len(history) >= MAX_TURNS * 2:
        raise ValueError("تاریخچه نامعتبر یا گفت‌وگو کامل است؛ گفت‌وگوی تازه را شروع کنید.")
    context = "\n\n".join(f"[{item['id']}]\n{item['text']}" for item in chunks)
    system = load_prompt(prompt_path) + "\n\nمنابع بازیابی‌شده برای پیام آخر:\n" + (context or "[منبع مرتبطی پیدا نشد]")
    messages = [{"role": "system", "content": system}]
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


def ask_model(question: str, prompt_path: Path, history: object, chunks: list[dict[str, object]]) -> str:
    base, key, model, timeout = configuration()
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {"model": model, "messages": conversation_messages(question, prompt_path, history, chunks)}
    http_request = request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                   headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"سرویس مدل خطای HTTP {exc.code} برگرداند؛ URL، کلید و نام مدل را بررسی کنید.") from exc
    except error.URLError as exc:
        raise RuntimeError("ارتباط با سرویس مدل برقرار نشد.") from exc
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("پاسخ سرویس مدل در قالب Chat Completions نبود.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("مدل پاسخ متنی برنگرداند.")
    return answer.strip()


def has_retrieved_citation(answer: str, chunks: list[dict[str, object]]) -> bool:
    """Check that a grounded answer names at least one chunk sent to the model."""
    return not chunks or any(f"[{item['id']}]" in answer for item in chunks)


def serve(version_dir: Path, port: int) -> None:
    prompt_path, page_path, knowledge_dir = version_dir / "prompt.py", version_dir / "web" / "index.html", version_dir / "knowledge"

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
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/questions":
                self.send_json(200, {"questions": QUESTIONS})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path not in {"/api/retrieve", "/api/ask"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 180_000:
                    raise ValueError("درخواست بیش از حد بزرگ است؛ گفت‌وگوی تازه را شروع کنید.")
                payload = json.loads(self.rfile.read(length))
                question, mode = payload.get("question"), payload.get("mode")
                if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                    raise ValueError("یک سؤال کوتاه وارد کنید.")
                documents = load_documents(knowledge_dir)
                chunks = retrieve(question.strip(), documents, mode)
                if self.path == "/api/retrieve":
                    self.send_json(200, {"chunks": chunks, "context_characters": sum(item["characters"] for item in chunks)})
                    return
                answer = ask_model(question.strip(), prompt_path, payload.get("history", []), chunks)
                self.send_json(200, {"answer": answer, "chunks": chunks,
                                     "context_characters": sum(item["characters"] for item in chunks),
                                     "citation_ok": has_retrieved_citation(answer, chunks)})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.send_json(502, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped.", flush=True)
