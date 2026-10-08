"""Student starter: complete the TODOs to make one read-only tool call."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """شما دستیار پشتیبانی فارسی پلتفرم فرضی رهنما هستید.
برای وضعیت دسترسی یا پرداخت از ابزار مناسب استفاده کنید.
فقط بر نتیجهٔ ابزار تکیه کنید و اگر رکوردی نیست، صریح بگویید."""

# TODO 1: دو تعریف OpenAI-compatible بسازید:
# get_enrollment_status(enrollment_id) و get_payment_status(order_id).
TOOLS = []


def emit(events: list[dict], kind: str, title: str, details: str) -> None:
    event = {"kind": kind, "title": title, "details": details}
    events.append(event)
    print(f"\n[{title}]\n{details}", flush=True)


def settings() -> tuple[str, str, str, int]:
    values = {}
    env_path = ROOT.parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    get = lambda key, default="": os.environ.get(key) or values.get(key, default)
    base, key, model = get("LLM_BASE_URL").rstrip("/"), get("LLM_API_KEY"), get("LLM_MODEL")
    if not (base and key and model):
        raise ValueError("تنظیمات LLM_BASE_URL، LLM_API_KEY و LLM_MODEL را در .env کامل کنید.")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1") or base.startswith("http://localhost")):
        raise ValueError("برای سرویس بیرونی از HTTPS استفاده کنید.")
    return base, key, model, int(get("LLM_TIMEOUT_SECONDS", "60"))


def ask_model(messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
    base, key, model, timeout = settings()
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    body = {"model": model, "messages": messages}
    if tools:
        body.update({"tools": tools, "tool_choice": "auto", "parallel_tool_calls": False})
    req = request.Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode(),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)["choices"][0]["message"]
    except error.HTTPError as exc:
        raise RuntimeError(f"سرویس مدل خطای HTTP {exc.code} برگرداند.") from exc
    except error.URLError as exc:
        raise RuntimeError("اتصال به سرویس مدل برقرار نشد.") from exc
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("پاسخ سرویس مدل قالب مورد انتظار را نداشت.") from exc


# TODO 2: فایل JSON مناسب را بخوانید و رکورد متناظر با شناسه را پیدا کنید.
def get_enrollment_status(enrollment_id: str) -> dict:
    raise NotImplementedError("خواندن وضعیت ثبت‌نام را پیاده‌سازی کنید.")


def get_payment_status(order_id: str) -> dict:
    raise NotImplementedError("خواندن وضعیت پرداخت را پیاده‌سازی کنید.")


# TODO 3: نام ابزار را به تابع Python مجاز وصل کنید؛ نام فایل یا مسیر را از مدل نگیرید.
def run_tool(tool_call: dict) -> dict:
    raise NotImplementedError("dispatch ابزار را پیاده‌سازی کنید.")


def answer(question: str, history: Optional[list[dict]] = None) -> dict:
    events = []
    messages = [{"role": "system", "content": PROMPT}, *(history or []), {"role": "user", "content": question}]
    emit(events, "model", "۱. درخواست اول به مدل", "تاریخچه، سؤال کاربر و تعریف ابزارهای خواندنی ارسال می‌شود.")
    first = ask_model(messages, TOOLS)
    tool_calls = first.get("tool_calls") or []
    if not tool_calls:
        text = first.get("content") or "برای پاسخ به شناسهٔ ثبت‌نام یا سفارش نیاز دارم."
        emit(events, "answer", "پاسخ مدل", text)
        return {"answer": text, "events": events}
    if len(tool_calls) != 1:
        raise ValueError("در این تمرین هر بار فقط یک ابزار اجرا می‌شود.")

    tool_call = tool_calls[0]
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    emit(events, "tool-call", "۲. مدل درخواست ابزار می‌دهد", json.dumps({"name": name, "arguments": args}, ensure_ascii=False, indent=2))
    result = run_tool(tool_call)
    emit(events, "tool-result", "۳. تابع Python داده را می‌خواند", json.dumps(result, ensure_ascii=False, indent=2))

    # TODO 4: پیام assistant حاوی tool_call و پیام tool حاوی نتیجه را به messages اضافه کنید.
    # سپس نتیجهٔ ask_model را بدون ابزار بگیرید و پاسخ را همراه events برگردانید.
    raise NotImplementedError("برگرداندن نتیجهٔ ابزار به مدل را پیاده‌سازی کنید.")
