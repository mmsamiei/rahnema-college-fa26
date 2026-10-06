"""Inspectable one-tool-call demo for exercise 03."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """شما دستیار پشتیبانی فارسی پلتفرم فرضی رهنما هستید.
برای اطلاع از وضعیت ثبت‌نام یا پرداخت، ابزار مناسب را صدا بزنید.
فقط بر نتیجهٔ ابزار تکیه کنید؛ اگر رکوردی پیدا نشد، همین را بگویید.
ابزارها فقط برای دادهٔ ساختگی این کارگاه هستند. ادعای انجام لغو یا بازپرداخت نکنید."""

TOOLS = [
    {"type": "function", "function": {"name": "get_enrollment_status",
     "description": "وضعیت دسترسی ثبت‌نام را با شناسهٔ ثبت‌نام می‌خواند.",
     "parameters": {"type": "object", "properties": {"enrollment_id": {"type": "string"}},
                     "required": ["enrollment_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_payment_status",
     "description": "وضعیت پرداخت را با شناسهٔ سفارش می‌خواند.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                     "required": ["order_id"], "additionalProperties": False}}},
]


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


def find_record(filename: str, field: str, value: str) -> Optional[dict]:
    rows = json.loads((ROOT / "data" / filename).read_text(encoding="utf-8"))
    return next((row for row in rows if row[field] == value), None)


def get_enrollment_status(enrollment_id: str) -> dict:
    record = find_record("enrollments.json", "enrollment_id", enrollment_id)
    return {"found": record is not None, "enrollment": record}


def get_payment_status(order_id: str) -> dict:
    record = find_record("payments.json", "order_id", order_id)
    return {"found": record is not None, "payment": record}


def run_tool(tool_call: dict) -> dict:
    try:
        name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("درخواست ابزار قالب درستی ندارد.") from exc
    functions = {"get_enrollment_status": get_enrollment_status, "get_payment_status": get_payment_status}
    if name not in functions:
        raise ValueError("مدل ابزاری را درخواست کرد که در این تمرین تعریف نشده است.")
    if not isinstance(arguments, dict):
        raise ValueError("آرگومان‌های ابزار باید یک شیء JSON باشند.")
    try:
        return functions[name](**arguments)
    except TypeError as exc:
        raise ValueError("آرگومان‌های ابزار با تعریف تابع هماهنگ نیستند.") from exc


def answer(question: str) -> dict:
    events = []
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": question}]
    emit(events, "model", "۱. درخواست اول به مدل", "سؤال کاربر و تعریف دو ابزار خواندنی ارسال می‌شود.")
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

    messages.append({"role": "assistant", "content": first.get("content"), "tool_calls": tool_calls})
    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result, ensure_ascii=False)})
    emit(events, "model", "۴. نتیجهٔ ابزار به مدل برمی‌گردد", "حالا مدل نتیجهٔ JSON را می‌بیند و پاسخ فارسی می‌سازد.")
    final = ask_model(messages)
    text = final.get("content") or "مدل پاسخ متنی برنگرداند."
    emit(events, "answer", "۵. پاسخ نهایی", text)
    return {"answer": text, "events": events}
