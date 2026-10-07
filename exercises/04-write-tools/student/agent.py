"""Teacher solution: multi-turn reads and one guarded write action."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CURRENT_USER_ID = "u-104"  # trusted demo identity; never supplied by the model
PROMPT = """شما دستیار پشتیبانی فارسی پلتفرم فرضی رهنما هستید.
برای پرسش دربارهٔ سفارش‌ها از ابزار خواندن استفاده کنید و شناسه‌ها را در پاسخ بیاورید.
تاریخچه را بخوانید تا اشاره‌هایی مثل «دومی» را به سفارش درست وصل کنید.
فقط وقتی پیام فعلی کاربر صریحاً درخواست لغو دارد، ابزار لغو را به‌کار ببرید.
درخواست «حذف سفارش/ثبت‌نام» در این دمو یعنی لغو ثبت‌نام، نه حذف دائمی اطلاعات.
اگر منظور کاربر یا ثبت‌نام هدف مبهم است، سؤال روشن‌کننده بپرسید و هیچ تغییری ندهید.
لغو فقط برای ثبت‌نام pending و پیش از شروع جلسهٔ اول مجاز است.
فقط نتیجهٔ ابزار را گزارش کنید. این تمرین فقط وضعیت ثبت‌نام را لغو می‌کند و پول را بازنمی‌گرداند."""

LIST_ORDERS = {"type": "function", "function": {"name": "list_my_orders",
    "description": "سفارش‌ها و وضعیت ثبت‌نام‌های کاربر فعلی را فهرست می‌کند.",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
GET_ENROLLMENT = {"type": "function", "function": {"name": "get_enrollment_status",
    "description": "وضعیت یک ثبت‌نام متعلق به کاربر فعلی را می‌خواند.",
    "parameters": {"type": "object", "properties": {"enrollment_id": {"type": "string"}},
                    "required": ["enrollment_id"], "additionalProperties": False}}}
GET_PAYMENT = {"type": "function", "function": {"name": "get_payment_status",
    "description": "وضعیت پرداخت یک سفارش متعلق به کاربر فعلی را می‌خواند.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"], "additionalProperties": False}}}
# TODO 1: یک تعریف ابزار برای cancel_enrollment(enrollment_id) اضافه کنید.
CANCEL_ENROLLMENT = None
READ_TOOLS = [LIST_ORDERS, GET_ENROLLMENT, GET_PAYMENT]


def emit(events: list[dict], kind: str, title: str, details: str) -> None:
    event = {"kind": kind, "title": title, "details": details}
    events.append(event)
    print(f"\n[{title}]\n{details}", flush=True)


def settings() -> tuple[str, str, str, int]:
    values = {}
    env_path = ROOT.parents[2] / ".env"
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


def read_rows(filename: str) -> list[dict]:
    return json.loads((DATA / filename).read_text(encoding="utf-8"))


def write_rows(filename: str, rows: list[dict]) -> None:
    (DATA / filename).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_my_orders() -> dict:
    # TODO 2: رکوردهای کاربر CURRENT_USER_ID را از فایل‌ها بخوانید و فهرست کنید.
    raise NotImplementedError("فهرست‌کردن سفارش‌های کاربر را پیاده‌سازی کنید.")


def get_enrollment_status(enrollment_id: str) -> dict:
    record = next((row for row in read_rows("enrollments.json")
                   if row["enrollment_id"] == enrollment_id and row["user_id"] == CURRENT_USER_ID), None)
    return {"found": record is not None, "enrollment": record}


def get_payment_status(order_id: str) -> dict:
    record = next((row for row in read_rows("payments.json")
                   if row["order_id"] == order_id and row["user_id"] == CURRENT_USER_ID), None)
    return {"found": record is not None, "payment": record}


def cancel_enrollment(enrollment_id: str) -> dict:
    # TODO 3: مالکیت و شرایط را بررسی کنید؛ فقط بعد از درخواست صریح، JSON را تغییر دهید.
    raise NotImplementedError("ابزار لغو ثبت‌نام را پیاده‌سازی کنید.")


def action_was_requested(question: str) -> bool:
    text = " ".join(question.lower().split())
    explicit_phrases = (
        "لغو کن", "لغوش کن", "کنسل کن", "کنسلش کن",
        "حذف کن", "حذفش کن", "پاک کن", "پاکش کن",
        "delete it", "delete this", "cancel it", "cancel this",
    )
    return any(phrase in text for phrase in explicit_phrases)


def run_tool(tool_call: dict, allow_cancel: bool) -> dict:
    try:
        name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("درخواست ابزار قالب درستی ندارد.") from exc
    functions = {"list_my_orders": list_my_orders, "get_enrollment_status": get_enrollment_status,
                 "get_payment_status": get_payment_status}
    if allow_cancel:
        # TODO 4: ابزار نوشتن را فقط وقتی پیام فعلی درخواست لغو دارد به dispatch اضافه کنید.
        pass
    if name not in functions or not isinstance(arguments, dict):
        raise ValueError("این ابزار در این پیام مجاز نیست یا آرگومان‌های آن معتبر نیستند.")
    try:
        return functions[name](**arguments)
    except TypeError as exc:
        raise ValueError("آرگومان‌های ابزار با تعریف تابع هماهنگ نیستند.") from exc


def answer(question: str, history: object) -> dict:
    if not isinstance(history, list) or len(history) > 20:
        raise ValueError("تاریخچهٔ مکالمه معتبر نیست یا بیش از حد طولانی است.")
    messages = [{"role": "system", "content": PROMPT}]
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("تاریخچه فقط باید پیام‌های کاربر و دستیار داشته باشد.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 8000:
            raise ValueError("متن یکی از پیام‌های تاریخچه معتبر نیست.")
        messages.append({"role": item["role"], "content": content})
    messages.append({"role": "user", "content": question})
    allow_cancel = action_was_requested(question)
    tools = READ_TOOLS + ([CANCEL_ENROLLMENT] if allow_cancel else [])
    events = []
    emit(events, "model", "درخواست مدل", f"سؤال فعلی + {len(history)} پیام قبلی؛ ابزار لغو {'فعال است' if allow_cancel else 'در دسترس نیست'}.")
    first = ask_model(messages, tools)
    tool_calls = first.get("tool_calls") or []
    if not tool_calls:
        text = first.get("content") or "پاسخی دریافت نشد."
        emit(events, "answer", "پاسخ دستیار", text)
        return {"answer": text, "events": events}
    if len(tool_calls) != 1:
        raise ValueError("در این تمرین هر نوبت فقط یک ابزار اجرا می‌شود.")
    tool_call = tool_calls[0]
    emit(events, "tool-call", "درخواست ابزار از مدل", json.dumps(tool_call["function"], ensure_ascii=False, indent=2))
    result = run_tool(tool_call, allow_cancel)
    emit(events, "tool-result", "نتیجهٔ Python", json.dumps(result, ensure_ascii=False, indent=2))
    messages.append({"role": "assistant", "content": first.get("content"), "tool_calls": tool_calls})
    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result, ensure_ascii=False)})
    emit(events, "model", "درخواست دوم مدل", "نتیجهٔ ابزار همراه تاریخچه برای پاسخ نهایی ارسال می‌شود.")
    final = ask_model(messages)
    text = final.get("content") or "مدل پاسخ متنی برنگرداند."
    emit(events, "answer", "پاسخ دستیار", text)
    return {"answer": text, "events": events}


def reset_data() -> None:
    seed = (DATA / "enrollments.seed.json").read_text(encoding="utf-8")
    (DATA / "enrollments.json").write_text(seed, encoding="utf-8")
