"""A bounded agent loop for the Planning exercise scaffold."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CURRENT_USER_ID = "u-104"  # trusted demo identity; never supplied by the model
MAX_TOOL_STEPS = 7

PROMPT = """شما دستیار پشتیبانی پلتفرم فرضی رهنما هستید.
در تصمیم اول فقط یکی را انتخاب کنید: final_answer برای پاسخ مستقیم، یا record_plan برای کاری که به بررسی و ابزار نیاز دارد.
برای سلام یا گفت‌وگوی عمومی، final_answer مناسب است. اگر record_plan را انتخاب کردید، برنامهٔ کوتاه و مرتبِ کارهای باقی‌مانده را ثبت کنید؛ این متن خلاصهٔ قابل‌نمایش برنامه است، نه زنجیرهٔ فکر خصوصی.
پس از ثبت برنامه، نتیجهٔ آن را در تاریخچه می‌بینید و در هر دور فقط یک ابزار عملیاتی انتخاب می‌کنید یا پاسخ نهایی می‌دهید. ابزار بعدی را پس از دیدن نتیجهٔ ابزار قبلی انتخاب کنید.
فقط دوره‌ها و اقدام‌های خواسته‌شده در پیام کاربر را بررسی کنید. از ابزارهای ثبت‌نام، پرداخت و دسترسی برای پیدا کردن شناسه‌ها و بررسی وضعیت استفاده کنید. هیچ وضعیت پرداخت یا دسترسی را بدون نتیجهٔ ابزار گزارش نکنید.
حساب آزمایشی فعلی u-104 است و Python آن را به ابزارها می‌دهد؛ برای بررسی از کاربر شماره موبایل یا شناسهٔ حساب نخواهید. در evidence فقط متن درخواست کاربر و نتیجهٔ ابزارهای واقعاً اجراشده را بیاورید.
فقط اگر کاربر تیکت خواسته و پرداخت موفق و دسترسی غیرفعال است تیکت بسازید. ابزار ساخت تیکت مالکیت، وضعیت داده و تکراری‌نبودن تیکت را دوباره بررسی می‌کند.
اگر درخواست مبهم است، سؤال روشن‌کننده بپرسید. نتیجهٔ ابزار Python مرجع نهایی است. حداکثر ۷ اجرای ابزار عملیاتی دارید؛ پس از آن بدون ابزار جمع‌بندی کنید و بخش‌های ناتمام را بگویید."""

LIST_ENROLLMENTS = {"type": "function", "function": {"name": "list_my_enrollments",
    "description": "ثبت‌نام‌های کاربر فعلی را همراه شناسهٔ ثبت‌نام، سفارش و نام دوره فهرست می‌کند.",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
GET_PAYMENT = {"type": "function", "function": {"name": "get_payment_status",
    "description": "وضعیت پرداخت سفارش متعلق به کاربر فعلی را می‌خواند.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"], "additionalProperties": False}}}
GET_ACCESS = {"type": "function", "function": {"name": "get_course_access",
    "description": "وضعیت دسترسی کاربر فعلی به دورهٔ یک ثبت‌نام را می‌خواند.",
    "parameters": {"type": "object", "properties": {"enrollment_id": {"type": "string"}},
                    "required": ["enrollment_id"], "additionalProperties": False}}}
CREATE_TICKET = {"type": "function", "function": {"name": "create_access_ticket",
    "description": "برای مغایرت پرداخت موفق و دسترسی غیرفعال تیکت می‌سازد. Python مالکیت و وضعیت‌ها را بررسی می‌کند؛ تیکت باز تکراری نمی‌سازد.",
    "parameters": {"type": "object", "properties": {
        "enrollment_id": {"type": "string"}, "issue_summary": {"type": "string"}},
        "required": ["enrollment_id", "issue_summary"], "additionalProperties": False}}}
TOOLS = [LIST_ENROLLMENTS, GET_PAYMENT, GET_ACCESS, CREATE_TICKET]
# TODO 1: schema ابزار اجباری record_plan با steps و evidence را تعریف کنید.
PLAN_TOOL = None
FINAL_TOOL = {"type": "function", "function": {"name": "final_answer",
    "description": "وقتی برای پیام فعلی نیازی به بررسی با ابزار نیست، پاسخ نهایی را ثبت می‌کند.",
    "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                    "required": ["answer"], "additionalProperties": False}}}


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


def ask_model(messages: list[dict], tools: Optional[list[dict]] = None,
              tool_choice: object = "auto") -> dict:
    base, key, model, timeout = settings()
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    body = {"model": model, "messages": messages}
    if tools:
        body.update({"tools": tools, "tool_choice": tool_choice, "parallel_tool_calls": False})
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


def list_my_enrollments() -> dict:
    enrollments = [row for row in read_rows("enrollments.json") if row["user_id"] == CURRENT_USER_ID]
    return {"enrollments": [{"enrollment_id": row["enrollment_id"], "order_id": row["order_id"],
                             "course": row["course"], "status": row["status"]} for row in enrollments]}


def get_payment_status(order_id: str) -> dict:
    owns_order = any(row["order_id"] == order_id and row["user_id"] == CURRENT_USER_ID
                     for row in read_rows("enrollments.json"))
    if not owns_order:
        return {"found": False, "message": "این سفارش متعلق به حساب آزمایشی فعلی نیست یا پیدا نشد."}
    payment = next((row for row in read_rows("payments.json") if row["order_id"] == order_id), None)
    return {"found": payment is not None, "payment": payment}


def get_course_access(enrollment_id: str) -> dict:
    owns_enrollment = any(row["enrollment_id"] == enrollment_id and row["user_id"] == CURRENT_USER_ID
                          for row in read_rows("enrollments.json"))
    if not owns_enrollment:
        return {"found": False, "message": "این ثبت‌نام متعلق به حساب آزمایشی فعلی نیست یا پیدا نشد."}
    access = next((row for row in read_rows("access.json") if row["enrollment_id"] == enrollment_id), None)
    return {"found": access is not None, "access": access}


def create_access_ticket(enrollment_id: str, issue_summary: str) -> dict:
    enrollment = next((row for row in read_rows("enrollments.json")
                       if row["enrollment_id"] == enrollment_id and row["user_id"] == CURRENT_USER_ID), None)
    if enrollment is None:
        return {"ok": False, "message": "ثبت‌نام متعلق به حساب آزمایشی فعلی نیست یا پیدا نشد."}
    payment = next((row for row in read_rows("payments.json") if row["order_id"] == enrollment["order_id"]), None)
    access = next((row for row in read_rows("access.json") if row["enrollment_id"] == enrollment_id), None)
    if payment is None or payment["status"] != "paid":
        return {"ok": False, "message": "پرداخت موفق ثبت نشده است؛ برای این سناریو تیکت دسترسی ساخته نمی‌شود."}
    if access is None or access["status"] == "active":
        return {"ok": False, "message": "مغایرت دسترسی وجود ندارد؛ دسترسی فعال است یا رکورد آن پیدا نشد."}
    tickets = read_rows("tickets.json")
    existing = next((row for row in tickets if row["enrollment_id"] == enrollment_id and row["status"] == "open"), None)
    if existing:
        return {"ok": True, "created": False, "ticket_id": existing["ticket_id"],
                "message": "برای این ثبت‌نام تیکت باز وجود داشت؛ تیکت تکراری ساخته نشد."}
    ticket_number = max([5000] + [int(row["ticket_id"].split("-")[-1]) for row in tickets]) + 1
    ticket = {"ticket_id": f"ticket-{ticket_number}", "user_id": CURRENT_USER_ID,
              "enrollment_id": enrollment_id, "order_id": enrollment["order_id"],
              "issue_summary": issue_summary[:500], "status": "open"}
    tickets.append(ticket)
    write_rows("tickets.json", tickets)
    return {"ok": True, "created": True, "ticket_id": ticket["ticket_id"], "status": "open",
            "message": "تیکت پشتیبانی ثبت شد."}


def run_tool(tool_call: dict) -> dict:
    try:
        name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("درخواست ابزار قالب درستی ندارد.") from exc
    functions = {"list_my_enrollments": list_my_enrollments, "get_payment_status": get_payment_status,
                 "get_course_access": get_course_access, "create_access_ticket": create_access_ticket}
    if name not in functions or not isinstance(arguments, dict):
        raise ValueError("نام ابزار یا آرگومان‌های آن معتبر نیستند.")
    try:
        return functions[name](**arguments)
    except TypeError as exc:
        raise ValueError("آرگومان‌های ابزار با تعریف تابع هماهنگ نیستند.") from exc


def save_plan(tool_call: dict, plans: list[dict]) -> dict:
    # TODO 2: نام ابزار و JSON ورودی را اعتبارسنجی کنید؛ plan نسخه‌دار را به plans بیفزایید.
    raise NotImplementedError("ثبت برنامه را پیاده‌سازی کنید.")


def final_answer_from_call(tool_call: dict) -> str:
    try:
        if tool_call["function"]["name"] != "final_answer":
            raise ValueError("تصمیم اول باید record_plan یا final_answer باشد.")
        answer = json.loads(tool_call["function"]["arguments"])["answer"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("پاسخ نهایی قالب درستی ندارد.") from exc
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 8000:
        raise ValueError("متن پاسخ نهایی معتبر نیست.")
    return answer.strip()


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
    events = []
    plans = []
    tool_steps = 0

    if PLAN_TOOL is not None:
        # TODO 3: مدل را با [PLAN_TOOL, FINAL_TOOL] و tool_choice="required" صدا بزنید.
        # اگر final_answer برگشت، متن آن را با final_answer_from_call بخوانید و پایان دهید.
        # اگر record_plan برگشت، save_plan را اجرا و plan را در events ثبت کنید.
        # فراخوانی plan و نتیجهٔ آن را با همان tool_call_id به messages بیفزایید.
        pass

    while tool_steps < MAX_TOOL_STEPS:
        emit(events, "model", f"انتخاب اقدام · گام {tool_steps + 1}",
             "مدل با دیدن تاریخچه و نتیجهٔ ابزارهای قبلی، یک ابزار یا پاسخ نهایی انتخاب می‌کند.")
        response = ask_model(messages, TOOLS)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            text = response.get("content")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("مدل پاسخ نهایی متنی برنگرداند.")
            emit(events, "answer", "پاسخ نهایی", text)
            return {"answer": text, "events": events, "plans": plans}
        tool_call = tool_calls[0]
        if len(tool_calls) > 1:
            emit(events, "warning", "درخواست چند ابزار",
                 "در هر گام فقط ابزار اول اجرا می‌شود؛ مدل بعد از دیدن نتیجه دوباره تصمیم می‌گیرد.")
        emit(events, "tool-call", f"درخواست ابزار · گام {tool_steps + 1}",
             json.dumps(tool_call["function"], ensure_ascii=False, indent=2))
        result = run_tool(tool_call)
        emit(events, "tool-result", f"مشاهدهٔ Python · گام {tool_steps + 1}",
             json.dumps(result, ensure_ascii=False, indent=2))
        messages.append({"role": "assistant", "content": response.get("content"), "tool_calls": [tool_call]})
        messages.append({"role": "tool", "tool_call_id": tool_call["id"],
                         "content": json.dumps(result, ensure_ascii=False)})
        tool_steps += 1

    emit(events, "limit", "رسیدن به سقف گام‌ها",
         f"پس از {MAX_TOOL_STEPS} اجرای ابزار عملیاتی، مدل بدون ابزار جمع‌بندی می‌کند.")
    final = ask_model(messages)
    text = final.get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("مدل پس از سقف ابزارها پاسخ نهایی متنی برنگرداند.")
    emit(events, "answer", "پاسخ نهایی", text)
    return {"answer": text, "events": events, "plans": plans}


def reset_data() -> None:
    for filename in ("tickets.json",):
        seed = (DATA / filename.replace(".json", ".seed.json")).read_text(encoding="utf-8")
        (DATA / filename).write_text(seed, encoding="utf-8")
