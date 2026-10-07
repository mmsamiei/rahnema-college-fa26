"""A small bounded agent loop for the Rahnema support workshop."""

import json
import os
from pathlib import Path
from typing import Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CURRENT_USER_ID = "u-104"  # trusted demo identity; never supplied by the model
MAX_TOOL_STEPS = 5

PROMPT = """شما دستیار پشتیبانی پلتفرم فرضی رهنما هستید.
کاربر می‌گوید پول دوره از حسابش کم شده اما دسترسی ندارد. با ابزارها شواهد را مرحله‌به‌مرحله بررسی کنید.
برای فهمیدن دوره و شناسه‌ها از list_my_enrollments استفاده کنید؛ بعد وضعیت پرداخت و دسترسی همان ثبت‌نام را جداگانه بخوانید.
فقط اگر پرداخت paid و دسترسی inactive است، یک تیکت برای همان ثبت‌نام بسازید.
اگر پرداخت ناموفق است یا دسترسی فعال است، تیکت نسازید و نتیجه را توضیح دهید.
اگر ابزار تیکت گفت تیکت باز از قبل وجود دارد، تیکت دیگری نسازید.
نتیجهٔ هر ابزار را بخوانید و سپس تصمیم بگیرید ابزار بعدی لازم است یا پاسخ نهایی کافی است.
اگر دوره یا ثبت‌نام مبهم است، قبل از ساخت تیکت سؤال روشن‌کننده بپرسید.
اطلاعات فقط مربوط به کاربر u-104 است. نتیجه‌ای را که ابزارها تأیید نکرده‌اند ادعا نکنید.
پس از حداکثر ۵ اجرای ابزار، کار را متوقف کنید و آنچه را بررسی شده و هر بخش ناتمام را بگویید."""

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
    tool_steps = 0
    # TODO 1: تا وقتی به MAX_TOOL_STEPS نرسیده‌اید، مدل را با messages و TOOLS صدا بزنید.
    # TODO 2: اگر مدل tool_calls نداد، متنش پاسخ نهایی است و از حلقه خارج شوید.
    # TODO 3: هر بار فقط یک ابزار را اجرا کنید؛ درخواست مدل و نتیجهٔ ابزار را به messages اضافه کنید.
    # TODO 4: شمارنده را زیاد کنید و در پایان سقف، یک پاسخ نهایی بدون ابزار از مدل بگیرید.
    raise NotImplementedError("حلقهٔ Agent را طبق TODOهای این تابع پیاده کنید.")


def reset_data() -> None:
    for filename in ("tickets.json",):
        seed = (DATA / filename.replace(".json", ".seed.json")).read_text(encoding="utf-8")
        (DATA / filename).write_text(seed, encoding="utf-8")
