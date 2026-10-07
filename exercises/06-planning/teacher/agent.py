"""A bounded agent loop that records a fresh plan before each decision."""

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
در درخواست اصلی، کاربر می‌خواهد وضعیت پرداخت و دسترسی دورهٔ مبانی یادگیری ماشین را بررسی کنید و اگر پرداخت موفق اما دسترسی غیرفعال است تیکت بسازید؛ سپس وضعیت پرداخت و دسترسی دورهٔ مبانی هوش مصنوعی را هم گزارش کنید.
دامنهٔ درخواست‌های دیگر را از متن کاربر تشخیص دهید؛ هیچ وضعیت پرداخت یا دسترسی را بدون نتیجهٔ ابزار گزارش نکنید.
در ابتدای هر دور برنامهٔ کوتاه و عملیاتیِ کارهای باقی‌مانده را با record_plan ثبت می‌کنید. این متن خلاصهٔ قابل‌نمایش برنامه است، نه زنجیرهٔ فکر خصوصی.
پس از هر مشاهده، برنامه را بر اساس شواهد تازه به‌روز کنید. قدم اول برنامه باید گام بعدی را مشخص کند؛ اگر کار تمام است، بنویسید آمادهٔ پاسخ هستید.
در مرحلهٔ اقدام، هر بار فقط یک ابزار فراخوانی کنید. ابزار بعدی را پس از دیدن نتیجه، در دور بعد انتخاب کنید.
از ابزارهای ثبت‌نام، پرداخت و دسترسی برای پیدا کردن شناسه‌ها و بررسی هر دو دوره استفاده کنید.
فقط اگر کاربر تیکت خواسته، پرداخت موفق و دسترسی غیرفعال است تیکت بسازید. ابزار ساخت تیکت خودش وجود تیکت تکراری را بررسی می‌کند؛ ابزار جداگانه‌ای برای فهرست تیکت‌ها ندارید.
اگر دسترسی فعال است، پرداخت ناموفق است، یا تیکت باز از قبل وجود دارد، تیکت تازه نسازید.
تیکت فقط برای ثبت‌نام متعلق به کاربر u-104 ساخته می‌شود؛ نتیجهٔ ابزار Python مرجع نهایی است.
اگر نام دوره یا ثبت‌نام مبهم است، سؤال روشن‌کننده بپرسید. چیزی را که ابزار تأیید نکرده ادعا نکنید.
حداکثر ۷ اجرای ابزار عملیاتی دارید. پس از آن بدون ابزار جمع‌بندی کنید و بخش‌های ناتمام را بگویید."""

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
PLAN_TOOL = {"type": "function", "function": {"name": "record_plan",
    "description": "برنامهٔ کوتاه و مرتبِ کارهای باقی‌مانده را ثبت می‌کند. گام اول، اقدام بعدی است. فقط شواهد قابل مشاهده را ذکر کنید، نه زنجیرهٔ فکر خصوصی.",
    "parameters": {"type": "object", "properties": {
        "steps": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"}},
        "required": ["steps", "evidence"], "additionalProperties": False}}}


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
    try:
        function = tool_call["function"]
        if function["name"] != "record_plan":
            raise ValueError("در ابتدای هر دور فقط record_plan مجاز است.")
        arguments = json.loads(function["arguments"])
        steps, evidence = arguments["steps"], arguments["evidence"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("خروجی Planning قالب درستی ندارد.") from exc
    if (not isinstance(steps, list) or not 1 <= len(steps) <= 7
            or any(not isinstance(step, str) or not step.strip() or len(step) > 200 for step in steps)
            or not isinstance(evidence, str) or len(evidence) > 500):
        raise ValueError("برنامه باید ۱ تا ۷ گام کوتاه و شواهد حداکثر ۵۰۰ نویسه‌ای داشته باشد.")
    plan = {"version": len(plans) + 1, "steps": steps, "evidence": evidence}
    plans.append(plan)
    return {"saved": True, "version": plan["version"]}


def tool_answer(result: Optional[dict]) -> str:
    if not result:
        return "بررسی تمام شد؛ نتیجه‌ای برای نمایش از ابزار دریافت نشد."
    if "ticket_id" in result and result.get("ok"):
        return f"تیکت {result['ticket_id']} ثبت شد."
    if result.get("ok") is False:
        return "تیکت ساخته نشد: " + result.get("message", "شرایط ساخت تیکت برقرار نبود.")
    return "نتیجهٔ آخرین ابزار: " + json.dumps(result, ensure_ascii=False)


def requested_scope(question: str) -> tuple[list[str], bool, bool, bool]:
    text = question.lower()
    courses = []
    if "یادگیری ماشین" in text:
        courses.append("مبانی یادگیری ماشین")
    if "هوش مصنوعی" in text:
        courses.append("مبانی هوش مصنوعی")
    if "python" in text or "پایتون" in text:
        courses.append("Python برای تحلیل داده")
    if not courses:
        courses = ["مبانی یادگیری ماشین", "مبانی هوش مصنوعی"]
    wants_ticket = "تیکت" in text and not any(phrase in text for phrase in ("تیکت نساز", "تیکت نمی‌خوام", "بدون تیکت"))
    wants_payment = any(word in text for word in ("پرداخت", "پول", "حسابم", "برداشت")) or wants_ticket
    wants_access = any(word in text for word in ("دسترسی", "ورود", "باز نمی")) or wants_ticket
    if not wants_payment and not wants_access:
        wants_payment = wants_access = True
    return courses, wants_payment, wants_access, wants_ticket


def missing_evidence(question: str, observations: dict) -> list[str]:
    courses, wants_payment, wants_access, wants_ticket = requested_scope(question)
    if observations["enrollments"] is None:
        return ["خواندن فهرست ثبت‌نام‌های کاربر"]
    missing = []
    rows = observations["enrollments"]
    for course in courses:
        enrollment = next((row for row in rows if row["course"] == course), None)
        if enrollment is None:
            continue  # The list result is evidence that this course is not enrolled.
        if wants_payment and enrollment["order_id"] not in observations["payments"]:
            missing.append(f"وضعیت پرداخت {course}")
        if wants_access and enrollment["enrollment_id"] not in observations["access"]:
            missing.append(f"وضعیت دسترسی {course}")
        if (wants_ticket and wants_payment and wants_access
                and observations["payments"].get(enrollment["order_id"], {}).get("payment", {}).get("status") == "paid"
                and observations["access"].get(enrollment["enrollment_id"], {}).get("access", {}).get("status") == "inactive"
                and enrollment["enrollment_id"] not in observations["tickets"]):
            missing.append(f"اقدام تیکت برای {course}")
    return missing


def capture_observation(tool_call: dict, result: dict, observations: dict) -> None:
    name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    if name == "list_my_enrollments":
        observations["enrollments"] = result.get("enrollments", [])
    elif name == "get_payment_status":
        observations["payments"][arguments["order_id"]] = result
    elif name == "get_course_access":
        observations["access"][arguments["enrollment_id"]] = result
    elif name == "create_access_ticket":
        observations["tickets"][arguments["enrollment_id"]] = result


def evidence_summary(question: str, observations: dict) -> str:
    courses, wants_payment, wants_access, wants_ticket = requested_scope(question)
    lines = []
    rows = observations["enrollments"] or []
    for course in courses:
        enrollment = next((row for row in rows if row["course"] == course), None)
        if enrollment is None:
            lines.append(f"{course}: ثبت‌نامی در دادهٔ کاربر پیدا نشد.")
            continue
        details = [course]
        payment = observations["payments"].get(enrollment["order_id"])
        access = observations["access"].get(enrollment["enrollment_id"])
        if wants_payment:
            details.append("پرداخت=" + (payment.get("payment", {}).get("status", "یافت نشد") if payment else "بررسی نشد"))
        if wants_access:
            details.append("دسترسی=" + (access.get("access", {}).get("status", "یافت نشد") if access else "بررسی نشد"))
        ticket = observations["tickets"].get(enrollment["enrollment_id"])
        if ticket:
            details.append("تیکت=" + ticket.get("ticket_id", ticket.get("message", "ثبت شد")))
        elif wants_ticket:
            details.append("تیکت بررسی/ثبت نشد")
        lines.append("، ".join(details))
    return "نتیجهٔ تأییدشده تا اینجا:\n" + "\n".join(lines)


def contradicts_evidence(question: str, answer_text: str, observations: dict) -> bool:
    """Catch explicit status claims that conflict with tool results."""
    courses, wants_payment, wants_access, _ = requested_scope(question)
    text = " " + answer_text.lower() + " "
    aliases = {
        "مبانی یادگیری ماشین": ("مبانی یادگیری ماشین", "یادگیری ماشین", "machine learning"),
        "مبانی هوش مصنوعی": ("مبانی هوش مصنوعی", "هوش مصنوعی", "artificial intelligence", " ai "),
        "Python برای تحلیل داده": ("python برای تحلیل داده", "پایتون برای تحلیل داده", "python"),
    }
    active_claims = ("دسترسی فعال", "دسترسی‌ام فعاله", "دسترسی‌ات فعاله", "access is active", "access: active", "دسترسی: active")
    inactive_claims = ("دسترسی غیرفعال", "دسترسی غیرفعاله", "access is inactive", "access: inactive", "دسترسی: inactive")
    paid_claims = ("پرداخت موفق", "پرداخت شده", "پرداخت انجام شده", "payment is paid", "payment: paid", "paid")
    failed_claims = ("پرداخت ناموفق", "پرداخت نشده", "پرداخت انجام نشده", "payment failed", "payment: failed", "failed")

    for course in courses:
        if not any(alias in text for alias in aliases.get(course, (course.lower(),))):
            continue
        enrollment = next((row for row in observations["enrollments"] or [] if row["course"] == course), None)
        if enrollment is None:
            continue
        if wants_access:
            access = observations["access"].get(enrollment["enrollment_id"], {}).get("access")
            if access:
                status = access.get("status")
                if status == "inactive" and any(claim in text for claim in active_claims):
                    return True
                if status == "active" and any(claim in text for claim in inactive_claims):
                    return True
        if wants_payment:
            payment = observations["payments"].get(enrollment["order_id"], {}).get("payment")
            if payment:
                status = payment.get("status")
                if status == "paid" and any(claim in text for claim in failed_claims):
                    return True
                if status in {"failed", "declined", "unpaid"} and any(claim in text for claim in paid_claims):
                    return True
    return False


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
    incomplete_finals = 0
    observations = {"enrollments": None, "payments": {}, "access": {}, "tickets": {}}

    while tool_steps < MAX_TOOL_STEPS:
        emit(events, "model", f"Planning · دور {tool_steps + 1}",
             "این دور با فراخوانی اجباری record_plan شروع می‌شود.")
        plan_response = ask_model(messages, [PLAN_TOOL],
                                  {"type": "function", "function": {"name": "record_plan"}})
        plan_calls = plan_response.get("tool_calls") or []
        if len(plan_calls) != 1:
            raise ValueError("مدل باید در ابتدای هر دور دقیقاً یک بار record_plan را فراخوانی کند.")
        plan_call = plan_calls[0]
        plan_result = save_plan(plan_call, plans)
        plan = plans[-1]
        plan_details = "شواهد: " + (plan["evidence"] or "درخواست اولیه") + "\nگام‌های باقی‌مانده:\n"
        plan_details += "\n".join(f"{index}. {step}" for index, step in enumerate(plan["steps"], 1))
        emit(events, "plan", f"برنامهٔ نسخهٔ {plan['version']}", plan_details)
        messages.append({"role": "assistant", "content": plan_response.get("content"), "tool_calls": plan_calls})
        messages.append({"role": "tool", "tool_call_id": plan_call["id"],
                         "content": json.dumps(plan_result, ensure_ascii=False)})

        emit(events, "model", f"انتخاب اقدام · دور {tool_steps + 1}",
             "مدل برنامه و نتیجهٔ ثبت آن را می‌بیند و سپس ابزار عملیاتی یا پاسخ نهایی را انتخاب می‌کند.")
        response = ask_model(messages, TOOLS)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            missing = missing_evidence(question, observations)
            if missing:
                incomplete_finals += 1
                emit(events, "warning", "شواهد برای پاسخ کافی نیست", "هنوز باید بررسی شود: " + "؛ ".join(missing))
                if incomplete_finals >= 2:
                    text = evidence_summary(question, observations)
                    emit(events, "answer", "جمع‌بندی داده‌های تأییدشده", text)
                    return {"answer": text, "events": events, "plans": plans}
                messages.append({"role": "system", "content": "پاسخ نهایی را هنوز ندهید. نتیجهٔ Python نشان می‌دهد این موارد هنوز با ابزار بررسی نشده‌اند: " + "؛ ".join(missing) + ". در دور بعد برنامه را به‌روز کنید و ابزار لازم را اجرا کنید."})
                continue
            text = response.get("content") or evidence_summary(question, observations)
            if contradicts_evidence(question, text, observations):
                emit(events, "warning", "پاسخ با مشاهده‌ها سازگار نیست", "وضعیت اعلام‌شده با نتیجهٔ ابزارها تعارض دارد؛ جمع‌بندی از دادهٔ تأییدشده ساخته می‌شود.")
                text = evidence_summary(question, observations)
            emit(events, "answer", "پاسخ نهایی", text)
            return {"answer": text, "events": events, "plans": plans}
        tool_call = tool_calls[0]
        if len(tool_calls) > 1:
            emit(events, "warning", "درخواست چند ابزار", "مدل چند ابزار پیشنهاد داد؛ فقط اولین ابزار اجرا می‌شود و پس از مشاهدهٔ نتیجه، plan و تصمیم بعدی از نو ساخته می‌شوند.")
        selected_calls = [tool_call]
        emit(events, "tool-call", f"درخواست ابزار · گام {tool_steps + 1}",
             json.dumps(tool_call["function"], ensure_ascii=False, indent=2))
        result = run_tool(tool_call)
        capture_observation(tool_call, result, observations)
        emit(events, "tool-result", f"مشاهدهٔ Python · گام {tool_steps + 1}",
             json.dumps(result, ensure_ascii=False, indent=2))
        messages.append({"role": "assistant", "content": response.get("content"), "tool_calls": selected_calls})
        messages.append({"role": "tool", "tool_call_id": tool_call["id"],
                         "content": json.dumps(result, ensure_ascii=False)})
        tool_steps += 1

    emit(events, "limit", "رسیدن به سقف گام‌ها", f"پس از {MAX_TOOL_STEPS} اجرای عملیاتی، دیگر برنامه یا ابزار تازه‌ای اجرا نمی‌شود.")
    final = ask_model(messages)
    missing = missing_evidence(question, observations)
    if missing:
        text = f"پس از {MAX_TOOL_STEPS} اجرای ابزار، حلقه متوقف شد. " + evidence_summary(question, observations)
        text += "\nموارد بررسی‌نشده: " + "؛ ".join(missing)
    else:
        text = final.get("content") or f"بررسی پس از {MAX_TOOL_STEPS} اجرای ابزار متوقف شد. {tool_answer(result)}"
        if contradicts_evidence(question, text, observations):
            emit(events, "warning", "پاسخ با مشاهده‌ها سازگار نیست", "وضعیت اعلام‌شده با نتیجهٔ ابزارها تعارض دارد؛ جمع‌بندی از دادهٔ تأییدشده ساخته می‌شود.")
            text = evidence_summary(question, observations)
    emit(events, "answer", "پاسخ نهایی", text)
    return {"answer": text, "events": events, "plans": plans}


def reset_data() -> None:
    for filename in ("tickets.json",):
        seed = (DATA / filename.replace(".json", ".seed.json")).read_text(encoding="utf-8")
        (DATA / filename).write_text(seed, encoding="utf-8")
