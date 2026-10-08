"""Three small agents: one coordinator and two limited specialists."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parent


class CatalogReport(BaseModel):
    recommendations: list[str] = Field(default_factory=list)
    note: str


class AccountReport(BaseModel):
    course_name: str
    payment_status: str
    enrollment_status: str
    access_enabled: bool | None
    mismatch: bool
    evidence: list[str] = Field(default_factory=list)
    note: str


@dataclass
class CatalogDeps:
    """Only the public course catalog is passed to this specialist."""

    courses: list[dict[str, Any]]
    trace: list[dict[str, str]]


@dataclass
class AccountDeps:
    """Only the selected user's account records are passed to this specialist."""

    user_id: str
    payments: list[dict[str, Any]]
    enrollments: list[dict[str, Any]]
    trace: list[dict[str, str]]


@dataclass
class SupportDeps:
    """Data available to the coordinator; it passes a smaller view to each specialist."""

    user_id: str
    courses: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    enrollments: list[dict[str, Any]]
    trace: list[dict[str, str]]


def load_data() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    courses = json.loads((ROOT / "data" / "courses.json").read_text(encoding="utf-8"))
    accounts = json.loads((ROOT / "data" / "accounts.json").read_text(encoding="utf-8"))
    return courses, accounts


def settings() -> tuple[str, str, str]:
    values: dict[str, str] = {}
    env_path = ROOT.parents[2] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")

    get = lambda key: os.environ.get(key) or values.get(key, "")
    base_url, api_key, model_name = get("LLM_BASE_URL").rstrip("/"), get("LLM_API_KEY"), get("LLM_MODEL")
    if not (base_url and api_key and model_name):
        raise ValueError("LLM_BASE_URL، LLM_API_KEY و LLM_MODEL را در .env تنظیم کنید.")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    if not (base_url.startswith("https://") or base_url.startswith("http://127.0.0.1")
            or base_url.startswith("http://localhost")):
        raise ValueError("برای سرویس بیرونی، LLM_BASE_URL باید HTTPS باشد.")
    return base_url, api_key, model_name


def make_model() -> OpenAIChatModel:
    base_url, api_key, model_name = settings()
    return OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


CATALOG_PROMPT = """تو متخصص معرفی دوره‌های رهنما هستی.
برای پیشنهاد، حتماً از search_courses استفاده کن و فقط دوره‌های برگشتی را معرفی کن.
به سطح دوره و پیش‌نیازها دقت کن. اگر گزینهٔ مناسبی پیدا نشد، صریح بگو.
در فیلد recommendations فقط نام دقیق دوره‌ها را به‌صورت فهرست رشته‌ها برگردان؛ توضیح تناسب و پیش‌نیاز را در note بنویس.
تو به اطلاعات کاربران، پرداخت و ثبت‌نام دسترسی نداری."""

ACCOUNT_PROMPT = """تو متخصص بررسی حساب کاربر رهنما هستی.
برای بررسی یک دوره، هر دو ابزار پرداخت و ثبت‌نام را اجرا کن.
فقط داده‌های ابزارها را گزارش کن؛ پرداخت موفق را با دسترسی فعال یکی ندان.
اگر پرداخت موفق است ولی دسترسی خاموش است، mismatch=true ثبت کن.
از خاموش بودن دسترسی، علت فنی را حدس نزن؛ اگر داده‌ای علت را نمی‌گوید، علت نامشخص است.
اگر رکوردی پیدا نشد یا پرداخت ناموفق بود، مغایرت را حدس نزن.
تو به کاتالوگ کامل دوره‌ها دسترسی نداری و نباید برای کاربر پاسخ نهایی بنویسی."""

SUPPORT_PROMPT = """تو دستیار اصلی پشتیبانی رهنما هستی.
درخواست کاربر ممکن است بررسی حساب، پیشنهاد دوره، یا هر دو را بخواهد.
برای بررسی پرداخت/دسترسی از investigate_account و برای پیشنهاد آموزشی از recommend_courses استفاده کن.
اگر هر دو بخش خواسته شده، هر دو ایجنت متخصص را صدا بزن و گزارش‌ها را ترکیب کن.
سؤال عمومی را خودت پاسخ بده و بی‌دلیل متخصص‌ها را صدا نزن.
فقط گزارش متخصص‌ها را مبنای ادعای واقعی قرار بده؛ پاسخ نهایی را خودت به فارسی و یکپارچه بنویس.
در این تمرین هیچ تیکت یا ارجاعی ثبت نمی‌شود؛ هرگز نگو موردی را ارجاع داده‌ای یا دسترسی را فعال می‌کنی.
اگر گزارش فقط نشان می‌دهد دسترسی خاموش است، همان را بگو و علت فنی را نامشخص اعلام کن.
هیچ داده‌ای را تغییر نده."""


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("ي", "ی").replace("ك", "ک").split())


def create_agents(model: Any | None = None) -> Agent[SupportDeps, str]:
    if model is None:
        model = make_model()

    catalog_agent = Agent(
        model,
        name="course_catalog_specialist",
        deps_type=CatalogDeps,
        output_type=CatalogReport,
        instructions=CATALOG_PROMPT,
    )

    @catalog_agent.tool
    def search_courses(ctx: RunContext[CatalogDeps], query: str) -> list[dict[str, Any]]:
        """Search the available course catalog by topic, level, or prerequisite."""
        query_words = set(re.findall(r"[\w]+", normalize(query)))
        aliases = {
            "ai": {"هوش", "مصنوعی", "یادگیری", "ماشین", "llm"},
            "شروع": {"مقدماتی", "مبانی", "شروع"},
            "داده": {"داده", "تحلیل", "pandas", "sql"},
        }
        expanded_words = set(query_words)
        for key, words in aliases.items():
            if key in query_words or (key == "ai" and {"هوش", "مصنوعی"}.issubset(query_words)):
                expanded_words.update(words)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for course in ctx.deps.courses:
            searchable = normalize(" ".join([
                course["title"], course["category"], course["level"], course["summary"],
                " ".join(course["prerequisites"]), " ".join(course["tags"]),
            ]))
            words = set(re.findall(r"[\w]+", searchable))
            score = len(expanded_words & words)
            if score:
                ranked.append((score, course))
        ranked.sort(key=lambda item: (-item[0], item[1]["course_id"]))
        matches = [course for _, course in ranked[:6]]
        ctx.deps.trace.append({
            "kind": "specialist-tool",
            "title": "ایجنت کاتالوگ ← ابزار search_courses",
            "details": json.dumps({"query": query, "matches": matches}, ensure_ascii=False, indent=2),
        })
        return matches

    account_agent = Agent(
        model,
        name="user_account_specialist",
        deps_type=AccountDeps,
        output_type=AccountReport,
        instructions=ACCOUNT_PROMPT,
    )

    def find_record(rows: list[dict[str, Any]], user_id: str, course_name: str) -> dict[str, Any] | None:
        requested = normalize(course_name)
        return next((row for row in rows
                     if row["user_id"] == user_id
                     and normalize(row["course_name"]) in requested), None)

    @account_agent.tool
    def get_my_payment(ctx: RunContext[AccountDeps], course_name: str) -> dict[str, Any]:
        """Read the selected user's payment for the named course."""
        row = find_record(ctx.deps.payments, ctx.deps.user_id, course_name)
        result = {"found": False, "course_name": course_name} if row is None else {"found": True, **row}
        ctx.deps.trace.append({
            "kind": "specialist-tool", "title": "ایجنت حساب ← ابزار get_my_payment",
            "details": json.dumps(result, ensure_ascii=False, indent=2),
        })
        return result

    @account_agent.tool
    def get_my_enrollment(ctx: RunContext[AccountDeps], course_name: str) -> dict[str, Any]:
        """Read the selected user's enrollment and access status."""
        row = find_record(ctx.deps.enrollments, ctx.deps.user_id, course_name)
        result = {"found": False, "course_name": course_name} if row is None else {"found": True, **row}
        ctx.deps.trace.append({
            "kind": "specialist-tool", "title": "ایجنت حساب ← ابزار get_my_enrollment",
            "details": json.dumps(result, ensure_ascii=False, indent=2),
        })
        return result

    support_agent = Agent(
        model,
        name="support_coordinator",
        deps_type=SupportDeps,
        instructions=SUPPORT_PROMPT,
    )

    @support_agent.tool
    async def investigate_account(ctx: RunContext[SupportDeps], course_name: str) -> AccountReport:
        """Delegate payment and access investigation to the account specialist."""
        ctx.deps.trace.append({
            "kind": "delegation", "title": "دستیار اصلی واگذار می‌کند: ایجنت حساب",
            "details": f"درخواست بررسی {course_name!r} برای کاربر {ctx.deps.user_id!r}. فقط دادهٔ حساب همین کاربر به متخصص می‌رسد.",
        })
        try:
            result = await account_agent.run(
                f"وضعیت پرداخت و دسترسی کاربر جاری برای دورهٔ «{course_name}» را بررسی کن.",
                deps=AccountDeps(
                    ctx.deps.user_id,
                    [row for row in ctx.deps.payments if row["user_id"] == ctx.deps.user_id],
                    [row for row in ctx.deps.enrollments if row["user_id"] == ctx.deps.user_id],
                    ctx.deps.trace,
                ),
                usage=ctx.usage,
            )
        except Exception as exc:
            ctx.deps.trace.append({
                "kind": "error", "title": "خطا داخل ایجنت حساب",
                "details": f"{type(exc).__name__}: {exc}",
            })
            raise
        ctx.deps.trace.append({
            "kind": "delegation-result", "title": "ایجنت حساب گزارش می‌دهد",
            "details": result.output.model_dump_json(indent=2),
        })
        return result.output

    @support_agent.tool
    async def recommend_courses(ctx: RunContext[SupportDeps], learning_goal: str) -> CatalogReport:
        """Delegate course matching to the catalog specialist."""
        ctx.deps.trace.append({
            "kind": "delegation", "title": "دستیار اصلی واگذار می‌کند: ایجنت کاتالوگ",
            "details": f"هدف یادگیری: {learning_goal!r}. متخصص فقط فهرست عمومی دوره‌ها را می‌گیرد.",
        })
        try:
            result = await catalog_agent.run(
                f"برای این هدف دوره‌های مناسب را پیشنهاد بده: {learning_goal}",
                deps=CatalogDeps(ctx.deps.courses, ctx.deps.trace),
                usage=ctx.usage,
            )
        except Exception as exc:
            ctx.deps.trace.append({
                "kind": "error", "title": "خطا داخل ایجنت کاتالوگ",
                "details": (f"{type(exc).__name__}: {exc}; cause="
                            f"{type(exc.__cause__).__name__}: {exc.__cause__!r}"),
            })
            raise
        ctx.deps.trace.append({
            "kind": "delegation-result", "title": "ایجنت کاتالوگ پیشنهاد می‌دهد",
            "details": result.output.model_dump_json(indent=2),
        })
        return result.output

    return support_agent


def answer(user_id: str, question: str) -> dict[str, Any]:
    courses, accounts = load_data()
    trace: list[dict[str, str]] = [{
        "kind": "setup", "title": "برنامه، عامل اصلی را با دسترسی‌های محدود می‌سازد",
        "details": f"کاربر آزمایشی: {user_id}. ایجنت حساب فقط رکوردهای همین کاربر را می‌خواند؛ ایجنت کاتالوگ فقط دوره‌های عمومی را می‌بیند.",
    }]
    deps = SupportDeps(
        user_id=user_id,
        courses=courses,
        payments=accounts["payments"],
        enrollments=accounts["enrollments"],
        trace=trace,
    )
    try:
        result = create_agents().run_sync(question, deps=deps)
    except Exception as exc:
        trace.append({"kind": "error", "title": "اجرای دستیار اصلی شکست خورد",
                      "details": f"{type(exc).__name__}: {exc}"})
        for event in trace:
            print(f"\n[{event['title']}]\n{event['details']}", flush=True)
        raise
    trace.append({"kind": "answer", "title": "دستیار اصلی پاسخ نهایی را می‌سازد", "details": str(result.output)})
    for event in trace:
        print(f"\n[{event['title']}]\n{event['details']}", flush=True)
    return {"answer": str(result.output), "events": trace}
