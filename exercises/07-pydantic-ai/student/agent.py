"""Student starter: complete the two TODOs to connect deps, Agent, and Tool."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parent
PROMPT = """شما دستیار پشتیبانی پلتفرم فرضی رهنما هستید.
برای گفتن وضعیت دسترسی دوره از ابزار get_course_access استفاده کنید.
فقط اطلاعاتی را گزارش کنید که ابزار برگردانده است. اگر رکوردی پیدا نشد، صریح بگویید."""


# TODO 1: یک dataclass به نام SupportDeps بسازید.
# باید user_id و دادهٔ دوره‌ها را نگه دارد تا Tool بتواند به آن‌ها دسترسی داشته باشد.
class SupportDeps:
    pass


def load_course_data() -> list[dict[str, Any]]:
    path = ROOT / "data" / "course_access.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


def create_agent(model: Any | None = None) -> Agent[SupportDeps, str]:
    """TODO 2: Agent را بسازید و ابزار get_course_access را به آن وصل کنید."""
    if model is None:
        model = make_model()

    # راهنما:
    # - Agent را با deps_type=SupportDeps و instructions=PROMPT بسازید.
    # - با @agent.tool یک تابع get_course_access تعریف کنید.
    # - امضای تابع باید ctx: RunContext[SupportDeps] و course_name: str داشته باشد.
    # - از ctx.deps.user_id و ctx.deps.course_data برای پیدا کردن رکورد استفاده کنید.
    # - اگر رکوردی نبود، نتیجه‌ای روشن با found=False برگردانید.
    raise NotImplementedError("TODO 2 را در exercises/07-pydantic-ai/student/agent.py کامل کنید.")


def trace_events(result: Any, user_id: str) -> list[dict[str, str]]:
    """Render the framework's tool call and tool result for the classroom UI."""
    events = [{
        "kind": "deps",
        "title": "۱. برنامه، وابستگی‌ها را می‌سازد",
        "details": f"SupportDeps(user_id={user_id!r}, course_data=...) به agent.run_sync داده شد.",
    }]
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            part_kind = getattr(part, "part_kind", "")
            if part_kind == "tool-call":
                arguments = getattr(part, "args", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                events.append({
                    "kind": "tool-call",
                    "title": "۲. مدل، Tool را انتخاب می‌کند",
                    "details": json.dumps({"tool": part.tool_name, "arguments": arguments},
                                          ensure_ascii=False, indent=2, default=str),
                })
            elif part_kind == "tool-return":
                events.append({
                    "kind": "tool-result",
                    "title": "۳. Python، Tool را با deps اجرا می‌کند",
                    "details": json.dumps(part.content, ensure_ascii=False, indent=2, default=str),
                })
    events.append({"kind": "answer", "title": "۴. Pydantic AI پاسخ نهایی را برمی‌گرداند",
                   "details": str(result.output)})
    return events


def answer(user_id: str, question: str) -> dict[str, Any]:
    deps = SupportDeps(user_id=user_id, course_data=load_course_data())
    result = create_agent().run_sync(question, deps=deps)
    events = trace_events(result, user_id)
    for event in events:
        print(f"\n[{event['title']}]\n{event['details']}", flush=True)
    return {"answer": str(result.output), "events": events}
