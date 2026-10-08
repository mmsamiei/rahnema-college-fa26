"""Small Pydantic AI example: one dependency object and one read-only tool."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parent
PROMPT = """شما دستیار پشتیبانی پلتفرم فرضی رهنما هستید.
برای گفتن وضعیت دسترسی دوره از ابزار get_course_access استفاده کنید.
فقط اطلاعاتی را گزارش کنید که ابزار برگردانده است. اگر رکوردی پیدا نشد، صریح بگویید."""


@dataclass
class SupportDeps:
    """اطلاعاتی که برنامه در اختیار Agent و ابزارهایش می‌گذارد."""

    user_id: str
    course_data: list[dict[str, Any]]


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
    base_url = get("LLM_BASE_URL").rstrip("/")
    api_key = get("LLM_API_KEY")
    model_name = get("LLM_MODEL")
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
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name, provider=provider)


def create_agent(model: Any | None = None) -> Agent[SupportDeps, str]:
    if model is None:
        model = make_model()

    agent = Agent(model, deps_type=SupportDeps, instructions=PROMPT)

    @agent.tool
    def get_course_access(ctx: RunContext[SupportDeps], course_name: str) -> dict[str, Any]:
        """وضعیت دسترسی کاربر فعلی به یک دوره را از دادهٔ آزمایشی می‌خواند."""
        row = next(
            (item for item in ctx.deps.course_data
             if item["user_id"] == ctx.deps.user_id and item["course_name"] == course_name),
            None,
        )
        if row is None:
            return {"found": False, "user_id": ctx.deps.user_id, "course_name": course_name}
        return {"found": True, **row}

    return agent


def trace_events(result: Any, user_id: str) -> list[dict[str, str]]:
    """Turn Pydantic AI's messages into a compact classroom trace."""
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
    events.append({
        "kind": "answer",
        "title": "۴. Pydantic AI پاسخ نهایی را برمی‌گرداند",
        "details": str(result.output),
    })
    return events


def answer(user_id: str, question: str) -> dict[str, Any]:
    agent = create_agent()
    deps = SupportDeps(user_id=user_id, course_data=load_course_data())
    result = agent.run_sync(question, deps=deps)
    events = trace_events(result, user_id)
    for event in events:
        print(f"\n[{event['title']}]\n{event['details']}", flush=True)
    return {"answer": str(result.output), "events": events}
