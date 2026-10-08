"""Offline checks for deps flow, tool registration, and user scoping."""

import json
import sys
import unittest
from pathlib import Path

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

EXERCISE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXERCISE / "teacher"))

from agent import SupportDeps, create_agent, load_course_data  # noqa: E402


COURSE = "مبانی یادگیری ماشین"


def deterministic_model_for(course_name: str):
    def deterministic_model(messages: list, info: AgentInfo) -> ModelResponse:
        last_parts = getattr(messages[-1], "parts", [])
        tool_return = next((part for part in last_parts if getattr(part, "part_kind", "") == "tool-return"), None)
        if tool_return is None:
            return ModelResponse(parts=[ToolCallPart("get_course_access", {"course_name": course_name})])
        content = tool_return.content
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return ModelResponse(parts=[TextPart(text)])

    return deterministic_model


class TeacherAgentTests(unittest.TestCase):
    def test_agent_calls_tool_and_returns_its_result_for_each_user(self) -> None:
        records = load_course_data()
        outputs = {}
        for user_id in ("u-104", "u-208"):
            agent = create_agent(FunctionModel(deterministic_model_for(COURSE)))
            result = agent.run_sync(
                f"وضعیت دسترسی من به دورهٔ {COURSE} چیست؟",
                deps=SupportDeps(user_id=user_id, course_data=records),
            )
            outputs[user_id] = json.loads(result.output)
            parts = [part for message in result.all_messages() for part in getattr(message, "parts", [])]
            self.assertTrue(any(getattr(part, "part_kind", "") == "tool-call" for part in parts))
            self.assertTrue(any(getattr(part, "part_kind", "") == "tool-return" for part in parts))

        self.assertEqual(outputs["u-104"]["access_status"], "inactive")
        self.assertEqual(outputs["u-208"]["access_status"], "active")
        self.assertEqual(outputs["u-104"]["user_id"], "u-104")
        self.assertEqual(outputs["u-208"]["user_id"], "u-208")

    def test_unknown_course_returns_not_found(self) -> None:
        agent = create_agent(FunctionModel(deterministic_model_for("دورهٔ ناموجود")))
        result = agent.run_sync(
            "این دوره را بررسی کن",
            deps=SupportDeps(user_id="u-104", course_data=load_course_data()),
        )
        self.assertEqual(json.loads(result.output)["found"], False)


if __name__ == "__main__":
    unittest.main()
