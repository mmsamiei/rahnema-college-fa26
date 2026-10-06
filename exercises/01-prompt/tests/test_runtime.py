"""Check that a follow-up actually reaches the model with prior turns."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"
spec = importlib.util.spec_from_file_location("exercise_runtime", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class FakeResponse:
    def __init__(self, answer):
        self.body = io.BytesIO(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())

    def __enter__(self):
        return self.body

    def __exit__(self, *_):
        self.body.close()


class ConversationTest(unittest.TestCase):
    def test_follow_up_includes_prior_user_and_assistant_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.py"
            prompt.write_text('SYSTEM_PROMPT = "پاسخ بده"\n', encoding="utf-8")
            history = [
                {"role": "user", "content": "مهلت بازگشت چقدر است؟"},
                {"role": "assistant", "content": "به قانون دسترسی ندارم."},
            ]
            captured = {}

            def fake_urlopen(http_request, timeout):
                captured.update(json.loads(http_request.data))
                return FakeResponse("پاسخ نوبت دوم")

            with patch.object(runtime, "configuration", return_value=("https://example.test/v1", "test-key", "test-model", 10)):
                with patch.object(runtime.request, "urlopen", side_effect=fake_urlopen):
                    answer = runtime.ask_model("پس از کجا بفهمم؟", prompt, history)

            self.assertEqual(answer, "پاسخ نوبت دوم")
            self.assertEqual([item["role"] for item in captured["messages"]],
                             ["system", "user", "assistant", "user"])
            self.assertEqual(captured["messages"][-2]["content"], "به قانون دسترسی ندارم.")

    def test_rejects_wrong_history_order(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.py"
            prompt.write_text('SYSTEM_PROMPT = "پاسخ بده"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                runtime.conversation_messages("سؤال تازه", prompt, [
                    {"role": "assistant", "content": "پاسخ"},
                    {"role": "user", "content": "سؤال"},
                ])


if __name__ == "__main__":
    unittest.main()
