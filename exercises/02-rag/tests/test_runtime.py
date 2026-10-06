import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"
spec = importlib.util.spec_from_file_location("rag_runtime", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class FakeResponse:
    def __init__(self):
        self.body = io.BytesIO('{"choices":[{"message":{"content":"پاسخ"}}]}'.encode("utf-8"))

    def __enter__(self):
        return self.body

    def __exit__(self, *_):
        self.body.close()


class RagRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"source": "refund", "title": "بازگشت وجه",
             "text": "# بازگشت وجه\nقاعده هفت روزه.<!-- chunk -->استثنای ویدئوی معیوب تا چهارده روز."},
            {"source": "access", "title": "دسترسی",
             "text": "# دسترسی\nپس از پرداخت، دسترسی تا پانزده دقیقه فعال می‌شود."},
        ]

    def test_manual_chunking_uses_document_markers(self):
        chunks = runtime.split_document(self.documents[0], "manual")
        self.assertEqual([chunk["id"] for chunk in chunks], ["refund#1", "refund#2"])
        self.assertIn("استثنای", chunks[1]["text"])

    def test_retrieval_returns_relevant_source_and_not_all_documents(self):
        chunks = runtime.retrieve("ویدئوی معیوب تا چه زمانی گزارش می‌شود؟", self.documents, "manual")
        self.assertEqual(chunks[0]["source"], "refund")
        self.assertLessEqual(len(chunks), 2)

    def test_full_and_window_have_different_chunk_shapes(self):
        self.assertEqual(len(runtime.split_document(self.documents[0], "full")), 1)
        self.assertGreaterEqual(len(runtime.split_document(self.documents[0], "window")), 1)

    def test_follow_up_history_and_retrieved_context_reach_model(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.py"
            prompt.write_text('SYSTEM_PROMPT = "پاسخ بده"\n', encoding="utf-8")
            captured = {}

            def fake_urlopen(http_request, timeout):
                captured.update(json.loads(http_request.data))
                return FakeResponse()

            with patch.object(runtime, "configuration", return_value=("https://example.test/v1", "key", "model", 10)):
                with patch.object(runtime.request, "urlopen", side_effect=fake_urlopen):
                    runtime.ask_model("پس استثنا چه می‌شود؟", prompt,
                                      [{"role": "user", "content": "بازگشت وجه چیست؟"},
                                       {"role": "assistant", "content": "هفت روز."}],
                                      [{"id": "refund#2", "text": "استثنای ویدئوی معیوب."}])
            joined = "\n".join(message["content"] for message in captured["messages"])
            self.assertIn("هفت روز.", joined)
            self.assertIn("[refund#2]", joined)

    def test_citation_check_requires_one_selected_source(self):
        chunks = [{"id": "refund#2"}]
        self.assertTrue(runtime.has_retrieved_citation("طبق [refund#2]، بله.", chunks))
        self.assertFalse(runtime.has_retrieved_citation("طبق قانون، بله.", chunks))


if __name__ == "__main__":
    unittest.main()
