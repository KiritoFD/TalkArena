import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from model_loader import LLMLoader


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ]


class _FakeOpenAI:
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.models = SimpleNamespace(list=self._list_models)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def _list_models(self):
        if self.api_key == "bad-key":
            raise RuntimeError("bad endpoint")
        return SimpleNamespace(data=[SimpleNamespace(id="gpt-5-mini")])

    def _create_completion(self, **kwargs):
        if self.api_key == "bad-key":
            raise RuntimeError("bad endpoint")
        return _FakeResponse("ok from backup")


class _FakeLocalFallback:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    def generate(self, text: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        return "ok from local"


class ModelLoaderFallbackTests(unittest.TestCase):
    @patch("openai.OpenAI", _FakeOpenAI)
    def test_generate_falls_back_to_backup_endpoint(self):
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEYS": "bad-key,good-key",
                "LLM_MODELS": "gpt-5-mini,gpt-5-nano",
                "LLM_PROVIDERS": "openai,openai",
                "LLM_MAX_RETRIES": "0",
            },
            clear=False,
        ):
            loader = LLMLoader()
            text = loader.generate("hello", max_new_tokens=8, temperature=0)

        self.assertEqual(text, "ok from backup")
        self.assertEqual(loader.active_endpoint, 1)
        self.assertEqual(loader.model_name, "gpt-5-nano")

    @patch("openai.OpenAI", _FakeOpenAI)
    def test_list_models_falls_back(self):
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEYS": "bad-key,good-key",
                "LLM_MODELS": "gpt-5-mini,gpt-5-nano",
                "LLM_MAX_RETRIES": "0",
            },
            clear=False,
        ):
            loader = LLMLoader()
            models = loader.list_models()

        self.assertIn("gpt-5-mini", models)
        self.assertEqual(loader.active_endpoint, 1)

    def test_generate_falls_back_to_local_modelscope_model(self):
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEYS": "bad-key",
                "LLM_MODELS": "gpt-5-mini",
                "LLM_MAX_RETRIES": "0",
                "LLM_ENABLE_LOCAL_FALLBACK": "1",
            },
            clear=False,
        ), patch("openai.OpenAI", _FakeOpenAI), patch.object(
            LLMLoader, "_get_local_fallback", return_value=_FakeLocalFallback()
        ):
            loader = LLMLoader()
            text = loader.generate("hello", max_new_tokens=8, temperature=0)

        self.assertEqual(text, "ok from local")
        self.assertEqual(loader.provider, "local")
        self.assertEqual(loader.model_name, "Qwen/Qwen2.5-0.5B-Instruct")


if __name__ == "__main__":
    unittest.main()
