import unittest
from unittest.mock import MagicMock, patch

from llm_interface.llm_config import supports_structured_output
from llm_interface.openai import OpenAIWrapper


def fake_response(reasoning_tokens=7, cached_tokens=3):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "hello"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response.usage.prompt_tokens_details.cached_tokens = cached_tokens
    response.usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
    return response


class TestReasoningEffort(unittest.TestCase):
    def setUp(self):
        with patch("llm_interface.openai.OpenAI"):
            self.wrapper = OpenAIWrapper(api_key="k", reasoning_effort="none")
        self.wrapper.client = MagicMock()
        self.wrapper.client.chat.completions.create.return_value = fake_response()

    def test_effort_is_forwarded_and_reasoning_tokens_reported(self):
        response = self.wrapper.chat(
            [{"role": "user", "content": "hi"}], model="gpt-5.6-luna"
        )
        kwargs = self.wrapper.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(response["usage"]["reasoning_tokens"], 7)
        self.assertEqual(response["usage"]["cached_tokens"], 3)

    def test_no_effort_by_default(self):
        with patch("llm_interface.openai.OpenAI"):
            wrapper = OpenAIWrapper(api_key="k")
        wrapper.client = MagicMock()
        wrapper.client.chat.completions.create.return_value = fake_response()
        wrapper.chat([{"role": "user", "content": "hi"}], model="gpt-5")
        kwargs = wrapper.client.chat.completions.create.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)


class TestStructuredOutputVersions(unittest.TestCase):
    def test_dotted_and_named_gpt5_models(self):
        for name in ("gpt-5", "gpt-5-mini", "gpt-5.6-luna", "gpt-5.4-nano", "gpt-6"):
            self.assertTrue(supports_structured_output(name), name)

    def test_older_models_keep_their_rules(self):
        self.assertTrue(supports_structured_output("gpt-4o"))
        self.assertFalse(supports_structured_output("gpt-3.5-turbo"))


if __name__ == "__main__":
    unittest.main()
