import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import RateLimitError

from llm_interface import errors
from llm_interface.llm_interface import LLMInterface
from llm_interface.openai_responses import OpenAIResponsesWrapper


def rate_limit_error():
    return RateLimitError(
        "Rate limit reached for gpt-5.6-luna on tokens per min (TPM)",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
        body=None,
    )


class TestResponsesWrapperRateLimit(unittest.TestCase):
    def test_429_becomes_a_retryable_error_dict(self):
        with patch("llm_interface.openai_responses.OpenAI"):
            wrapper = OpenAIResponsesWrapper(api_key="k")
        wrapper.client = MagicMock()
        wrapper.client.responses.create.side_effect = rate_limit_error()
        result = wrapper.chat([{"role": "user", "content": "hi"}], model="m")
        self.assertEqual(result["error_type"], errors.RATE_LIMIT)


class FlakyClient:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            return {
                "error": "Rate limited",
                "error_type": errors.RATE_LIMIT,
                "content": None,
                "done": False,
                "usage": None,
            }
        return {
            "message": {"content": "ok"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "done": True,
        }


class TestInterfaceRetriesRateLimits(unittest.TestCase):
    def test_retries_with_a_growing_delay_then_succeeds(self):
        client = FlakyClient(failures=2)
        llm = LLMInterface(
            model_name="m",
            client=client,
            use_cache=False,
            max_retries=3,
            retry_delay=1.0,
        )
        with patch("llm_interface.llm_interface.time.sleep") as sleep:
            answer = llm.chat([{"role": "user", "content": "go"}])
        self.assertEqual(answer, "ok")
        self.assertEqual(client.calls, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [15.0, 30.0])

    def test_gives_up_after_max_retries(self):
        client = FlakyClient(failures=10)
        llm = LLMInterface(
            model_name="m",
            client=client,
            use_cache=False,
            max_retries=2,
            retry_delay=1.0,
        )
        with patch("llm_interface.llm_interface.time.sleep"):
            with self.assertRaises(Exception):
                llm.chat([{"role": "user", "content": "go"}])
        self.assertEqual(client.calls, 3)


if __name__ == "__main__":
    unittest.main()
