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


class TestEffortRejectedWithTools(unittest.TestCase):
    def test_retries_without_effort_and_remembers(self):
        import httpx
        from openai import BadRequestError

        with patch("llm_interface.openai.OpenAI"):
            wrapper = OpenAIWrapper(api_key="k", reasoning_effort="low")
        wrapper.client = MagicMock()
        rejection = BadRequestError(
            "Function tools with reasoning_effort are not supported for gpt-5.6-luna",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body=None,
        )
        wrapper.client.chat.completions.create.side_effect = [
            rejection,
            fake_response(),
        ]
        tools = [{"type": "function", "function": {"name": "ping", "parameters": {}}}]

        wrapper.chat(
            [{"role": "user", "content": "hi"}], tools=tools, model="gpt-5.6-luna"
        )

        calls = wrapper.client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", calls[1].kwargs)

        # the next tool request skips the parameter up front
        wrapper.client.chat.completions.create.side_effect = [fake_response()]
        wrapper.chat(
            [{"role": "user", "content": "hi"}], tools=tools, model="gpt-5.6-luna"
        )
        self.assertNotIn(
            "reasoning_effort", wrapper.client.chat.completions.create.call_args.kwargs
        )

        # requests without tools keep it
        wrapper.client.chat.completions.create.side_effect = [fake_response()]
        wrapper.chat([{"role": "user", "content": "hi"}], model="gpt-5.6-luna")
        self.assertEqual(
            wrapper.client.chat.completions.create.call_args.kwargs["reasoning_effort"],
            "low",
        )


class TestStructuredOutputWithNonStrictTools(unittest.TestCase):
    def setUp(self):
        with patch("llm_interface.openai.OpenAI"):
            self.wrapper = OpenAIWrapper(api_key="k")
        self.wrapper.client = MagicMock()

    def test_non_strict_tools_use_create_with_a_strict_json_schema(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            file: str = ""
            relevant: bool

        response = fake_response()
        response.choices[0].message.content = '{"file": "", "relevant": true}'
        response.choices[0].message.refusal = None
        self.wrapper.client.chat.completions.create.return_value = response
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = self.wrapper.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            model="gpt-5.6-luna",
            response_schema=Answer,
        )

        self.wrapper.client.chat.completions.parse.assert_not_called()
        kwargs = self.wrapper.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])
        schema = kwargs["response_format"]["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(sorted(schema["required"]), ["file", "relevant"])
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(result["message"]["content"], '{"file": "", "relevant": true}')

    def test_strict_tools_keep_using_parse(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            relevant: bool

        parsed = MagicMock()
        parsed.choices = [MagicMock()]
        parsed.choices[0].message.parsed = Answer(relevant=True)
        parsed.choices[0].message.tool_calls = None
        parsed.choices[0].message.__contains__ = lambda self, key: False
        parsed.choices[0].finish_reason = "stop"
        parsed.usage = fake_response().usage
        self.wrapper.client.chat.completions.parse.return_value = parsed
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "strict": True,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = self.wrapper.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            model="gpt-5.6-luna",
            response_schema=Answer,
        )

        self.wrapper.client.chat.completions.create.assert_not_called()
        self.assertEqual(result["message"]["content"], Answer(relevant=True))


class TestStructuredOutputVersions(unittest.TestCase):
    def test_dotted_and_named_gpt5_models(self):
        for name in ("gpt-5", "gpt-5-mini", "gpt-5.6-luna", "gpt-5.4-nano", "gpt-6"):
            self.assertTrue(supports_structured_output(name), name)

    def test_older_models_keep_their_rules(self):
        self.assertTrue(supports_structured_output("gpt-4o"))
        self.assertFalse(supports_structured_output("gpt-3.5-turbo"))


if __name__ == "__main__":
    unittest.main()
