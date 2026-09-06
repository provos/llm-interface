import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llm_interface import errors
from llm_interface.llm_interface import LLMInterface
from llm_interface.llm_tool import create_tool
from llm_interface.openai_responses import (
    PROVIDER_ITEMS,
    OpenAIResponsesWrapper,
    translate_messages_for_responses,
    translate_tools_for_responses,
)


def item(**fields):
    obj = MagicMock()
    for key, value in fields.items():
        setattr(obj, key, value)
    obj.model_dump.return_value = dict(fields)
    return obj


def fake_response(output, status="completed", reason=None, cached=3, reasoning=7):
    response = MagicMock()
    response.output = output
    response.status = status
    response.incomplete_details = MagicMock(reason=reason) if reason else None
    response.usage.input_tokens = 100
    response.usage.output_tokens = 20
    response.usage.total_tokens = 120
    response.usage.input_tokens_details.cached_tokens = cached
    response.usage.output_tokens_details.reasoning_tokens = reasoning
    return response


def text_message(text):
    return item(
        type="message", role="assistant", content=[item(type="output_text", text=text)]
    )


class TestTranslation(unittest.TestCase):
    def test_tools_are_flattened(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        flat = translate_tools_for_responses(tools)
        self.assertEqual(flat[0]["name"], "read_file")
        self.assertEqual(flat[0]["type"], "function")
        self.assertFalse(flat[0]["strict"])
        self.assertNotIn("function", flat[0])

    def test_messages_become_instructions_and_items(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "a"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "text",
            },
            {"role": "assistant", "content": "done"},
        ]
        instructions, items = translate_messages_for_responses(messages)
        self.assertEqual(instructions, "Be brief.")
        self.assertEqual(items[0], {"role": "user", "content": "hi"})
        self.assertEqual(
            items[1],
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "a"}',
            },
        )
        self.assertEqual(
            items[2],
            {"type": "function_call_output", "call_id": "call_1", "output": "text"},
        )
        self.assertEqual(items[3], {"role": "assistant", "content": "done"})

    def test_provider_items_are_replayed_instead_of_rebuilt(self):
        replay = [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "x"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "f",
                "arguments": "{}",
            },
        ]
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "f", "arguments": {}}}
                ],
                PROVIDER_ITEMS: replay,
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        _, items = translate_messages_for_responses(messages)
        self.assertEqual(items[:2], replay)
        self.assertEqual(items[2]["type"], "function_call_output")

    def test_tool_message_without_call_id_fails_fast(self):
        with self.assertRaises(ValueError):
            translate_messages_for_responses([{"role": "tool", "content": "x"}])


class TestChat(unittest.TestCase):
    def setUp(self):
        with patch("llm_interface.openai_responses.OpenAI"):
            self.wrapper = OpenAIResponsesWrapper(
                api_key="k", reasoning_effort="medium"
            )
        self.wrapper.client = MagicMock()

    def test_request_shape_and_tool_call_response(self):
        class Answer(BaseModel):
            file: str = ""
            relevant: bool

        reasoning = item(type="reasoning", id="rs_1", encrypted_content="enc")
        call = item(
            type="function_call",
            id="fc_1",
            call_id="call_1",
            name="read_file",
            arguments='{"path": "a"}',
        )
        self.wrapper.client.responses.create.return_value = fake_response(
            [reasoning, call]
        )
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
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=tools,
            model="gpt-5.6-luna",
            response_schema=Answer,
            max_tokens=500,
        )

        params = self.wrapper.client.responses.create.call_args.kwargs
        self.assertEqual(params["model"], "gpt-5.6-luna")
        self.assertEqual(params["instructions"], "sys")
        self.assertEqual(params["input"], [{"role": "user", "content": "hi"}])
        self.assertFalse(params["store"])
        self.assertEqual(params["include"], ["reasoning.encrypted_content"])
        self.assertEqual(params["reasoning"], {"effort": "medium"})
        self.assertEqual(params["max_output_tokens"], 500)
        self.assertEqual(params["tools"][0]["name"], "read_file")
        self.assertEqual(params["text"]["format"]["type"], "json_schema")
        self.assertTrue(params["text"]["format"]["strict"])
        self.assertEqual(
            sorted(params["text"]["format"]["schema"]["required"]), ["file", "relevant"]
        )

        self.assertEqual(
            result["message"]["tool_calls"],
            [{"id": "call_1", "name": "read_file", "arguments": '{"path": "a"}'}],
        )
        self.assertEqual(result["message"][PROVIDER_ITEMS][0]["type"], "reasoning")
        self.assertEqual(result["message"][PROVIDER_ITEMS][1]["call_id"], "call_1")
        self.assertEqual(result["usage"]["cached_tokens"], 3)
        self.assertEqual(result["usage"]["reasoning_tokens"], 7)
        self.assertFalse(result["done"])

    def test_final_text_answer(self):
        self.wrapper.client.responses.create.return_value = fake_response(
            [text_message('{"ok": true}')]
        )
        result = self.wrapper.chat(
            [{"role": "user", "content": "hi"}], model="gpt-5.6-luna"
        )
        self.assertEqual(result["message"]["content"], '{"ok": true}')
        self.assertNotIn("tool_calls", result["message"])
        self.assertTrue(result["done"])

    def test_effort_none_skips_reasoning_include(self):
        with patch("llm_interface.openai_responses.OpenAI"):
            wrapper = OpenAIResponsesWrapper(api_key="k", reasoning_effort="none")
        wrapper.client = MagicMock()
        wrapper.client.responses.create.return_value = fake_response(
            [text_message("x")]
        )
        wrapper.chat([{"role": "user", "content": "hi"}], model="gpt-5.6-luna")
        params = wrapper.client.responses.create.call_args.kwargs
        self.assertEqual(params["reasoning"], {"effort": "none"})
        self.assertNotIn("include", params)

    def test_length_and_refusal(self):
        self.wrapper.client.responses.create.return_value = fake_response(
            [], status="incomplete", reason="max_output_tokens"
        )
        result = self.wrapper.chat([{"role": "user", "content": "hi"}], model="m")
        self.assertEqual(result["error_type"], errors.LENGTH)

        refusal = item(
            type="message",
            role="assistant",
            content=[item(type="refusal", refusal="no")],
        )
        self.wrapper.client.responses.create.return_value = fake_response([refusal])
        result = self.wrapper.chat([{"role": "user", "content": "hi"}], model="m")
        self.assertEqual(result["refusal"], "no")


class RecordingClient:
    """A client that first asks for a tool, then answers; records what it was sent."""

    keeps_provider_items = True

    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if len(self.calls) == 1:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "call_1", "name": "ping", "arguments": "{}"}],
                    PROVIDER_ITEMS: [
                        {"type": "reasoning", "id": "rs_1"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "ping",
                            "arguments": "{}",
                        },
                    ],
                },
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "done": False,
            }
        return {
            "message": {"content": "pong!"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "done": True,
        }


class TestProviderItemsRoundTrip(unittest.TestCase):
    def test_tool_loop_keeps_provider_items_on_the_assistant_turn(self):
        def ping() -> str:
            """Ping."""
            return "pong"

        client = RecordingClient()
        llm = LLMInterface(model_name="m", client=client, use_cache=False)
        answer = llm.chat(
            [{"role": "user", "content": "go"}], tools=[create_tool(ping)]
        )

        self.assertEqual(answer, "pong!")
        second = client.calls[1]
        assistant = [m for m in second if m.get("role") == "assistant"][0]
        self.assertEqual(assistant[PROVIDER_ITEMS][0]["type"], "reasoning")
        self.assertEqual(second[-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
