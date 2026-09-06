import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from anthropic import APIConnectionError, APIError, APITimeoutError
from pydantic import BaseModel

import llm_interface.errors as errors
from llm_interface.anthropic import (
    AnthropicWrapper,
    translate_messages_for_anthropic,
    translate_tools_for_anthropic,
)


def _stub_stream(mock_client, response):
    """Make ``client.messages.stream(...)`` a context manager yielding ``response``."""
    manager = MagicMock()
    manager.__enter__.return_value.get_final_message.return_value = response
    manager.__exit__.return_value = False
    mock_client.messages.stream.return_value = manager
    return manager


class TestAnthropicWrapper(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.mock_client = MagicMock()
        self.anthropic_wrapper = AnthropicWrapper(api_key=self.api_key)
        self.anthropic_wrapper.client = self.mock_client

    @patch("requests.get")
    def test_list_models(self, mock_get):
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "type": "model",
                    "id": "claude-3-opus-20240229",
                    "display_name": "Claude 3 Opus",
                    "created_at": "2024-02-29T00:00:00Z",
                },
                {
                    "type": "model",
                    "id": "claude-3-sonnet-20240229",
                    "display_name": "Claude 3 Sonnet",
                    "created_at": "2024-02-29T00:00:00Z",
                },
            ],
            "has_more": False,
            "first_id": "model_1",
            "last_id": "model_2",
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Call the list method
        response = self.anthropic_wrapper.list()

        # Verify API call
        mock_get.assert_called_once_with(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )

        # Verify response format
        self.assertEqual(len(response.models), 2)

        # Check first model
        model = response.models[0]
        self.assertEqual(model.model, "claude-3-opus-20240229")
        self.assertEqual(
            model.modified_at, datetime(2024, 2, 29, 0, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(model.digest, "unknown")
        self.assertEqual(model.size, 0)
        self.assertEqual(model.details.family, "claude")
        self.assertEqual(model.details.families, ["claude"])

    def test_chat_basic(self):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello, I'm Claude!")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        _stub_stream(self.mock_client, mock_response)

        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
        ]

        # Call chat method
        response = self.anthropic_wrapper.chat(
            messages, model="claude-3-sonnet-20240229"
        )

        # Verify correct parameters passed to Anthropic
        self.mock_client.messages.stream.assert_called_once()
        call_args = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_args["system"], "You are a helpful assistant.")
        self.assertEqual(call_args["model"], "claude-3-sonnet-20240229")
        self.assertEqual(len(call_args["messages"]), 1)

        # Verify response format
        self.assertEqual(response["message"]["content"], "Hello, I'm Claude!")
        self.assertEqual(response["usage"]["prompt_tokens"], 10)
        self.assertEqual(response["usage"]["completion_tokens"], 5)
        self.assertEqual(response["usage"]["total_tokens"], 15)
        self.assertEqual(response["usage"]["cache_creation_tokens"], 0)
        self.assertEqual(response["usage"]["reasoning_tokens"], 0)
        self.assertTrue(response["done"])

    def test_chat_reports_cache_writes_and_estimates_thinking(self):
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="thinking", thinking="(summarized)"),
            MagicMock(type="text", text="x" * 400),
        ]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 1000
        mock_response.usage.cache_read_input_tokens = 5000
        mock_response.usage.cache_creation_input_tokens = 700

        _stub_stream(self.mock_client, mock_response)

        response = self.anthropic_wrapper.chat(
            [{"role": "user", "content": "hi"}], model="claude-sonnet-5"
        )

        self.assertEqual(response["usage"]["cached_tokens"], 5000)
        self.assertEqual(response["usage"]["cache_creation_tokens"], 700)
        # 1000 output tokens, 400 chars (~100 tokens) visible: ~900 thinking
        self.assertEqual(response["usage"]["reasoning_tokens"], 900)

    def test_chat_with_tools(self):
        # Set up mock response with tool use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "search_weather"
        tool_block.input = '{"location": "San Francisco"}'

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        _stub_stream(self.mock_client, mock_response)

        # Define tools and messages
        tools = [
            {
                "function": {
                    "name": "search_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                }
            }
        ]

        messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

        # Call chat method
        response = self.anthropic_wrapper.chat(messages, tools=tools)

        # Verify correct parameters passed to Anthropic
        self.mock_client.messages.stream.assert_called_once()
        call_args = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(len(call_args["tools"]), 1)
        self.assertEqual(call_args["tools"][0]["name"], "search_weather")

        # Verify tool call in response
        self.assertIn("tool_calls", response["message"])
        tool_call = response["message"]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "tool_123")
        self.assertEqual(tool_call["name"], "search_weather")
        self.assertEqual(tool_call["arguments"], '{"location": "San Francisco"}')

    def test_chat_with_tool_response(self):
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="The weather in San Francisco is sunny.")
        ]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 8
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        _stub_stream(self.mock_client, mock_response)

        # Create messages with tool response
        messages = [
            {"role": "user", "content": "What's the weather in San Francisco?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_123",
                        "function": {
                            "name": "search_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool_123",
                "content": '{"temp": 72, "condition": "sunny"}',
            },
        ]

        response = self.anthropic_wrapper.chat(messages)

        # Verify translated messages were passed correctly
        call_args = self.mock_client.messages.stream.call_args[1]
        translated_messages = call_args["messages"]
        self.assertEqual(len(translated_messages), 3)

        # Verify response
        self.assertEqual(
            response["message"]["content"], "The weather in San Francisco is sunny."
        )
        self.assertEqual(response["usage"]["total_tokens"], 28)

    @patch("llm_interface.anthropic.encode_image_to_base64")
    def test_chat_with_images(self, mock_encode_image):
        # Setup mocks
        mock_encode_image.return_value = "base64encodedimage"

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="I see a cat in the image.")
        ]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 7
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        _stub_stream(self.mock_client, mock_response)

        # Create messages with images
        messages = [
            {
                "role": "user",
                "content": "What's in this image?",
                "images": ["path/to/image.jpg"],
            }
        ]

        response = self.anthropic_wrapper.chat(messages)

        # Verify image translation
        call_args = self.mock_client.messages.stream.call_args[1]
        translated_messages = call_args["messages"]
        self.assertEqual(len(translated_messages), 1)
        self.assertEqual(len(translated_messages[0]["content"]), 2)  # Text + image
        self.assertEqual(translated_messages[0]["content"][1]["type"], "image")

        # Verify response
        self.assertEqual(response["message"]["content"], "I see a cat in the image.")

    def test_chat_error_handling(self):
        # Test timeout error
        self.mock_client.messages.stream.side_effect = APITimeoutError(request=Mock())
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "timeout")

        # Test connection error
        self.mock_client.messages.stream.side_effect = APIConnectionError(
            request=Mock(), message="Connection error"
        )
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "connection")

        # Test API error
        self.mock_client.messages.stream.side_effect = APIError(
            request=Mock(), message="API error", body=None
        )
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "provider_specific")


class TestTranslateMessagesForAnthropic(unittest.TestCase):
    """Tests for translate_messages_for_anthropic covering multi tool-call
    turns, merged tool_result groups, and the no-op case."""

    def test_translate_multiple_tool_calls_single_message(self):
        messages = [
            {"role": "user", "content": "What's the weather in two cities?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "SF"},
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "NYC"},
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny in SF"},
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "Error: tool failed",
                "is_error": True,
            },
        ]

        translated = translate_messages_for_anthropic(messages)

        # user, assistant (2 tool_use blocks), ONE merged tool_result message
        self.assertEqual(len(translated), 3)

        assistant_msg = translated[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        tool_use_blocks = [
            b for b in assistant_msg["content"] if b["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_use_blocks), 2)
        # The fabricated <thinking> text block must be gone entirely.
        self.assertFalse(any(b["type"] == "text" for b in assistant_msg["content"]))
        self.assertEqual(tool_use_blocks[0]["input"], {"location": "SF"})
        self.assertEqual(tool_use_blocks[1]["input"], {"location": "NYC"})

        tool_result_msg = translated[2]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertEqual(len(tool_result_msg["content"]), 2)
        self.assertEqual(tool_result_msg["content"][0]["tool_use_id"], "call_1")
        self.assertNotIn("is_error", tool_result_msg["content"][0])
        self.assertEqual(tool_result_msg["content"][1]["tool_use_id"], "call_2")
        self.assertTrue(tool_result_msg["content"][1]["is_error"])

    def test_translate_string_tool_arguments_parsed_to_dict(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                    }
                ],
            }
        ]
        translated = translate_messages_for_anthropic(messages)
        tool_use_block = translated[0]["content"][0]
        self.assertIsInstance(tool_use_block["input"], dict)
        self.assertEqual(tool_use_block["input"], {"location": "SF"})

    def test_translate_noop_for_plain_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        translated = translate_messages_for_anthropic(messages)
        self.assertEqual(translated, messages)

    def test_translate_non_consecutive_tool_messages_not_merged(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "first"},
            {"role": "user", "content": "in between"},
            {"role": "tool", "tool_call_id": "call_2", "content": "second"},
        ]
        translated = translate_messages_for_anthropic(messages)
        # Two separate tool_result groups since they aren't consecutive.
        self.assertEqual(len(translated), 3)
        self.assertEqual(len(translated[0]["content"]), 1)
        self.assertEqual(translated[1]["role"], "user")
        self.assertEqual(len(translated[2]["content"]), 1)


class TestTranslateToolsForAnthropic(unittest.TestCase):
    def test_strict_tool_gets_additional_properties_false(self):
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "desc",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    "strict": True,
                }
            }
        ]
        translated = translate_tools_for_anthropic(tools)
        self.assertTrue(translated[0]["strict"])
        self.assertFalse(translated[0]["input_schema"]["additionalProperties"])

    def test_non_strict_tool_has_no_strict_keys(self):
        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "desc",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            }
        ]
        translated = translate_tools_for_anthropic(tools)
        self.assertNotIn("strict", translated[0])
        self.assertNotIn("additionalProperties", translated[0]["input_schema"])


class TestAnthropicWrapperStructuredOutputsAndOptions(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.mock_client = MagicMock()
        self.wrapper = AnthropicWrapper(api_key=self.api_key)
        self.wrapper.client = self.mock_client

    @staticmethod
    def _make_text_response(text="ok", stop_reason="end_turn", cache_read=0):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text=text)]
        mock_response.stop_reason = stop_reason
        mock_response.stop_details = None
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1
        mock_response.usage.cache_read_input_tokens = cache_read
        return mock_response

    def test_structured_call_streams_with_output_format_and_cache_control(self):
        class Person(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text='{"name": "Alice"}')]
        mock_response.stop_reason = "end_turn"
        mock_response.stop_details = None
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.parsed_output = Person(name="Alice")

        _stub_stream(self.mock_client, mock_response)

        response = self.wrapper.chat(
            messages=[{"role": "user", "content": "Who is it?"}],
            response_schema=Person,
        )

        self.mock_client.messages.stream.assert_called_once()
        self.mock_client.messages.create.assert_not_called()
        self.mock_client.messages.parse.assert_not_called()
        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["output_format"], Person)
        # structured calls keep prompt caching: stream() accepts both
        self.assertEqual(call_kwargs["cache_control"], {"type": "ephemeral"})

        self.assertEqual(response["message"]["content"], Person(name="Alice"))
        self.assertTrue(response["done"])

    def test_stream_called_with_cache_control_by_default(self):
        _stub_stream(self.mock_client, self._make_text_response())

        self.wrapper.chat(messages=[{"role": "user", "content": "Hi"}])

        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["cache_control"], {"type": "ephemeral"})

    def test_prompt_caching_disabled(self):
        wrapper = AnthropicWrapper(api_key=self.api_key, prompt_caching=False)
        wrapper.client = self.mock_client
        _stub_stream(self.mock_client, self._make_text_response())

        wrapper.chat(messages=[{"role": "user", "content": "Hi"}])

        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertNotIn("cache_control", call_kwargs)

    def test_refusal_stop_reason(self):
        mock_response = self._make_text_response(stop_reason="refusal")
        mock_response.stop_details = MagicMock(explanation="policy violation")
        _stub_stream(self.mock_client, mock_response)

        response = self.wrapper.chat(messages=[{"role": "user", "content": "Hi"}])

        self.assertEqual(response["refusal"], "policy violation")
        self.assertIsNone(response["content"])
        self.assertFalse(response["done"])

    def test_refusal_without_explanation_defaults_to_refused(self):
        mock_response = self._make_text_response(stop_reason="refusal")
        mock_response.stop_details = None
        _stub_stream(self.mock_client, mock_response)

        response = self.wrapper.chat(messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(response["refusal"], "refused")

    def test_max_tokens_stop_reason(self):
        mock_response = self._make_text_response(stop_reason="max_tokens")
        _stub_stream(self.mock_client, mock_response)

        response = self.wrapper.chat(messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(response["error_type"], errors.LENGTH)
        self.assertIsNone(response["content"])
        self.assertFalse(response["done"])

    def test_tool_choice_none_mapping(self):
        _stub_stream(self.mock_client, self._make_text_response())

        self.wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}], tool_choice="none"
        )
        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["tool_choice"], {"type": "none"})

    def test_tool_choice_auto_mapping(self):
        _stub_stream(self.mock_client, self._make_text_response())

        self.wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}], tool_choice="auto"
        )
        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["tool_choice"], {"type": "auto"})

    def test_tool_choice_dict_passthrough(self):
        _stub_stream(self.mock_client, self._make_text_response())

        explicit_choice = {"type": "tool", "name": "get_weather"}
        self.wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice=explicit_choice,
        )
        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["tool_choice"], explicit_choice)

    def test_thinking_and_effort_forwarded_without_temperature(self):
        wrapper = AnthropicWrapper(
            api_key=self.api_key, thinking={"type": "adaptive"}, effort="high"
        )
        wrapper.client = self.mock_client
        _stub_stream(self.mock_client, self._make_text_response())

        wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}],
            options={"temperature": 0.9},
        )

        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["thinking"], {"type": "adaptive"})
        self.assertEqual(call_kwargs["output_config"], {"effort": "high"})
        # Current models reject temperature when adaptive thinking is on.
        self.assertNotIn("temperature", call_kwargs)

    def test_temperature_sent_when_thinking_not_configured(self):
        _stub_stream(self.mock_client, self._make_text_response())

        self.wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}],
            options={"temperature": 0.5},
        )
        call_kwargs = self.mock_client.messages.stream.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)

    def test_cache_read_input_tokens_none_guarded(self):
        mock_response = self._make_text_response(cache_read=None)
        _stub_stream(self.mock_client, mock_response)

        response = self.wrapper.chat(messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(response["usage"]["cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()


class TestAnthropicStreamParseFailure(unittest.TestCase):
    def test_validation_error_during_stream_becomes_model_error(self):
        from pydantic import ValidationError

        wrapper = AnthropicWrapper(api_key="k")
        wrapper.client = MagicMock()
        manager = MagicMock()
        manager.__enter__.return_value.get_final_message.side_effect = (
            ValidationError.from_exception_data("X", [])
        )
        manager.__exit__.return_value = False
        wrapper.client.messages.stream.return_value = manager

        class Person(BaseModel):
            name: str

        response = wrapper.chat(
            messages=[{"role": "user", "content": "hi"}], response_schema=Person
        )
        self.assertIn("error", response)
        self.assertIn("could not be parsed", response["error"])
        self.assertIsNone(response["content"])


class TestUnparseableToolArguments(unittest.TestCase):
    def test_raw_string_arguments_do_not_crash_translation(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "Error: bad args"},
        ]
        translated = translate_messages_for_anthropic(messages)
        tool_use = translated[0]["content"][-1]
        self.assertEqual(tool_use["type"], "tool_use")
        self.assertEqual(tool_use["input"], {"raw_arguments": "{not json"})
