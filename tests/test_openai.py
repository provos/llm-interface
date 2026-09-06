import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from openai.pagination import SyncPage
from openai.types import CompletionUsage, Model

import llm_interface.errors as errors
from llm_interface.openai import OpenAIWrapper, translate_tools_for_openai


class TestOpenAIWrapper(unittest.TestCase):
    def setUp(self):
        # Mock the OpenAI client
        self.mock_client = Mock()

        # Patch the OpenAI class to use the mock client
        patcher = patch("llm_interface.openai.OpenAI", return_value=self.mock_client)
        self.addCleanup(patcher.stop)
        self.mock_openai = patcher.start()

        self.api_key = "test_api_key"
        self.openai_wrapper = OpenAIWrapper(api_key=self.api_key)

    def test_chat_basic(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Chat response content", tool_calls=[], done=True
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=10, total_tokens=19
            ),
        )

        messages = [{"role": "user", "content": "Hello, assistant!"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(
            response,
            {
                "message": {"content": "Chat response content"},
                "usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 10,
                    "total_tokens": 19,
                },
                "done": True,
            },
        )

        self.mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=messages, max_completion_tokens=4096
        )

    def test_chat_with_length_error(self):
        self.mock_client.chat.completions.create.side_effect = LengthFinishReasonError(
            completion=MagicMock()
        )

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(
            response,
            {
                "error": "Response exceeded the maximum allowed length.",
                "error_type": errors.LENGTH,
                "content": None,
                "done": False,
                "usage": None,
            },
        )

    def test_chat_with_content_filter_error(self):
        self.mock_client.chat.completions.create.side_effect = (
            ContentFilterFinishReasonError()
        )

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(messages=messages)

        self.assertEqual(
            response,
            {
                "error": "Content was rejected by the content filter.",
                "error_type": errors.CONTENT_FILTER,
                "content": None,
                "done": False,
                "usage": None,
            },
        )

    def test_chat_with_response_schema(self):
        # Mock the beta chat completion with parsing
        mock_parsed_content = {"parsed": "data"}
        mock_message = MagicMock(parsed=mock_parsed_content, tool_calls=[], done=True)
        # Configure __contains__ to allow 'in' checks
        mock_message.__contains__.side_effect = lambda key: key in mock_message.__dict__

        mock_response = Mock(
            choices=[Mock(message=mock_message, finish_reason="stop")],
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=10, total_tokens=19
            ),
        )
        self.mock_client.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "Test message"}]
        response = self.openai_wrapper.chat(
            messages=messages, response_schema="DummySchema"
        )

        self.assertEqual(
            response,
            {
                "message": {"content": mock_parsed_content},
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 9,
                    "total_tokens": 19,
                },
                "done": True,
            },
        )

    def test_chat_with_options(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Chat response with options", tool_calls=[], done=True
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=10, total_tokens=19
            ),
        )

        messages = [{"role": "user", "content": "Test message"}]
        options = {"temperature": 0.7}
        response = self.openai_wrapper.chat(messages=messages, options=options)

        self.assertEqual(
            response,
            {
                "message": {"content": "Chat response with options"},
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 9,
                    "total_tokens": 19,
                },
                "done": True,
            },
        )

    def test_chat_exception_propagation(self):
        self.mock_client.chat.completions.create.side_effect = Exception(
            "Test exception"
        )

        with self.assertRaises(Exception) as context:
            self.openai_wrapper.chat(
                messages=[{"role": "user", "content": "Test message"}]
            )

        self.assertEqual(str(context.exception), "Test exception")

    def test_list_models(self):
        # Create mock model data
        models_data = [
            Model(id="gpt-4", created=1687882411, object="model", owned_by="openai"),
            Model(
                id="gpt-3.5-turbo",
                created=1677610602,
                object="model",
                owned_by="openai",
            ),
        ]
        mock_response = SyncPage[Model](data=models_data, object="list")

        # Set up the mock return value
        self.mock_client.models.list.return_value = mock_response

        # Call the list method
        response = self.openai_wrapper.list()

        print(response)

        # Verify the response format matches Ollama's format
        self.assertIn("models", response)
        self.assertEqual(len(response["models"]), 2)

        # Verify first model data
        model = response["models"][0]
        self.assertEqual(model["model"], "gpt-4")
        self.assertEqual(
            model["modified_at"], datetime.fromtimestamp(1687882411, tz=timezone.utc)
        )
        self.assertEqual(model["digest"], "unknown")
        self.assertEqual(model["size"], 0)

        # Verify model details
        self.assertEqual(model["details"]["family"], "openai")
        self.assertEqual(model["details"]["families"], ["openai"])
        self.assertEqual(model["details"]["parameter_size"], "unknown")
        self.assertEqual(model["details"]["quantization_level"], "unknown")

    def test_chat_tool_choice_forwarded(self):
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(content="ok", tool_calls=[], done=True),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        )

        self.openai_wrapper.chat(
            messages=[{"role": "user", "content": "Hi"}], tool_choice="none"
        )

        call_kwargs = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["tool_choice"], "none")

    def test_chat_with_response_schema_uses_non_beta_parse(self):
        mock_parsed_content = {"parsed": "data"}
        mock_message = MagicMock(parsed=mock_parsed_content, tool_calls=[], done=True)
        mock_message.__contains__.side_effect = lambda key: key in mock_message.__dict__

        mock_response = Mock(
            choices=[Mock(message=mock_message, finish_reason="stop")],
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=10, total_tokens=19
            ),
        )
        self.mock_client.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "Test message"}]
        self.openai_wrapper.chat(messages=messages, response_schema="DummySchema")

        self.mock_client.chat.completions.parse.assert_called_once()
        # The deprecated beta namespace must not be used.
        self.mock_client.beta.chat.completions.parse.assert_not_called()


class TestTranslateToolsForOpenAI(unittest.TestCase):
    """Tests for translate_tools_for_openai's message normalization."""

    def test_does_not_mutate_caller_messages(self):
        original_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {"a": 1}},
                    }
                ],
            }
        ]
        snapshot = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {"a": 1}},
                    }
                ],
            }
        ]

        translated = translate_tools_for_openai(original_messages)

        # The caller's messages (and nested dicts) must be untouched.
        self.assertEqual(original_messages, snapshot)
        self.assertIsInstance(
            original_messages[0]["tool_calls"][0]["function"]["arguments"], dict
        )
        # The translated copy has the arguments stringified.
        self.assertIsInstance(
            translated[0]["tool_calls"][0]["function"]["arguments"], str
        )

    def test_stringifies_arguments_for_every_tool_call_in_one_message(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {"a": 1}},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "g", "arguments": {"b": 2}},
                    },
                ],
            }
        ]
        translated = translate_tools_for_openai(messages)
        self.assertEqual(len(translated[0]["tool_calls"]), 2)
        for tool_call in translated[0]["tool_calls"]:
            self.assertIsInstance(tool_call["function"]["arguments"], str)

    def test_strips_unsupported_tool_message_keys(self):
        messages = [
            {
                "role": "tool",
                "name": "f",
                "tool_call_id": "call_1",
                "content": "result",
                "is_error": True,
            }
        ]
        translated = translate_tools_for_openai(messages)
        self.assertEqual(len(translated), 1)
        self.assertNotIn("name", translated[0])
        self.assertNotIn("is_error", translated[0])
        self.assertEqual(translated[0]["role"], "tool")
        self.assertEqual(translated[0]["tool_call_id"], "call_1")
        self.assertEqual(translated[0]["content"], "result")

    def test_plain_messages_pass_through(self):
        messages = [{"role": "user", "content": "Hello"}]
        translated = translate_tools_for_openai(messages)
        self.assertEqual(translated, messages)


if __name__ == "__main__":
    unittest.main()


class TestToolMessageValidation(unittest.TestCase):
    def test_missing_tool_call_id_fails_fast(self):
        from llm_interface.openai import translate_tools_for_openai

        with self.assertRaises(ValueError):
            translate_tools_for_openai([{"role": "tool", "content": "x"}])
