# Copyright 2025 Niels Provos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llm_interface.gemini import (
    GeminiWrapper,
    convert_gemini_models_to_ollama_response,
    translate_messages_for_gemini,
    translate_tools_for_gemini,
)


class MockGenaiModel:
    def __init__(self, name):
        self.name = name


class MockGenaiPart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class MockGenaiFunctionCall:
    def __init__(self, name, args, id="tool_123"):
        self.name = name
        self.args = args
        self.id = id


class MockGenaiContent:
    def __init__(self, parts):
        self.parts = parts


class MockGenaiCandidate:
    def __init__(self, content):
        self.content = content


class MockGenaiResponse:
    def __init__(self, text=None, candidates=None, usage_metadata=None):
        self.text = text
        self.candidates = candidates
        self.usage_metadata = usage_metadata


class MockGenaiUsageMetadata:
    def __init__(self, prompt_token_count, candidates_token_count, total_token_count):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class TestGeminiUtils(unittest.TestCase):
    def test_translate_tools_for_gemini(self):
        ollama_tools = [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            }
        ]
        expected_gemini_tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ]
        self.assertEqual(
            translate_tools_for_gemini(ollama_tools), expected_gemini_tools
        )

    @patch("llm_interface.gemini.Path")
    @patch("llm_interface.gemini.types.Part")
    def test_translate_messages_for_gemini(self, mock_part, mock_path):
        # Mock image handling
        mock_image_path_instance = MagicMock()
        mock_image_path_instance.read_bytes.return_value = b"imagedata"
        mock_path.return_value = mock_image_path_instance
        mock_part.from_bytes.return_value = "mock_image_part"

        ollama_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "weather"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search",
                "content": '{"result": "sunny"}',
            },
            {
                "role": "user",
                "content": "What's in this image?",
                "images": ["image.jpg"],
            },
        ]

        expected_gemini_messages = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "weather"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "search",
                            "response": {
                                "name": "search",
                                "content": {"result": "sunny"},
                            },
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [{"text": "What's in this image?"}, "mock_image_part"],
            },
        ]

        translated_messages = translate_messages_for_gemini(ollama_messages)
        self.assertEqual(translated_messages, expected_gemini_messages)
        mock_part.from_bytes.assert_called_once_with(
            data=b"imagedata", mime_type="image/jpeg"
        )

    @patch("llm_interface.gemini.datetime")
    def test_convert_gemini_models_to_ollama_response(self, mock_datetime):
        mock_now = datetime(2025, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        gemini_models = [
            MockGenaiModel(name="models/gemini-1.5-flash"),
            MockGenaiModel(name="models/gemini-pro"),
        ]
        response = convert_gemini_models_to_ollama_response(gemini_models)

        self.assertEqual(len(response.models), 2)
        self.assertEqual(response.models[0].model, "models/gemini-1.5-flash")
        self.assertEqual(response.models[0].modified_at, mock_now)
        self.assertEqual(response.models[0].details.family, "gemini")
        self.assertEqual(response.models[1].model, "models/gemini-pro")


class TestGeminiWrapper(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.mock_genai_client = MagicMock()
        # Mock the client factory used by GeminiWrapper
        with patch(
            "llm_interface.gemini.genai.Client", return_value=self.mock_genai_client
        ):
            self.gemini_wrapper = GeminiWrapper(api_key=self.api_key)

    @patch("llm_interface.gemini.genai.Client")
    @patch("llm_interface.gemini.types.HttpOptions")
    def test_timeout_conversion_seconds_to_milliseconds(
        self, mock_http_options, mock_genai_client
    ):
        """Test that timeout is correctly converted from seconds to milliseconds."""
        mock_http_options_instance = MagicMock()
        mock_http_options.return_value = mock_http_options_instance
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        # Test with default timeout (600.0 seconds)
        GeminiWrapper(api_key="test_key")

        # Verify HttpOptions was called with timeout in milliseconds
        mock_http_options.assert_called_once_with(timeout=600000)  # 600.0 * 1000

        # Verify genai.Client was called with the HttpOptions
        mock_genai_client.assert_called_once_with(
            api_key="test_key", http_options=mock_http_options_instance
        )

        # Reset mocks for second test
        mock_http_options.reset_mock()
        mock_genai_client.reset_mock()

        # Test with custom timeout (30.5 seconds)
        GeminiWrapper(api_key="test_key", timeout=30.5)

        # Verify HttpOptions was called with timeout in milliseconds
        mock_http_options.assert_called_with(timeout=30500)  # 30.5 * 1000

        # Verify genai.Client was called with the HttpOptions
        mock_genai_client.assert_called_with(
            api_key="test_key", http_options=mock_http_options.return_value
        )

    def test_http_options_timeout_type_assumption(self):
        """
        Test that validates our assumption about HttpOptions timeout parameter type.

        This test checks if HttpOptions expects timeout in milliseconds (int) or seconds (float).
        If the Gemini API changes to expect seconds instead of milliseconds, this test will
        help us detect that change and update our conversion logic accordingly.
        """
        from google.genai import types

        # Get the HttpOptions model fields (it's a Pydantic model)
        model_fields = types.HttpOptions.model_fields
        timeout_field = model_fields.get("timeout")

        # Verify that timeout field exists
        self.assertIsNotNone(timeout_field, "HttpOptions should have a timeout field")

        # Check the type annotation
        timeout_annotation = timeout_field.annotation

        # Current expectation: timeout should be Optional[int] (milliseconds)
        # If this changes to Optional[float], it likely means they switched to seconds

        # Extract the actual types from the annotation
        if hasattr(timeout_annotation, "__args__"):
            # Handle Union/Optional types (e.g., Optional[int] = Union[int, None])
            actual_types = list(timeout_annotation.__args__)
        else:
            # Handle simple types
            actual_types = [timeout_annotation]

        # Check if our current assumption (int for milliseconds) is still valid
        has_int = int in actual_types
        has_float = float in actual_types
        has_none = type(None) in actual_types

        # Verify it's Optional (has None type)
        self.assertTrue(
            has_none, f"Expected timeout to be Optional, but got: {timeout_annotation}"
        )

        if has_float and not has_int:
            self.fail(
                "HttpOptions timeout field now expects float (likely seconds). "
                "Update GeminiWrapper to not multiply by 1000. "
                f"Current annotation: {timeout_annotation}"
            )
        elif has_int and not has_float:
            # This is our current expectation - timeout in milliseconds as int
            pass
        else:
            self.fail(
                f"Unexpected timeout field type annotation: {timeout_annotation}. "
                "Please review the HttpOptions API documentation and update the test."
            )

        # Test that float values are rejected (confirming it expects int milliseconds)
        with self.assertRaises((TypeError, ValueError)):
            types.HttpOptions(timeout=30.5)  # Should fail if it expects int

    @patch("llm_interface.gemini.genai.Client")
    def test_dynamic_timeout_detection(self, mock_genai_client):
        """Test that the dynamic timeout detection logic works correctly."""
        # Mock the client to avoid actual API calls
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance

        # Create a wrapper instance to test the dynamic detection
        wrapper = GeminiWrapper(api_key="test_key")

        # Test the _get_appropriate_timeout_value method directly
        # With current API (expects int milliseconds), 30 seconds should become 30000 ms
        result = wrapper._get_appropriate_timeout_value(30.0)

        # The result should be an integer (milliseconds) with current API
        self.assertIsInstance(
            result, int, "Expected integer result for current API (milliseconds)"
        )
        self.assertEqual(
            result, 30000, "30 seconds should convert to 30000 milliseconds"
        )

        # Test with fractional seconds
        result_fractional = wrapper._get_appropriate_timeout_value(30.5)
        self.assertIsInstance(
            result_fractional, int, "Expected integer result for fractional seconds"
        )
        self.assertEqual(
            result_fractional,
            30500,
            "30.5 seconds should convert to 30500 milliseconds",
        )

        # Test edge cases
        result_zero = wrapper._get_appropriate_timeout_value(0.0)
        self.assertEqual(result_zero, 0, "0 seconds should convert to 0 milliseconds")

        result_large = wrapper._get_appropriate_timeout_value(600.0)
        self.assertEqual(
            result_large, 600000, "600 seconds should convert to 600000 milliseconds"
        )

    @patch("llm_interface.gemini.convert_gemini_models_to_ollama_response")
    def test_list(self, mock_convert):
        mock_models_list = [MockGenaiModel(name="model1")]
        self.mock_genai_client.models.list.return_value = mock_models_list
        mock_convert.return_value = "converted_list"

        result = self.gemini_wrapper.list()

        self.mock_genai_client.models.list.assert_called_once()
        mock_convert.assert_called_once_with(mock_models_list)
        self.assertEqual(result, "converted_list")

    @patch("llm_interface.gemini.translate_messages_for_gemini")
    @patch("llm_interface.gemini.translate_tools_for_gemini")
    def test_chat_basic(self, mock_translate_tools, mock_translate_messages):
        mock_translate_messages.return_value = [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]
        mock_response = MockGenaiResponse(
            text="Hi there!",
            usage_metadata=MockGenaiUsageMetadata(
                prompt_token_count=10, candidates_token_count=5, total_token_count=15
            ),
            candidates=[
                MockGenaiCandidate(
                    content=MockGenaiContent(parts=[MockGenaiPart(text="Hi there!")])
                )
            ],
        )
        self.mock_genai_client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = self.gemini_wrapper.chat(messages=messages, model="gemini-pro")

        mock_translate_messages.assert_called_once_with(messages)
        self.mock_genai_client.models.generate_content.assert_called_once()
        call_args = self.mock_genai_client.models.generate_content.call_args[1]
        self.assertEqual(call_args["model"], "gemini-pro")
        self.assertEqual(
            call_args["contents"], [{"role": "user", "parts": [{"text": "Hello"}]}]
        )
        self.assertNotIn("tools", call_args["config"])

        self.assertEqual(response["message"]["content"], "Hi there!")
        self.assertEqual(response["usage"]["prompt_tokens"], 10)
        self.assertEqual(response["usage"]["completion_tokens"], 5)
        self.assertEqual(response["usage"]["total_tokens"], 15)
        mock_translate_tools.assert_not_called()

    @patch("llm_interface.gemini.translate_messages_for_gemini")
    @patch("llm_interface.gemini.translate_tools_for_gemini")
    def test_chat_with_tools(self, mock_translate_tools, mock_translate_messages):
        mock_translate_messages.return_value = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]}
        ]
        mock_translate_tools.return_value = [
            {"function_declarations": [...]}
        ]  # Simplified
        mock_function_call = MockGenaiFunctionCall(
            name="get_weather", args={"location": "SF"}
        )
        mock_response = MockGenaiResponse(
            candidates=[
                MockGenaiCandidate(
                    content=MockGenaiContent(
                        parts=[MockGenaiPart(function_call=mock_function_call)]
                    )
                )
            ],
            usage_metadata=MockGenaiUsageMetadata(
                prompt_token_count=20, candidates_token_count=10, total_token_count=30
            ),
        )
        self.mock_genai_client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "What's the weather?"}]
        # Simplified tool definition for the test
        tools = [{"function": {"name": "get_weather", "description": "Get weather"}}]
        response = self.gemini_wrapper.chat(
            messages=messages, tools=tools, model="gemini-flash"
        )

        mock_translate_messages.assert_called_once_with(messages)
        mock_translate_tools.assert_called_once_with(tools)
        self.mock_genai_client.models.generate_content.assert_called_once()
        call_args = self.mock_genai_client.models.generate_content.call_args[1]
        self.assertEqual(call_args["model"], "gemini-flash")
        self.assertEqual(
            call_args["contents"],
            [{"role": "user", "parts": [{"text": "What's the weather?"}]}],
        )
        self.assertIn("tools", call_args["config"])
        self.assertEqual(
            call_args["config"]["tools"], [{"function_declarations": [...]}]
        )
        self.assertEqual(
            call_args["config"]["tool_config"]["function_calling_config"]["mode"],
            "AUTO",
        )

        self.assertEqual(response["message"]["content"], "")
        self.assertIn("tool_calls", response["message"])
        tool_call = response["message"]["tool_calls"][0]
        self.assertEqual(tool_call["name"], "get_weather")
        self.assertEqual(
            tool_call["arguments"], '{"location": "SF"}'
        )  # Note: Gemini returns dict, we dump to str
        self.assertEqual(response["usage"]["total_tokens"], 30)

    @patch("llm_interface.gemini.translate_messages_for_gemini")
    def test_chat_with_tool_response(self, mock_translate_messages):
        # Simulate messages including a user request, assistant tool call, and user tool response
        mock_translated_messages = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "SF"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {
                                "name": "get_weather",
                                "content": {"temp": "75F"},
                            },
                        }
                    }
                ],
            },
        ]
        mock_translate_messages.return_value = mock_translated_messages

        mock_response = MockGenaiResponse(
            text="The weather in SF is 75F.",
            usage_metadata=MockGenaiUsageMetadata(
                prompt_token_count=50, candidates_token_count=8, total_token_count=58
            ),
            candidates=[
                MockGenaiCandidate(
                    content=MockGenaiContent(
                        parts=[MockGenaiPart(text="The weather in SF is 75F.")]
                    )
                )
            ],
        )
        self.mock_genai_client.models.generate_content.return_value = mock_response

        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        }
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": '{"temp": "75F"}'},
        ]
        response = self.gemini_wrapper.chat(messages=messages, model="gemini-pro")

        mock_translate_messages.assert_called_once_with(messages)
        self.mock_genai_client.models.generate_content.assert_called_once()
        call_args = self.mock_genai_client.models.generate_content.call_args[1]
        self.assertEqual(call_args["contents"], mock_translated_messages)

        self.assertEqual(response["message"]["content"], "The weather in SF is 75F.")
        self.assertEqual(response["usage"]["total_tokens"], 58)

    @patch("llm_interface.gemini.translate_messages_for_gemini")
    def test_chat_with_system_message(self, mock_translate_messages):
        # System message is handled by prepending to contents in the wrapper
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        # Mock translation *without* system message, as wrapper handles it
        mock_translate_messages.return_value = [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]

        # Add usage_metadata to the mock response
        mock_response = MockGenaiResponse(
            text="Hi.",
            usage_metadata=MockGenaiUsageMetadata(
                prompt_token_count=5, candidates_token_count=2, total_token_count=7
            ),
            candidates=[
                MockGenaiCandidate(
                    content=MockGenaiContent(parts=[MockGenaiPart(text="Hi.")])
                )
            ],  # Add candidates for consistency
        )
        self.mock_genai_client.models.generate_content.return_value = mock_response

        response = self.gemini_wrapper.chat(
            messages=messages
        )  # Store response to check usage

        mock_translate_messages.assert_called_once_with(messages)
        call_args = self.mock_genai_client.models.generate_content.call_args[1]
        # Verify system message was added correctly
        expected_contents = [
            {"role": "user", "parts": [{"text": "System: Be concise."}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
        ]
        self.assertEqual(call_args["contents"], expected_contents)

        # Verify usage data is present in the response
        self.assertIn("usage", response)
        self.assertEqual(response["usage"]["prompt_tokens"], 5)
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertEqual(response["usage"]["total_tokens"], 7)
        self.assertEqual(response["message"]["content"], "Hi.")

    class MySchema(BaseModel):
        name: str
        age: int

    @patch("llm_interface.gemini.translate_messages_for_gemini")
    def test_chat_structured_output(self, mock_translate_messages):
        mock_translate_messages.return_value = [
            {"role": "user", "parts": [{"text": "Extract info"}]}
        ]
        # Gemini returns the structured data directly in the text field when schema is used
        # Use triple quotes for multiline JSON string
        mock_json_string = """{
 "name": "Bob",
 "age": 30
}"""
        mock_response = MockGenaiResponse(
            text=mock_json_string,  # Gemini returns JSON string
            usage_metadata=MockGenaiUsageMetadata(10, 5, 15),
            candidates=[
                MockGenaiCandidate(
                    content=MockGenaiContent(
                        parts=[MockGenaiPart(text=mock_json_string)]
                    )
                )
            ],
        )
        self.mock_genai_client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "Extract info"}]
        response = self.gemini_wrapper.chat(
            messages=messages, response_schema=self.MySchema
        )

        call_args = self.mock_genai_client.models.generate_content.call_args[1]
        self.assertEqual(call_args["config"]["response_mime_type"], "application/json")
        self.assertEqual(call_args["config"]["response_schema"], self.MySchema)

        # The wrapper returns the raw text, expecting the caller (LLMInterface) to parse
        self.assertEqual(response["message"]["content"], mock_json_string)

    def test_chat_error_handling(self):
        self.mock_genai_client.models.generate_content.side_effect = Exception(
            "Generic API error"
        )

        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(Exception) as cm:
            self.gemini_wrapper.chat(messages=messages)
        self.assertEqual(str(cm.exception), "Generic API error")


if __name__ == "__main__":
    unittest.main()
