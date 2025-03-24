import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from anthropic import APIConnectionError, APIError, APITimeoutError

from llm_interface.anthropic import AnthropicWrapper


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

        self.mock_client.messages.create.return_value = mock_response

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
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["system"], "You are a helpful assistant.")
        self.assertEqual(call_args["model"], "claude-3-sonnet-20240229")
        self.assertEqual(len(call_args["messages"]), 1)

        # Verify response format
        self.assertEqual(response["message"]["content"], "Hello, I'm Claude!")
        self.assertEqual(response["usage"]["prompt_tokens"], 10)
        self.assertEqual(response["usage"]["completion_tokens"], 5)
        self.assertEqual(response["usage"]["total_tokens"], 15)
        self.assertTrue(response["done"])

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

        self.mock_client.messages.create.return_value = mock_response

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
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
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

        self.mock_client.messages.create.return_value = mock_response

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
        call_args = self.mock_client.messages.create.call_args[1]
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

        self.mock_client.messages.create.return_value = mock_response

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
        call_args = self.mock_client.messages.create.call_args[1]
        translated_messages = call_args["messages"]
        self.assertEqual(len(translated_messages), 1)
        self.assertEqual(len(translated_messages[0]["content"]), 2)  # Text + image
        self.assertEqual(translated_messages[0]["content"][1]["type"], "image")

        # Verify response
        self.assertEqual(response["message"]["content"], "I see a cat in the image.")

    def test_chat_error_handling(self):
        # Test timeout error
        self.mock_client.messages.create.side_effect = APITimeoutError(request=Mock())
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "timeout")

        # Test connection error
        self.mock_client.messages.create.side_effect = APIConnectionError(
            request=Mock(), message="Connection error"
        )
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "connection")

        # Test API error
        self.mock_client.messages.create.side_effect = APIError(
            request=Mock(), message="API error", body=None
        )
        response = self.anthropic_wrapper.chat([{"role": "user", "content": "Hello"}])
        self.assertIn("error", response)
        self.assertEqual(response["error_type"], "provider_specific")


if __name__ == "__main__":
    unittest.main()
