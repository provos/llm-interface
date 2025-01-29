import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from llm_interface.anthropic import AnthropicWrapper


class TestAnthropicWrapper(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.anthropic_wrapper = AnthropicWrapper(api_key=self.api_key)

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


if __name__ == "__main__":
    unittest.main()
