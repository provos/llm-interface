import unittest
from datetime import datetime
from unittest.mock import patch  # Added patch

# Added llm_from_config import
from llm_interface.llm_config import (
    _parse_model_date,
    llm_from_config,
    supports_structured_output,
)


class TestLLMConfig(unittest.TestCase):
    def test_base_models_without_dates(self):
        """Test models that should always support structured outputs."""
        self.assertTrue(supports_structured_output("gpt-4o"))
        self.assertTrue(supports_structured_output("gpt-4o-mini"))
        self.assertTrue(supports_structured_output("o3-mini"))
        self.assertTrue(supports_structured_output("o1"))

    def test_dated_models(self):
        """Test models with dates against their minimum version requirements."""
        # Models meeting minimum date requirements
        self.assertTrue(supports_structured_output("gpt-4o-mini-2024-07-18"))
        self.assertTrue(supports_structured_output("gpt-4o-2024-08-06"))
        self.assertTrue(supports_structured_output("gpt-4o-2024-11-20"))
        self.assertTrue(supports_structured_output("o3-mini-2025-01-31"))
        self.assertTrue(supports_structured_output("o1-2024-12-17"))

        # Models with dates before minimum requirements
        self.assertFalse(supports_structured_output("gpt-4o-mini-2024-07-17"))
        self.assertFalse(supports_structured_output("gpt-4o-2024-08-05"))
        self.assertFalse(supports_structured_output("o3-mini-2025-01-30"))
        self.assertFalse(supports_structured_output("o1-2024-12-16"))

    def test_gpt5_models(self):
        """Test that GPT-5 and its variants automatically support structured outputs."""
        # Base GPT-5 models
        self.assertTrue(supports_structured_output("gpt-5"))
        self.assertTrue(supports_structured_output("gpt-5-mini"))
        self.assertTrue(supports_structured_output("gpt-5-nano"))

        # GPT-5 models with dates
        self.assertTrue(supports_structured_output("gpt-5-2025-01-01"))
        self.assertTrue(supports_structured_output("gpt-5-mini-2025-06-15"))
        self.assertTrue(supports_structured_output("gpt-5-nano-2026-12-31"))

        # Future GPT versions should also work
        self.assertTrue(supports_structured_output("gpt-6"))
        self.assertTrue(supports_structured_output("gpt-6-mini"))
        self.assertTrue(supports_structured_output("gpt-10"))
        self.assertTrue(supports_structured_output("gpt-10-ultra-2030-01-01"))

    def test_unsupported_models(self):
        """Test models that should not support structured outputs."""
        self.assertFalse(supports_structured_output("o1-mini"))
        self.assertFalse(supports_structured_output("o1-preview"))
        self.assertFalse(supports_structured_output("unknown-model"))

    def test_invalid_date_formats(self):
        """Test models with invalid date formats."""
        self.assertFalse(
            supports_structured_output("gpt-4o-mini-2024-13-01")
        )  # Invalid month
        self.assertFalse(
            supports_structured_output("gpt-4o-mini-2024-12-32")
        )  # Invalid day
        self.assertFalse(
            supports_structured_output("gpt-4o-mini-invalid-date")
        )  # Malformed date

    def test_parse_model_date(self):
        """Test date parsing from model names."""
        self.assertEqual(_parse_model_date("model-2024-01-31"), datetime(2024, 1, 31))
        self.assertEqual(_parse_model_date("model-2024-12-01"), datetime(2024, 12, 1))
        self.assertIsNone(_parse_model_date("model-invalid-date"))
        self.assertIsNone(_parse_model_date("model-2024-13-01"))  # Invalid month
        self.assertIsNone(_parse_model_date("model"))  # No date

    @patch("llm_interface.llm_config.os.getenv")
    def test_support_flags_defaults(self, mock_getenv):
        """Test the default support flags set by llm_from_config for various providers."""
        # Mock getenv to return a dummy API key for providers that require it
        mock_getenv.return_value = "dummy_api_key"

        # OpenAI - Supports structured, supports JSON
        llm_openai_struct = llm_from_config(
            provider="openai", model_name="gpt-4o-2024-08-06"
        )
        self.assertTrue(llm_openai_struct.support_structured_outputs)
        self.assertTrue(llm_openai_struct.support_json_mode)

        # OpenAI - Does NOT support structured, supports JSON
        llm_openai_no_struct = llm_from_config(
            provider="openai", model_name="gpt-4o-mini-2024-07-17"
        )
        self.assertFalse(llm_openai_no_struct.support_structured_outputs)
        self.assertTrue(llm_openai_no_struct.support_json_mode)

        # OpenAI - o1-mini (special case) - No structured, No JSON
        llm_openai_o1_mini = llm_from_config(provider="openai", model_name="o1-mini")
        self.assertFalse(
            llm_openai_o1_mini.support_structured_outputs
        )  # supports_structured_output returns False
        self.assertFalse(
            llm_openai_o1_mini.support_json_mode
        )  # Explicitly set to False in llm_config

        # Anthropic - No JSON nor structured outputs in LLMInterface
        llm_anthropic = llm_from_config(
            provider="anthropic", model_name="claude-3-opus-20240229"
        )
        self.assertFalse(llm_anthropic.support_json_mode)
        self.assertFalse(llm_anthropic.support_structured_outputs)

        # Gemini - Supports JSON, Supports Structured
        llm_gemini = llm_from_config(provider="gemini", model_name="gemini-pro")
        self.assertTrue(llm_gemini.support_json_mode)
        self.assertTrue(llm_gemini.support_structured_outputs)

        # OpenRouter - Supports JSON, Supports Structured
        llm_openrouter = llm_from_config(
            provider="openrouter", model_name="google/gemini-flash-1.5"
        )
        self.assertTrue(llm_openrouter.support_json_mode)
        self.assertTrue(llm_openrouter.support_structured_outputs)

        # Ollama - Supports JSON, Supports Structured
        llm_ollama = llm_from_config(provider="ollama", model_name="llama3")
        self.assertTrue(llm_ollama.support_json_mode)
        self.assertTrue(llm_ollama.support_structured_outputs)

    @patch("llm_interface.llm_config.os.getenv")
    def test_support_flags_override_true(self, mock_getenv):
        """Test overriding support flags to True."""
        mock_getenv.return_value = "dummy_api_key"

        # Override Anthropic (defaults: json=False, structured=False)
        llm_anthropic = llm_from_config(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            json_mode=True,  # Override
            structured_outputs=True,  # Override
        )
        self.assertTrue(llm_anthropic.support_json_mode)
        self.assertTrue(llm_anthropic.support_structured_outputs)

        # Override OpenAI o1-mini (defaults: json=False, structured=False)
        llm_o1_mini = llm_from_config(
            provider="openai",
            model_name="o1-mini",
            json_mode=True,  # Override
            structured_outputs=True,  # Override
        )
        self.assertTrue(llm_o1_mini.support_json_mode)
        self.assertTrue(llm_o1_mini.support_structured_outputs)

    @patch("llm_interface.llm_config.os.getenv")
    def test_support_flags_override_false(self, mock_getenv):
        """Test overriding support flags to False."""
        mock_getenv.return_value = "dummy_api_key"

        # Override Ollama (defaults: json=True, structured=True)
        llm_ollama = llm_from_config(
            provider="ollama",
            model_name="llama3",
            json_mode=False,  # Override
            structured_outputs=False,  # Override
        )
        self.assertFalse(llm_ollama.support_json_mode)
        self.assertFalse(llm_ollama.support_structured_outputs)

        # Override Gemini (defaults: json=True, structured=True)
        llm_gemini = llm_from_config(
            provider="gemini",
            model_name="gemini-pro",
            json_mode=False,  # Override
            structured_outputs=False,  # Override
        )
        self.assertFalse(llm_gemini.support_json_mode)
        self.assertFalse(llm_gemini.support_structured_outputs)


if __name__ == "__main__":
    unittest.main()
