import unittest
from datetime import datetime

from llm_interface.llm_config import supports_structured_output, _parse_model_date


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


if __name__ == "__main__":
    unittest.main()
