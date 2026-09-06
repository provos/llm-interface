import unittest

from llm_interface.token_usage import TokenUsage


class TestTokenUsage(unittest.TestCase):
    def test_update_accumulates_every_counter(self):
        usage = TokenUsage()
        usage.update(
            prompt_tokens=10,
            completion_tokens=20,
            cached_tokens=5,
            cache_creation_tokens=7,
            reasoning_tokens=3,
        )
        usage.update(
            prompt_tokens=1,
            completion_tokens=2,
            cached_tokens=1,
            cache_creation_tokens=1,
            reasoning_tokens=1,
        )
        self.assertEqual(usage.prompt_tokens, 11)
        self.assertEqual(usage.completion_tokens, 22)
        self.assertEqual(usage.total_tokens, 33)
        self.assertEqual(usage.cached_tokens, 6)
        self.assertEqual(usage.cache_creation_tokens, 8)
        self.assertEqual(usage.reasoning_tokens, 4)

    def test_str_and_stats_mention_cache_writes(self):
        usage = TokenUsage()
        usage.update(prompt_tokens=1, completion_tokens=1, cache_creation_tokens=42)
        self.assertIn("42 cache writes", str(usage))
        self.assertEqual(usage.get_all_stats()["cache_creation_tokens"], 42)
        self.assertNotIn("cached_tokens", usage.get_all_stats())

    def test_reset_clears_cache_writes(self):
        usage = TokenUsage()
        usage.update(cache_creation_tokens=9)
        usage.reset()
        self.assertEqual(usage.cache_creation_tokens, 0)


if __name__ == "__main__":
    unittest.main()
