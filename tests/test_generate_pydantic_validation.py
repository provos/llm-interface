import unittest

from pydantic import BaseModel

from llm_interface.llm_interface import LLMInterface


class Answer(BaseModel):
    value: int


class AlwaysAnswers:
    """A client that returns a well-formed answer every time."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return {
            "message": {"content": '{"value": 1}'},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "done": True,
        }


class TestValidationExhausted(unittest.TestCase):
    def test_returns_none_when_every_attempt_fails_validation(self):
        client = AlwaysAnswers()
        llm = LLMInterface(
            model_name="m",
            client=client,
            use_cache=False,
            support_structured_outputs=True,
        )
        result = llm.generate_pydantic(
            prompt_template="q",
            output_schema=Answer,
            extra_validation=lambda r: "still wrong",
        )
        self.assertIsNone(result)
        self.assertEqual(client.calls, 3)

    def test_returns_the_answer_once_validation_passes(self):
        client = AlwaysAnswers()
        llm = LLMInterface(
            model_name="m",
            client=client,
            use_cache=False,
            support_structured_outputs=True,
        )
        seen = []

        def validate(r):
            seen.append(r)
            return None if len(seen) >= 2 else "not yet"

        result = llm.generate_pydantic(
            prompt_template="q", output_schema=Answer, extra_validation=validate
        )
        self.assertEqual(result, Answer(value=1))
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
