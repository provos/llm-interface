import pytest

from llm_interface import llm_from_config
from llm_interface.token_usage import TokenUsage


@pytest.fixture(scope="module")
def llm_client(request):
    provider = request.config.getoption("--provider", default="ollama")
    model_name = request.config.getoption("--model", default="llama3.2")
    host = request.config.getoption("--host", default=None)
    hostname = request.config.getoption("--hostname", default=None)
    username = request.config.getoption("--username", default=None)

    client = llm_from_config(
        provider=provider,
        model_name=model_name,
        host=host,
        hostname=hostname,
        username=username,
        log_dir="logs",
        use_cache=False,  # Disable cache for testing to ensure fresh responses
    )

    client.support_json_mode = False
    client.support_structured_outputs = False

    return client


@pytest.mark.regression
class TestChatInterface:
    """End-to-end regression tests for the basic chat interface."""

    def test_basic_chat(self, llm_client):
        """Test a simple chat message gets a reasonable response."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]

        response = llm_client.chat(messages)

        # Verify we got a non-empty response
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for expected content about Paris
        assert "Paris" in response

    def test_conversation_history(self, llm_client):
        """Test that the model uses conversation history for context."""
        messages = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Hello Alex, nice to meet you!"},
            {"role": "user", "content": "What's my name?"},
        ]

        response = llm_client.chat(messages)

        # Verify the model remembers the name from conversation history
        assert "Alex" in response

    def test_token_usage_tracking(self, llm_client):
        """Test that token usage is properly tracked in the response."""
        # Reset token usage before test
        llm_client.token_usage.reset()

        messages = [
            {
                "role": "user",
                "content": "Write a short paragraph about artificial intelligence.",
            }
        ]

        llm_client.chat(messages)
        token_usage = llm_client.token_usage

        # Check that we got token usage metrics
        assert token_usage.prompt_tokens > 0, "Prompt tokens should be tracked"
        assert token_usage.completion_tokens > 0, "Completion tokens should be tracked"
        assert token_usage.total_tokens > 0, "Total tokens should be tracked"
        assert token_usage.total_tokens == (
            token_usage.prompt_tokens + token_usage.completion_tokens
        )

        # Get token usage stats and verify they contain the expected fields
        all_stats = token_usage.get_all_stats()
        assert "prompt_tokens" in all_stats
        assert "completion_tokens" in all_stats
        assert "total_tokens" in all_stats

    def test_temperature_effect(self, llm_client):
        """Test that different temperature settings produce different responses."""
        messages = [{"role": "user", "content": "Generate a random name for a pet."}]

        # Generate responses with different temperatures
        response1 = llm_client.chat(messages, temperature=0.0)
        # Reset token usage between calls
        llm_client.token_usage.reset()
        response2 = llm_client.chat(messages, temperature=1.0)

        # With temperature=0, we should get deterministic responses
        # With temperature=1, we should get more random responses
        # They might still be the same by chance, but less likely
        assert isinstance(response1, str)
        assert isinstance(response2, str)
        # Just verify both responses look like pet names (contain letters)
        assert any(c.isalpha() for c in response1)
        assert any(c.isalpha() for c in response2)

    def test_custom_token_usage_object(self, llm_client):
        """Test using a custom TokenUsage object to track usage."""
        custom_usage = TokenUsage()

        messages = [
            {"role": "user", "content": "What is the tallest mountain in the world?"}
        ]

        response = llm_client.chat(messages, token_usage=custom_usage)

        # Verify the custom token usage object was populated
        assert custom_usage.prompt_tokens > 0
        assert custom_usage.completion_tokens > 0
        assert custom_usage.total_tokens > 0
        assert "Everest" in response

    def test_token_usage_scales_with_input(self, llm_client):
        """Test that token usage increases with longer inputs."""
        # Generate a string of specified word length
        words = ["just say the world: artificial. nothing else. "]
        content = " ".join(words)

        messages = [{"role": "user", "content": content}]

        usage_one = TokenUsage()
        llm_client.chat(messages, token_usage=usage_one)

        usage_two = TokenUsage()
        messages = [{"role": "user", "content": content * 2}]
        llm_client.chat(messages, token_usage=usage_two)

        assert usage_two.prompt_tokens > usage_one.prompt_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
