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
from typing import Dict

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """
    Tracks token usage statistics for LLM interactions with one partiular model.

    This class provides a standardized way to track token usage across different
    LLM providers, including input tokens, output tokens, and provider-specific metrics.

    Attributes:
        prompt_tokens (int): Number of tokens used in the prompt/input
        completion_tokens (int): Number of tokens generated in the response
        total_tokens (int): Total tokens used (prompt + completion)
        cached_tokens (int): Number of tokens retrieved from cache (if supported)
        reasoning_tokens (int): Number of tokens used for reasoning (if supported)
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def update(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ):
        """
        Updates token usage statistics.

        Args:
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion
            total_tokens (int): Total tokens used
            cached_tokens (int): Number of tokens from cache
            reasoning_tokens (int): Number of reasoning tokens
            **kwargs: Additional provider-specific metrics
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        # If total_tokens is provided, use it; otherwise, add prompt and completion
        if total_tokens > 0:
            self.total_tokens += total_tokens
        else:
            self.total_tokens += prompt_tokens + completion_tokens

        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens

    def reset(self):
        """Resets all token usage statistics to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0

    def get_all_stats(self) -> Dict[str, int]:
        """
        Returns all token usage statistics as a dictionary.

        Returns:
            dict: All token usage statistics
        """
        stats = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

        # Only include non-zero values for optional fields
        if self.cached_tokens > 0:
            stats["cached_tokens"] = self.cached_tokens
        if self.reasoning_tokens > 0:
            stats["reasoning_tokens"] = self.reasoning_tokens

        return stats

    def __str__(self) -> str:
        """String representation of token usage."""
        result = f"Token usage: {self.total_tokens} total ({self.prompt_tokens} prompt, {self.completion_tokens} completion)"
        if self.cached_tokens > 0:
            result += f", {self.cached_tokens} cached"
        if self.reasoning_tokens > 0:
            result += f", {self.reasoning_tokens} reasoning"
        return result
