# Copyright 2024 Niels Provos
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

from typing import Any, List, Optional, Sequence

from httpcore import TimeoutException
from ollama import Client

from . import errors


def strip_is_error(
    messages: Optional[Sequence[Any]],
) -> Optional[Sequence[Any]]:
    """Strip the internal `is_error` marker from tool messages before sending them
    to Ollama.

    `is_error` is added by LLMInterface's tool loop to flag a failed tool call
    (see `LLMInterface._execute_tool_calls`); Ollama's Message type doesn't know
    about it, so it must not be forwarded.
    """
    if not messages:
        return messages

    cleaned: List[Any] = []
    changed = False
    for message in messages:
        if isinstance(message, dict) and "is_error" in message:
            message = {k: v for k, v in message.items() if k != "is_error"}
            changed = True
        cleaned.append(message)

    return cleaned if changed else messages


class OllamaWrapper(Client):
    def chat(self, *args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = strip_is_error(kwargs["messages"])
        try:
            return super().chat(*args, **kwargs)
        except TimeoutException as e:
            # Handle the timeout error
            return {
                "error": f"The request timed out: {e}",
                "error_type": errors.TIMEOUT,
                "content": None,
                "done": False,
                "usage": None,
            }
