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

from httpcore import TimeoutException
from ollama import Client

from . import errors


class OllamaWrapper(Client):
    def chat(self, *args, **kwargs):
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
