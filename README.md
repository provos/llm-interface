# LLM Interface

[![PyPI version](https://badge.fury.io/py/llm-interface.svg)](https://badge.fury.io/py/llm-interface)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-interface.svg)](https://pypi.org/project/llm-interface/)
[![codecov](https://codecov.io/gh/provos/llm-interface/graph/badge.svg?token=A5HBFRHACO)](https://codecov.io/gh/provos/llm-interface)

A flexible Python interface for working with various Language Model providers, including OpenAI, Anthropic, and Ollama. This library provides a unified way to interact with different LLM providers while supporting features like structured outputs, tool execution, and response caching.

## Features

- **Multiple Provider Support**
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Ollama (local and remote)
  - Remote Ollama via SSH

- **Advanced Capabilities**
  - Structured output parsing with Pydantic models
  - Function/tool calling support
  - Response caching
  - Comprehensive logging
  - JSON mode support
  - System prompt handling

- **Developer-Friendly**
  - Type hints throughout
  - Extensive test coverage
  - Flexible configuration options
  - Error handling and retries

## Installation

Install using pip:

```bash
pip install llm-interface
```

Or using Poetry:

```bash
poetry add llm-interface
```

## Basic Usage

### Simple Chat Completion

```python
from llm_interface import llm_from_config

# Create an OpenAI interface
llm = llm_from_config(
    provider="openai",
    model_name="gpt-4",
)

# Simple chat
response = llm.chat([
    {"role": "user", "content": "What is the capital of France?"}
])
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class LocationInfo(BaseModel):
    city: str
    country: str
    population: int

response = llm.generate_pydantic(
    prompt_template="Provide information about Paris",
    output_schema=LocationInfo,
    system="You are a helpful geography assistant"
)
```

### Tool/Function Calling

```python
from llm_interface.llm_tool import tool

@tool(name="get_weather")
def get_weather(location: str, units: str = "celsius") -> str:
    """Get weather information for a location.
    
    Args:
        location: City or location name
        units: Temperature units (celsius/fahrenheit)
    """
    # Implementation here
    return f"Weather in {location}"

response = llm.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather]
)
```

### Remote Ollama Setup

```python
llm = llm_from_config(
    provider="remote_ollama",
    model_name="llama2",
    hostname="example.com",
    username="user"
)
```

## Configuration

The library supports various configuration options through the `llm_from_config` function:

```python
llm = llm_from_config(
    provider="openai",          # "openai", "anthropic", "ollama", or "remote_ollama"
    model_name="gpt-4",        # Model name
    max_tokens=4096,           # Maximum tokens in response
    host=None,                 # Local Ollama host
    hostname=None,             # Remote SSH hostname
    username=None,             # Remote SSH username
    log_dir="logs",           # Directory for logs
    use_cache=True            # Enable response caching
)
```

## Environment Variables

Required environment variables based on provider:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Remote Ollama: requires an SSH key to be loaded in SSH agent

## Development

This project uses Poetry for dependency management:

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .

# Run linter
poetry run flake8
```

## License

Apache License 2.0 - See LICENSE file for details.