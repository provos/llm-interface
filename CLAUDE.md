# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Interface is a Python library that provides a unified API for working with multiple Language Model providers (OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter). It emphasizes structured outputs, caching, and enterprise-level features.

## Key Commands

### Development
```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests only
poetry run pytest --run-regression     # Run regression tests (requires API keys)

# Run a single test
poetry run pytest tests/test_file.py::TestClass::test_method

# Format code (required before commits)
poetry run black .

# Lint code
poetry run flake8

# Build package
poetry build

# Install in development mode
pip install -e .
```

## Architecture

### Core Design Pattern
The library uses a unified interface pattern where `LLMInterface` is the main entry point, with provider-specific wrapper classes inheriting from `BaseWrapper`:

- `llm_interface/llm_interface.py` - Main interface and factory function `llm_from_config()`
- `llm_interface/base_wrapper.py` - Abstract base class defining the provider interface
- Provider wrappers: `openai.py`, `anthropic.py`, `gemini.py`, `ollama.py`, `openrouter.py`

### Key Architectural Decisions

1. **Structured Outputs**: Uses Pydantic models for type-safe structured outputs. Each provider wrapper implements `_generate_structured()` differently based on provider capabilities.

2. **Tool Calling**: Implemented via decorators in `llm_interface/tools.py`. Tools are registered and converted to provider-specific formats.

3. **Caching**: Response caching uses `diskcache` library, implemented in the base wrapper. Cache keys are computed from request parameters.

4. **Configuration**: Supports both dictionary and YAML configuration. The `llm_from_config()` factory handles provider instantiation.

### Provider-Specific Implementation Notes

- **OpenAI**: Supports native structured outputs for newer models (gpt-4o-mini-2024-07-18+)
- **Anthropic**: Uses manual JSON parsing for structured outputs
- **Gemini**: Supports native response schemas, timeout in milliseconds
- **Ollama**: Supports both local and SSH connections, custom structured output via JSON mode
- **OpenRouter**: Routes to multiple providers, inherits OpenAI implementation

### Testing Strategy

Tests are organized by marker:
- `unit` - Fast, no external dependencies
- `integration` - Requires API access
- `regression` - Full provider testing

Mock providers in `tests/mock_providers.py` enable testing without API calls.

## Important Patterns

1. **Error Handling**: Custom exceptions in `llm_interface/llm_exceptions.py`. Always preserve exception chains.

2. **Logging**: Uses Python's logging module. Debug logging includes full request/response details.

3. **Timeout Handling**: Provider-specific timeout implementations. Gemini uses milliseconds, others use seconds.

4. **Token Tracking**: Token usage tracked in response metadata for cost monitoring.

5. **Type Safety**: Extensive type hints throughout. Run `mypy` for type checking.

## Pre-commit Hooks

The repository uses pre-commit hooks that automatically format code with Black and check with Flake8. These will run automatically on commit if configured.