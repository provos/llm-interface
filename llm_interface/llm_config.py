import os
import re
from datetime import datetime
from typing import Literal, Optional

from .anthropic import AnthropicWrapper
from .gemini import GeminiWrapper
from .llm_interface import LLMInterface
from .openai import OpenAIWrapper
from .openrouter import OpenRouterWrapper
from .remote_ollama import RemoteOllama
from .ssh import SSHConnection


def _parse_model_date(model_name: str) -> Optional[datetime]:
    """Extract date from model name if present."""
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", model_name)
    if date_match:
        try:
            return datetime.strptime(date_match.group(0), "%Y-%m-%d")
        except ValueError:
            return None
    return None


def supports_structured_output(model_name: str) -> bool:
    """
    Check if a model supports structured outputs based on its name and version date.

    Args:
        model_name: Name of the OpenAI model

    Returns:
        bool: Whether the model supports structured outputs
    """
    # Models that always support structured outputs (no date requirements)
    base_models = {
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "o1",
    }

    # Models with minimum date requirements
    dated_model_requirements = {
        "gpt-4o-mini": datetime(2024, 7, 18),
        "gpt-4o": datetime(2024, 8, 6),
        "o3-mini": datetime(2025, 1, 31),
        "o1": datetime(2024, 12, 17),
    }

    # First check for exact match in base models
    if model_name in base_models:
        return True

    # Check if model has a date
    model_date = _parse_model_date(model_name)
    if not model_date:
        return False

    # Extract base model name (without date)
    base_model = re.sub(r"-\d{4}-\d{1,2}-\d{1,2}", "", model_name)

    # If we have a base model with date requirements, check the date
    if base_model in dated_model_requirements:
        required_date = dated_model_requirements[base_model]
        return model_date >= required_date

    return False


def llm_from_config(
    provider: Literal[
        "ollama", "remote_ollama", "openai", "anthropic", "gemini", "openrouter"
    ] = "ollama",
    model_name: str = "llama3",
    max_tokens: int = 4096,
    host: Optional[str] = None,
    hostname: Optional[str] = None,
    username: Optional[str] = None,
    log_dir: str = "logs",
    use_cache: bool = True,
    timeout: float = 600.0,
) -> LLMInterface:
    """
    Creates and configures a language model interface based on specified provider and parameters.

    This function initializes a LLMInterface instance with the appropriate wrapper/client
    based on the selected provider (ollama, remote_ollama, openai, or anthropic).

    Args:
        provider (Literal["ollama", "remote_ollama", "openai", "anthropic", "gemini", "openrouter"]): The LLM provider to use.
            Defaults to "ollama".
        model_name (str): Name of the model to use. Defaults to "llama3".
        max_tokens (int): Maximum number of tokens for model responses. Defaults to 4096.
        host (Optional[str]): Host address for local ollama instance. Only used with "ollama" provider.
        hostname (Optional[str]): Remote hostname for SSH connection. Required for "remote_ollama".
        username (Optional[str]): Username for SSH connection. Required for "remote_ollama".
        log_dir (str): Directory for storing logs. Defaults to "logs".
        use_cache (bool): Whether to cache model responses. Defaults to True.
        timeout (float): Timeout in seconds for model requests. Defaults to 600.0.

    Returns:
        LLMInterface: Configured interface for interacting with the specified LLM.

    Raises:
        ValueError: If required API keys are not found in environment variables,
            or if an invalid provider is specified.

    Examples:
        >>> # Create an OpenAI interface
        >>> llm = llm_from_config(provider="openai", model_name="gpt-4")

        >>> # Create a local Ollama interface
        >>> llm = llm_from_config(provider="ollama", model_name="llama2")

        >>> # Create a remote Ollama interface
        >>> llm = llm_from_config(
        ...     provider="remote_ollama",
        ...     hostname="example.com",
        ...     username="user"
        ... )
    """
    match provider:
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            wrapper = OpenAIWrapper(
                api_key=api_key, max_tokens=max_tokens, timeout=timeout
            )

            support_structured_outputs = supports_structured_output(model_name)
            support_json_mode = model_name not in ["o1-mini", "o1-preview"]
            support_system_prompt = model_name not in ["o1-mini", "o1-preview"]

            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=support_json_mode,
                support_structured_outputs=support_structured_outputs,
                support_system_prompt=support_system_prompt,
                use_cache=use_cache,
                timeout=timeout,
            )
        case "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            wrapper = AnthropicWrapper(api_key=api_key, max_tokens=max_tokens)
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=False,
                use_cache=use_cache,
                timeout=timeout,
            )
        case "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            wrapper = GeminiWrapper(api_key=api_key, max_tokens=max_tokens)
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=True,
                support_structured_outputs=True,  # Gemini supports structured outputs
                support_system_prompt=True,
                use_cache=use_cache,
                timeout=timeout,
            )
        case "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OPENROUTER_API_KEY not found in environment variables"
                )

            # Get optional site info from env vars for analytics/rankings
            site_url = os.getenv("OPENROUTER_SITE_URL")
            site_name = os.getenv("OPENROUTER_SITE_NAME")

            wrapper = OpenRouterWrapper(
                api_key=api_key,
                max_tokens=max_tokens,
                site_url=site_url,
                site_name=site_name,
            )

            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=wrapper,
                support_json_mode=True,
                support_structured_outputs=True,
                support_system_prompt=True,
                use_cache=use_cache,
                timeout=timeout,
            )
        case "ollama" | "remote_ollama":
            # Enable structured outputs for Llama 3+ models
            supports_structured = True
            if provider == "remote_ollama":
                ssh = SSHConnection(
                    hostname=hostname,
                    username=username,
                )
                client = RemoteOllama(ssh_connection=ssh, model_name=model_name)
            else:
                client = None
            requires_thinking = model_name.lower().startswith("deepseek-r")
            return LLMInterface(
                model_name=model_name,
                log_dir=log_dir,
                client=client,
                host=host,
                support_json_mode=True,
                support_structured_outputs=supports_structured,
                requires_thinking=requires_thinking,
                use_cache=use_cache,
                timeout=timeout,
            )

    raise ValueError(f"Invalid LLM provider in config: {provider}")
