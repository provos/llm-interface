import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Callable, Dict, List, Literal, Optional, get_type_hints

# typing.Union and (on 3.10+) the `X | None` shorthand both need to be
# recognized as "optional" wrappers when building JSON schemas.
_UNION_ORIGINS = (typing.Union,)
if hasattr(types, "UnionType"):  # pragma: no branch - always true on Python >= 3.10
    _UNION_ORIGINS = _UNION_ORIGINS + (types.UnionType,)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any] = field(repr=False)
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary format compatible with Ollama."""
        function: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.strict:
            function["strict"] = True
        return {
            "type": "function",
            "function": function,
        }

    def execute(self, **kwargs: Dict[str, Any]) -> Any:
        return self.func(**kwargs)


def _type_to_json_schema(type_hint: Any) -> Dict[str, Any]:
    """Convert Python type hints to JSON Schema types."""
    origin = typing.get_origin(type_hint)

    # Literal["a", "b", ...] -> {"type": <inferred>, "enum": [...]}
    if origin is Literal:
        values = typing.get_args(type_hint)
        value_types = {type(v) for v in values}
        if value_types == {bool}:
            inferred_type = "boolean"
        elif value_types == {int}:
            inferred_type = "integer"
        elif value_types and value_types <= {int, float}:
            inferred_type = "number"
        elif value_types == {str}:
            inferred_type = "string"
        else:
            inferred_type = "string"
        return {"type": inferred_type, "enum": list(values)}

    # Optional[X] / Union[X, None] / X | None -> the schema for X
    if origin in _UNION_ORIGINS:
        non_none_args = [
            arg for arg in typing.get_args(type_hint) if arg is not type(None)
        ]
        if len(non_none_args) == 1:
            return _type_to_json_schema(non_none_args[0])
        elif non_none_args:
            # Multiple non-None options: fall back to the first one.
            return _type_to_json_schema(non_none_args[0])
        return {"type": "string"}

    if type_hint == str:
        return {"type": "string"}
    elif type_hint == int:
        return {"type": "integer"}
    elif type_hint == float:
        return {"type": "number"}
    elif type_hint == bool:
        return {"type": "boolean"}
    elif type_hint == list or origin == list:
        item_type = Any
        args = typing.get_args(type_hint)
        if args:
            item_type = args[0]
        return {"type": "array", "items": _type_to_json_schema(item_type)}
    elif type_hint == dict or origin == dict:
        return {"type": "object"}
    else:
        return {"type": "string"}


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse a function's docstring into a main summary and detailed parameter docs.

    The docstring is split on the first blank line to extract the first paragraph as the
    main description. The 'Args:' section is then scanned (case-insensitive) to collect
    each parameter's description, handling continuation lines.

    Args:
        docstring (str): The full raw docstring of the function.

    Returns:
        tuple[str, dict[str, str]]: A pair where the first element is the main description
            (the first paragraph, dedented), and the second is a dictionary mapping each
            parameter name to its description from the Args section.
    """
    if not docstring:
        return "", {}

    # Split docstring into sections
    parts = re.split(r"\n\s*\n", dedent(docstring).strip())

    # Get main description (first paragraph)
    main_desc = parts[0].strip()

    # Parse parameter descriptions
    param_desc: Dict[str, str] = {}
    current_param = None
    in_args_section = False

    # Join all parts after the main description
    remaining_text = "\n".join(parts[1:]) if len(parts) > 1 else ""

    args_lines: List[str] = []
    for line in remaining_text.split("\n"):
        line = line.rstrip()

        # Check if we're entering the Args section
        if line.lower().endswith("args:"):
            in_args_section = True
            continue

        if not in_args_section:
            continue

        if line and not line.startswith(" "):
            in_args_section = False
            continue

        args_lines.append(line)

    # find parameter descriptions
    args_lines = dedent("\n".join(args_lines)).split("\n")
    for line in args_lines:
        # Check for new parameter
        if line and not line.startswith(" "):
            # Look for parameter definition (param: description)
            param_match = re.match(r"(\w+):\s*(.*)", line)
            if param_match:
                current_param = param_match.group(1)
                param_desc[current_param] = param_match.group(2)
        # Add to existing parameter description
        elif current_param and line:
            param_desc[current_param] = param_desc[current_param] + " " + line.strip()

    return main_desc, param_desc


def create_tool(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Tool:
    """
    Create a Tool instance from a Python function using its type hints and docstring.

    Args:
        func (Callable): The function to convert into a tool
        name (Optional[str]): Optional custom name for the tool. If not provided, uses the function name
        description (Optional[str]): Optional custom description. If not provided, uses the function's docstring

    Returns:
        Tool: A Tool instance representing the function

    Example:
        @create_tool
        def get_weather(location: str, units: str = "celsius") -> str:
            '''Get the weather for a specific location.

            Args:
                location: The city or location to get weather for
                units: Temperature units (celsius or fahrenheit)
            '''
            # Function implementation
            pass
    """
    # Get function metadata
    func_name = name or func.__name__
    func_doc = inspect.getdoc(func) or ""
    func_desc = description or func_doc.split("\n\n")[0] if func_doc else func_name

    # Parse docstring
    main_desc, param_docs = _parse_docstring(func_doc)
    func_desc = description or main_desc or func_name

    # Get type hints
    type_hints = get_type_hints(func)

    # Get default values
    signature = inspect.signature(func)

    # Build parameters schema
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    for param_name, param in signature.parameters.items():
        if param_name == "self":  # Skip self parameter for methods
            continue

        param_schema = _type_to_json_schema(type_hints.get(param_name, Any))

        # Add description if available
        if param_name in param_docs:
            param_schema["description"] = param_docs[param_name]

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            parameters["required"].append(param_name)

        parameters["properties"][param_name] = param_schema

    # OpenAI strict mode rejects schemas with optional properties, so only
    # enable it automatically when every parameter is required (no defaults).
    is_strict = len(parameters["required"]) == len(parameters["properties"])

    return Tool(
        name=func_name,
        description=func_desc,
        parameters=parameters,
        func=func,
        strict=is_strict,
    )


def tool(
    name: Optional[str] = None, description: Optional[str] = None
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to create a Tool from a function.

    Args:
        name (Optional[str]): Optional custom name for the tool
        description (Optional[str]): Optional custom description

    Returns:
        Callable: Decorator function that creates a Tool

    Example:
        @tool(name="weather", description="Get weather information")
        def get_weather(location: str, units: str = "celsius") -> str:
            '''Get the weather for a specific location.'''
            # Function implementation
            pass
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return create_tool(func, name=name, description=description)

    return decorator
