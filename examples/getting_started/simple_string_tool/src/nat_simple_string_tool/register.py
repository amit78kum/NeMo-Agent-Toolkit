from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class StringToolConfig(FunctionGroupBaseConfig, name="string_tools"):
    include: list[str] = Field(default_factory=lambda: [
        "concatenate", "split", "count", "reverse", "upper", "lower"
    ], description="List of string functions to include in the group.")


@register_function_group(config_type=StringToolConfig)
async def string_tools(_config: StringToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register a simple string tools function group.

    Exposes basic string operations for use in workflows and examples.
    """
    group = FunctionGroup(config=_config)

    async def _concatenate(parts: list[str]) -> str:
        if not parts:
            raise ValueError("Provide one or more strings to concatenate.")
        return "".join(parts)

    async def _split(params: dict) -> list[str]:
        # Expect a dict with keys: `text` and optional `sep`.
        text = params.get("text")
        sep = params.get("sep", " ")
        if text is None:
            raise ValueError("Provide 'text' to split.")
        return text.split(sep)

    async def _count(text: str) -> int:
        if text is None:
            raise ValueError("Provide a text string to count characters.")
        return len(text)

    async def _reverse(text: str) -> str:
        if text is None:
            raise ValueError("Provide a text string to reverse.")
        return text[::-1]

    async def _upper(text: str) -> str:
        if text is None:
            raise ValueError("Provide a text string to uppercase.")
        return text.upper()

    async def _lower(text: str) -> str:
        if text is None:
            raise ValueError("Provide a text string to lowercase.")
        return text.lower()

    group.add_function(name="concatenate", fn=_concatenate, description=_concatenate.__doc__)
    group.add_function(name="split", fn=_split, description=_split.__doc__)
    group.add_function(name="count", fn=_count, description=_count.__doc__)
    group.add_function(name="reverse", fn=_reverse, description=_reverse.__doc__)
    group.add_function(name="upper", fn=_upper, description=_upper.__doc__)
    group.add_function(name="lower", fn=_lower, description=_lower.__doc__)

    yield group
