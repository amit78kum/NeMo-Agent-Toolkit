from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class MathToolConfig(FunctionGroupBaseConfig, name="math_tools"):
    include: list[str] = Field(default_factory=lambda: [
        "add", "subtract", "multiply", "divide", "factorial", "power"
    ], description="List of math functions to include in the group.")


@register_function_group(config_type=MathToolConfig)
async def math_tools(_config: MathToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register a simple math tools function group.

    Exposes basic math operations for use in workflows and examples.
    """
    group = FunctionGroup(config=_config)

    async def _add(numbers: list[float]) -> float:
        if not isinstance(numbers, list) or not numbers:
            raise ValueError("Provide a non-empty list of numbers to add.")
        return sum(numbers)

    async def _subtract(a: float, b: float) -> float:
        if a is None or b is None:
            raise ValueError("Provide two numbers to subtract (a, b).")
        return a - b

    async def _multiply(numbers: list[float]) -> float:
        if not isinstance(numbers, list) or not numbers:
            raise ValueError("Provide a non-empty list of numbers to multiply.")
        result = 1.0
        for n in numbers:
            result *= n
        return result

    async def _divide(a: float, b: float) -> float:
        if a is None or b is None:
            raise ValueError("Provide two numbers to divide (a, b).")
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    async def _factorial(n: int) -> int:
        if n is None or not isinstance(n, int):
            raise ValueError("Provide an integer to compute factorial.")
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers.")
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    async def _power(a: float, b: float) -> float:
        if a is None or b is None:
            raise ValueError("Provide two numbers to compute power (a, b).")
        return a ** b

    group.add_function(name="add", fn=_add, description=_add.__doc__)
    group.add_function(name="subtract", fn=_subtract, description=_subtract.__doc__)
    group.add_function(name="multiply", fn=_multiply, description=_multiply.__doc__)
    group.add_function(name="divide", fn=_divide, description=_divide.__doc__)
    group.add_function(name="factorial", fn=_factorial, description=_factorial.__doc__)
    group.add_function(name="power", fn=_power, description=_power.__doc__)

    yield group
