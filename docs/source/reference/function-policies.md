<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Function Policies

## Overview

Function policies provide a standardized way to intercept and modify function calls in the NeMo Agent toolkit. They work by wrapping functions with hooks that execute before (`on_pre_invoke`) and after (`on_post_invoke`) the function runs.

Think of policies as a pipeline:

1. **User calls function** → Original input enters the pipeline
2. **Pre-invoke policies run** → Each policy can transform the input
3. **Function executes** → Gets the potentially modified input
4. **Post-invoke policies run** → Each policy can transform the output
5. **User receives result** → Gets the potentially modified output

Multiple policies chain together, with each policy receiving the output from the previous policy. Policies are executed in the order you configure them.

## Key Concepts

**Function Policy**: A component that intercepts function execution via `on_pre_invoke` and `on_post_invoke` hooks. Policies receive context objects for inspection and return transformed values.

**Context Objects**: Immutable objects passed to policy hooks containing execution state:
- `PreInvokeContext` - Current input state before execution
- `PostInvokeContext` - Complete execution history after the function runs

**Policy Chaining**: Multiple policies execute in sequence. Each policy receives the value returned by the previous policy, forming a transformation pipeline.

**Dynamic Middleware**: The middleware component (`dynamic_middleware`) that discovers functions and applies configured policies to them.

## Component-Based Architecture

Function policies are first-class components in NAT, just like LLMs, embedders, and retrievers. You configure them in YAML and reference them in middleware configuration.

```yaml
# Define policies in the function_policies section
function_policies:
  input_validator:
    _type: input_validation
    enabled: true
    max_length: 1000

  output_sanitizer:
    _type: output_sanitization
    enabled: true
    sanitize_pii: true

# Apply policies via middleware
middleware:
  security_middleware:
    _type: dynamic_middleware
    register_workflow_functions: true
    pre_invoke_policy: ["input_validator"]
    post_invoke_policy: ["output_sanitizer"]

# Middleware applies to workflow functions
workflow:
  _type: react_agent
  middleware: ["security_middleware"]
  # Other workflow config...
```

## Built-in Logging Policy

NAT includes a built-in `logging` policy for observing function inputs and outputs:

```yaml
function_policies:
  my_logger:
    _type: logging
    enabled: true
    log_level: INFO
    max_value_length: 200

middleware:
  observer:
    _type: dynamic_middleware
    register_llms: true
    pre_invoke_policy: ["my_logger"]
    post_invoke_policy: ["my_logger"]
```

This logs pre-invoke and post-invoke context values without modifying them.

## Creating Custom Function Policies

### Step 1: Define the Configuration

Create a configuration class inheriting from `FunctionPolicyBaseConfig`:

```python
from pydantic import Field
from nat.data_models.function_policy import FunctionPolicyBaseConfig


class LoggingPolicyConfig(FunctionPolicyBaseConfig, name="logging"):
    """Configuration for logging policy."""

    log_level: str = Field(default="INFO", description="Logging level")
    max_value_length: int = Field(default=200, description="Max length for logged values")
```

### Step 2: Implement the Policy Class

Create the policy class inheriting from `FunctionPolicyBase`:

```python
from nat.function_policy.interface import FunctionPolicyBase
from nat.function_policy.interface import PreInvokeContext, PostInvokeContext
from typing import Any
import logging

logger = logging.getLogger(__name__)


class LoggingPolicy(FunctionPolicyBase[LoggingPolicyConfig]):
    """Policy that logs function inputs and outputs without modification."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
        """Log pre-invoke context."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.log(
            level,
            "[%s] pre_invoke: function=%s, args=%s",
            self.name,
            context.function_context.name,
            self._truncate(context.function_args),
        )
        return None  # Don't modify args

    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        """Log post-invoke context."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.log(
            level,
            "[%s] post_invoke: function=%s, output=%s",
            self.name,
            context.function_context.name,
            self._truncate(context.function_output),
        )
        return None  # Don't modify output

    def _truncate(self, value: Any) -> str:
        """Truncate value string representation if too long."""
        s = repr(value)
        if len(s) > self.config.max_value_length:
            return s[: self.config.max_value_length] + "..."
        return s
```

### Step 3: Register the Policy

Use the `@register_function_policy` decorator to register your policy:

```python
from nat.cli.register_workflow import register_function_policy
from nat.builder.builder import Builder


@register_function_policy(config_type=LoggingPolicyConfig)
async def logging_policy(config: LoggingPolicyConfig, builder: Builder):
    """Build a logging policy from configuration."""
    yield LoggingPolicy(config=config)
```

## Policy Execution Model

### Pre-Invoke Pipeline

When a function is called, pre-invoke policies execute in order:

```
Original Args ((10,))
    ↓
Policy 1: Add 5 → Returns (15,)
    ↓
Policy 2: Multiply by 2 → Returns (30,)
    ↓
Function receives: 30
```

Each policy receives the output from the previous policy via `context.function_args`. The `context.original_args` always contains the original args the user provided.

### Post-Invoke Pipeline

After the function runs, post-invoke policies execute in order:

```
Function returns: 100
    ↓
Policy 1: Log result → Returns 100
    ↓
Policy 2: Round to nearest 10 → Returns 100
    ↓
User receives: 100
```

Each policy receives the output from the previous policy as the second argument. The `PostInvokeContext` provides a complete audit trail with `original_args`, `function_args`, and `function_output`.

### Return Value Behavior

**Transforming Values**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    # Modify the first arg, return same-length tuple
    modified_first = context.function_args[0] + 1
    return (modified_first,) + context.function_args[1:]
```

**Preserving Values**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    # Log or validate without modifying
    validate(context.function_args[0])
    return context.function_args  # Or return None to preserve
```

**Blocking Execution**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    if is_malicious(context.function_args[0]):
        raise SecurityException("Blocked malicious input")
    return context.function_args
```

## Working with Context Objects

### PreInvokeContext

Contains the current args state:

```python
@dataclass
class PreInvokeContext:
    function_context: FunctionMiddlewareContext  # Function metadata
    original_args: tuple[Any, ...]  # What the user sent
    function_args: tuple[Any, ...]  # Current args (after previous policies)
    function_kwargs: dict[str, Any]  # Keyword arguments (read-only)
```

**Usage example**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    # Check if args were already modified
    if context.function_args != context.original_args:
        logger.info("Args were modified by previous policy")

    # Get function name for logging
    logger.info(f"Processing {context.function_context.name}")

    # Modify first arg, return same-length tuple
    modified_first = transform(context.function_args[0])
    return (modified_first,) + context.function_args[1:]
```

### PostInvokeContext

Contains the complete execution state (frozen/immutable):

```python
@dataclass(frozen=True)
class PostInvokeContext:
    function_context: FunctionMiddlewareContext  # Function metadata
    original_args: tuple[Any, ...]  # What the user sent
    function_args: tuple[Any, ...]  # What the function received
    function_kwargs: dict[str, Any]  # Keyword arguments
    function_output: Any  # What the function returned
```

**Usage example**:
```python
async def on_post_invoke(self, context: PostInvokeContext) -> Any:
    # Audit trail: see complete execution
    logger.info(f"User sent: {context.original_args}")
    logger.info(f"Function received: {context.function_args}")
    logger.info(f"Function returned: {context.function_output}")

    # Transform output if needed
    return sanitize(context.function_output)
```

## Advanced Configuration

### Multiple Policies

Execute policies in order:

```yaml
function_policies:
  validator:
    _type: input_validation
  rate_limiter:
    _type: rate_limiting
  sanitizer:
    _type: output_sanitization

middleware:
  my_middleware:
    _type: dynamic_middleware
    pre_invoke_policy: ["validator", "rate_limiter"]  # Order matters
    post_invoke_policy: ["sanitizer"]
```

### Enabling and Disabling Policies

All policies have an `enabled` configuration field:

```yaml
function_policies:
  debug_validator:
    _type: input_validation
    enabled: false  # Disabled - won't execute
    max_length: 500
```

### Combining with Dynamic Middleware

Use `dynamic_middleware` to control which functions get policies:

```yaml
middleware:
  security_middleware:
    _type: dynamic_middleware

    # Apply to specific components
    llms: ["gpt4"]
    embedders: ["text-embedding"]

    # Or use auto-discovery
    register_workflow_functions: true
    register_llms: true

    # Policies apply to all discovered functions
    pre_invoke_policy: ["input_validator"]
    post_invoke_policy: ["output_sanitizer"]
```

## Testing Your Policy

### Unit Testing

Test policies independently:

```python
import pytest
from unittest.mock import Mock
from nat.function_policy.interface import PreInvokeContext
from nat.function_policy.logging.logging_policy import LoggingPolicy
from nat.function_policy.logging.logging_policy_config import LoggingPolicyConfig


@pytest.mark.asyncio
async def test_logging_policy():
    """Test that logging policy observes without modifying."""
    config = LoggingPolicyConfig(log_level="DEBUG")
    policy = LoggingPolicy(config=config)

    # Create mock context
    mock_context = Mock(spec=FunctionMiddlewareContext)
    mock_context.name = "test_function"

    context = PreInvokeContext(
        function_context=mock_context,
        original_args=({"value": 10},),
        function_args=({"value": 10},),
        function_kwargs={}
    )

    # Execute policy
    result = await policy.on_pre_invoke(context)

    # Verify result is None (no modification)
    assert result is None
```

### Integration Testing

Test policies with actual workflows:

```python
@pytest.mark.asyncio
async def test_policy_with_workflow():
    """Test policy integration with a workflow."""
    # Build workflow with policies
    builder = WorkflowBuilder()
    await builder.add_function_policy("logger", LoggingPolicyConfig())
    # ... build workflow ...

    # Run workflow and verify policy executed
    result = await workflow.ainvoke("test input")
    assert result is not None
```

## Best Practices

### Policy Design

**Keep policies focused**: Each policy should do one thing well.

**Make policies composable**: Design policies to work together in any order.

**Handle None returns**: Return `None` to preserve the current value unchanged.

**Use context for inspection only**: Don't try to modify context objects - return transformed values instead.

### Error Handling

**Raise exceptions to block**: If a policy needs to stop execution, raise an exception:

```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    if context.function_args and not is_valid(context.function_args[0]):
        raise ValueError("Invalid input")
    return context.function_args
```

**Log errors appropriately**: Use structured logging for debugging:

```python
try:
    result = transform(context.function_input)
    return result
except Exception as e:
    logger.error(f"Transform failed: {e}", exc_info=True)
    raise
```

### Performance Considerations

**Keep policies lightweight**: Policies execute on every function call - avoid expensive operations.

**Use async operations**: Policies are async - you can use `await` for I/O operations:

```python
async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
    # Async validation against external service
    if context.function_args:
        is_valid = await validate_with_service(context.function_args[0])
        if not is_valid:
            raise ValidationError("Input rejected by service")
    return context.function_args
```

**Consider caching**: For expensive operations, cache results when possible.

## CLI Commands

### List Available Policies

```bash
nat info function-policies
```

List with detailed information:

```bash
nat info function-policies --verbose
```

### Show Policy Details

```bash
nat info function-policy <policy_name>
```

Example:

```bash
nat info function-policy logging
```

## Relationship to Middleware

Function policies work with middleware to intercept function calls:

- **Dynamic Middleware** (`dynamic_middleware`) discovers and wraps functions
- **Function Policies** define what happens at interception points
- **Middleware applies policies** to discovered functions

```
┌─────────────────────────────────────────┐
│         Dynamic Middleware              │
│                                         │
│  1. Discovers functions                 │
│  2. Wraps them with policy hooks        │
│  3. Executes policies on each call      │
└─────────────────────────────────────────┘
                   │
                   │ applies
                   ↓
┌─────────────────────────────────────────┐
│         Function Policies               │
│                                         │
│  • on_pre_invoke: Transform inputs      │
│  • on_post_invoke: Transform outputs    │
└─────────────────────────────────────────┘
```

See the [Middleware](middleware.md) documentation for more details on middleware configuration.

## Common Use Cases

### Security Monitoring

```python
class SecurityMonitorPolicy(FunctionPolicyBase[SecurityMonitorPolicyConfig]):
    """Monitor function calls for security threats."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
        # Check for suspicious patterns in first arg
        if context.function_args and contains_sql_injection(context.function_args[0]):
            logger.warning(f"SQL injection attempt in {context.function_context.name}")
            raise SecurityException("Blocked suspicious input")
        return context.function_args

    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        # Check for data leaks in output
        if contains_pii(context.function_output):
            logger.warning(f"PII detected in output from {context.function_context.name}")
            return redact_pii(context.function_output)
        return context.function_output
```

### Input Sanitization

```python
class InputSanitizationPolicy(FunctionPolicyBase[InputSanitizationPolicyConfig]):
    """Sanitize inputs before function execution."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
        if not context.function_args:
            return context.function_args

        # Remove dangerous characters from first arg
        sanitized = sanitize_input(context.function_args[0])

        if sanitized != context.function_args[0]:
            logger.info(f"Sanitized input for {context.function_context.name}")

        return (sanitized,) + context.function_args[1:]
```

### Performance Monitoring

```python
class PerformanceMonitorPolicy(FunctionPolicyBase[PerformanceMonitorPolicyConfig]):
    """Monitor function execution time."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
        # Store start time (you'd use a more sophisticated approach in practice)
        self._start_times[context.function_context.name] = time.time()
        return context.function_args

    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        # Calculate and log execution time
        elapsed = time.time() - self._start_times.get(context.function_context.name, 0)
        logger.info(f"{context.function_context.name} took {elapsed:.2f}s")
        return context.function_output
```

## Troubleshooting

### Policy Not Executing

**Check policy is enabled**:
```yaml
function_policies:
  my_policy:
    enabled: true  # Make sure this is true
```

**Check middleware configuration**:
```yaml
middleware:
  my_middleware:
    pre_invoke_policy: ["my_policy"]  # Policy must be referenced
```

### Policy Execution Order Issues

Remember that policies execute in the order you list them:

```yaml
pre_invoke_policy: ["first", "second", "third"]  # This order
```

Each policy receives the output from the previous policy.

### Context Is None or Missing Fields

Make sure you're using the correct context type:
- `PreInvokeContext` for `on_pre_invoke`
- `PostInvokeContext` for `on_post_invoke`

The context is always provided by the middleware - you don't create it yourself.
