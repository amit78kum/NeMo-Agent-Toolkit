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
  input_logger:
    _type: input_logging
    enabled: true
    log_level: INFO

  output_sanitizer:
    _type: output_sanitization
    enabled: true
    sanitize_pii: true

# Apply policies via middleware
middleware:
  security_middleware:
    _type: dynamic_middleware
    register_workflow_functions: true
    pre_invoke_policy: ["input_logger"]
    post_invoke_policy: ["output_sanitizer"]

# Middleware applies to workflow functions
workflow:
  _type: react_agent
  middleware: ["security_middleware"]
  # Other workflow config...
```

## Creating Custom Function Policies

### Step 1: Define the Configuration

Create a configuration class inheriting from `FunctionPolicyBaseConfig`:

```python
from pydantic import Field
from nat.data_models.function_policy import FunctionPolicyBaseConfig


class InputLoggingPolicyConfig(FunctionPolicyBaseConfig, name="input_logging"):
    """Configuration for input logging policy."""

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    include_function_name: bool = Field(
        default=True,
        description="Whether to log function names"
    )
```

### Step 2: Implement the Policy Class

Create the policy class inheriting from `FunctionPolicyBase`:

```python
from nat.function_policy.interface import FunctionPolicyBase
from nat.function_policy.interface import PreInvokeContext, PostInvokeContext
from typing import Any
import logging

logger = logging.getLogger(__name__)


class InputLoggingPolicy(FunctionPolicyBase[InputLoggingPolicyConfig]):
    """Policy that logs function inputs and outputs."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
        """Log input before function executes."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        if self.config.include_function_name:
            logger.log(log_level, f"Calling {context.function_context.name}")

        logger.log(log_level, f"Input: {context.function_input}")

        # Return input unchanged (or return modified input)
        return context.function_input

    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        """Log output after function executes."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        if self.config.include_function_name:
            logger.log(log_level, f"Finished {context.function_context.name}")

        logger.log(log_level, f"Output: {context.function_output}")

        # Return output unchanged (or return modified output)
        return context.function_output
```

### Step 3: Register the Policy

Use the `@register_function_policy` decorator to register your policy:

```python
from nat.cli.register_workflow import register_function_policy
from nat.builder.builder import Builder


@register_function_policy(config_type=InputLoggingPolicyConfig)
async def input_logging(config: InputLoggingPolicyConfig, builder: Builder):
    """Build an input logging policy from configuration."""
    yield InputLoggingPolicy(config=config)
```

## Policy Execution Model

### Pre-Invoke Pipeline

When a function is called, pre-invoke policies execute in order:

```
Original Input (10)
    ↓
Policy 1: Add 5 → Returns 15
    ↓
Policy 2: Multiply by 2 → Returns 30
    ↓
Function receives: 30
```

Each policy receives the output from the previous policy via `context.function_input`. The `context.original_input` always contains the original value the user provided.

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

Each policy receives the output from the previous policy as the second argument. The `PostInvokeContext` provides a complete audit trail with `original_input`, `function_input`, and `function_output`.

### Return Value Behavior

**Transforming Values**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    # Modify the input
    return context.function_input + 1
```

**Preserving Values**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    # Log or validate without modifying
    validate(context.function_input)
    return context.function_input  # Or return None to preserve
```

**Blocking Execution**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    if is_malicious(context.function_input):
        raise SecurityException("Blocked malicious input")
    return context.function_input
```

## Working with Context Objects

### PreInvokeContext

Contains the current input state:

```python
@dataclass
class PreInvokeContext:
    function_context: FunctionMiddlewareContext  # Function metadata
    original_input: Any  # What the user sent
    function_input: Any  # Current value (after previous policies)
```

**Usage example**:
```python
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    # Check if input was already modified
    if context.function_input != context.original_input:
        logger.info("Input was modified by previous policy")

    # Get function name for logging
    logger.info(f"Processing {context.function_context.name}")

    return transform(context.function_input)
```

### PostInvokeContext

Contains the complete execution state (frozen/immutable):

```python
@dataclass(frozen=True)
class PostInvokeContext:
    function_context: FunctionMiddlewareContext  # Function metadata
    original_input: Any  # What the user sent
    function_input: Any  # What the function received
    function_output: Any  # What the function returned
```

**Usage example**:
```python
async def on_post_invoke(self, context: PostInvokeContext) -> Any:
    # Audit trail: see complete execution
    logger.info(f"User sent: {context.original_input}")
    logger.info(f"Function received: {context.function_input}")
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
  logger:
    _type: input_logging
  sanitizer:
    _type: output_sanitization

middleware:
  my_middleware:
    _type: dynamic_middleware
    pre_invoke_policy: ["validator", "logger"]  # Order matters
    post_invoke_policy: ["sanitizer"]
```

### Enabling and Disabling Policies

All policies have an `enabled` configuration field:

```yaml
function_policies:
  debug_logger:
    _type: input_logging
    enabled: false  # Disabled - won't execute
    log_level: DEBUG
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


@pytest.mark.asyncio
async def test_input_logging_policy():
    """Test that policy logs input correctly."""
    config = InputLoggingPolicyConfig(log_level="INFO")
    policy = InputLoggingPolicy(config=config)

    # Create mock context
    mock_context = Mock(spec=FunctionMiddlewareContext)
    mock_context.name = "test_function"

    context = PreInvokeContext(
        function_context=mock_context,
        original_input={"value": 10},
        function_input={"value": 10}
    )

    # Execute policy
    result = await policy.on_pre_invoke(context)

    # Verify result
    assert result == {"value": 10}
```

### Integration Testing

Test policies with actual workflows:

```python
@pytest.mark.asyncio
async def test_policy_with_workflow():
    """Test policy integration with a workflow."""
    # Build workflow with policies
    builder = WorkflowBuilder()
    await builder.add_function_policy("logger", InputLoggingPolicyConfig())
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
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    if not is_valid(context.function_input):
        raise ValueError("Invalid input")
    return context.function_input
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
async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
    # Async validation against external service
    is_valid = await validate_with_service(context.function_input)
    if not is_valid:
        raise ValidationError("Input rejected by service")
    return context.function_input
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
nat info function-policy input_logging
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

    async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
        # Check for suspicious patterns
        if contains_sql_injection(context.function_input):
            logger.warning(f"SQL injection attempt in {context.function_context.name}")
            raise SecurityException("Blocked suspicious input")
        return context.function_input

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

    async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
        # Remove dangerous characters
        sanitized = sanitize_input(context.function_input)

        if sanitized != context.function_input:
            logger.info(f"Sanitized input for {context.function_context.name}")

        return sanitized
```

### Performance Monitoring

```python
class PerformanceMonitorPolicy(FunctionPolicyBase[PerformanceMonitorPolicyConfig]):
    """Monitor function execution time."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
        # Store start time (you'd use a more sophisticated approach in practice)
        self._start_times[context.function_context.name] = time.time()
        return context.function_input

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
