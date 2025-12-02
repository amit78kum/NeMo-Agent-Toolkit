# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from nat.builder.function import Function
from nat.data_models.component import ComponentGroup
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.middleware import DynamicMiddlewareConfig
from nat.middleware.dynamic_function_middleware import DiscoveredComponent
from nat.middleware.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.dynamic_function_middleware import WorkflowInventory

# ==================== Test Fixtures ====================


@pytest.fixture
def mock_builder():
    """Create a mock builder with all required methods."""
    builder = Mock()
    builder._functions = {}
    builder._function_policies = {}
    builder.get_llm = AsyncMock()
    builder.get_embedder = AsyncMock()
    builder.get_retriever = AsyncMock()
    builder.get_memory_client = AsyncMock()
    builder.get_object_store_client = AsyncMock()
    builder.get_auth_provider = AsyncMock()
    builder.get_function = AsyncMock()
    builder.get_function_config = Mock()
    return builder


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with typical methods."""
    client = Mock()
    client.config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1024}
    client.description = "OpenAI GPT-4 LLM"

    # Allowlisted async methods (should be wrapped)
    client.invoke = AsyncMock(return_value="response")
    client.ainvoke = AsyncMock(return_value="response")
    client.stream = AsyncMock()
    client.astream = AsyncMock()
    client.generate = AsyncMock(return_value="generated text")
    client.agenerate = AsyncMock(return_value="generated text")

    # Configuration methods (should be excluded from wrapping)
    client.bind = Mock(return_value=client)
    client.with_config = Mock(return_value=client)
    client.dict = Mock(return_value={})

    return client


@pytest.fixture
def mock_embedder_client():
    """Create a mock embedder client with typical methods."""
    client = Mock()
    client.config = {"model": "text-embedding-ada-002", "dimensions": 1536}
    client.description = "OpenAI text embedder"

    # Embedder methods
    client.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    client.embed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    client.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    return client


@pytest.fixture
def mock_retriever_client():
    """Create a mock retriever client with comprehensive methods."""
    client = Mock()
    client.config = {"top_k": 5, "threshold": 0.7}
    client.description = "Vector store retriever"

    # Retriever methods
    client.retrieve = AsyncMock(return_value=[{"doc": "result1"}, {"doc": "result2"}])
    client.aretrieve = AsyncMock(return_value=[{"doc": "result1"}, {"doc": "result2"}])
    client.get_relevant_documents = AsyncMock(return_value=[{"doc": "result"}])
    client.aget_relevant_documents = AsyncMock(return_value=[{"doc": "result"}])

    return client


@pytest.fixture
def mock_memory_client():
    """Create a mock memory client with comprehensive methods."""
    client = Mock()
    client.config = {"backend": "redis", "ttl": 3600}
    client.description = "Redis memory provider"

    # Memory methods
    client.add = AsyncMock()
    client.get = AsyncMock(return_value={"key": "value"})
    client.search = AsyncMock(return_value=[{"result": "data"}])
    client.delete = AsyncMock()
    client.clear = AsyncMock()

    return client


@pytest.fixture
def mock_object_store_client():
    """Create a mock object store client with comprehensive methods."""
    client = Mock()
    client.config = {"bucket": "my-bucket", "region": "us-west-2"}
    client.description = "S3 object store"

    # Object store methods
    client.put = AsyncMock()
    client.get = AsyncMock(return_value=b"file_content")
    client.delete = AsyncMock()
    client.list = AsyncMock(return_value=["file1", "file2"])
    client.exists = AsyncMock(return_value=True)

    return client


@pytest.fixture
def mock_auth_provider_client():
    """Create a mock auth provider client with comprehensive methods."""
    client = Mock()
    client.config = {"provider_type": "oauth2", "token_url": "https://auth.example.com/token"}
    client.description = "OAuth2 authentication provider"

    # Auth methods
    client.authenticate = AsyncMock(return_value={"token": "abc123", "expires_in": 3600})
    client.refresh_token = AsyncMock(return_value={"token": "def456", "expires_in": 3600})
    client.validate_token = AsyncMock(return_value=True)

    return client


@pytest.fixture
def mock_function():
    """Create a mock NAT Function instance."""
    func = Mock(spec=Function)
    func.instance_name = "test_function"
    func.config = FunctionBaseConfig()
    func.middleware = []
    func.configure_middleware = Mock()
    return func


# ==================== Test Policy Fixtures ====================


def create_tracking_policy(name: str,
                           execution_tracker: list,
                           pre_invoke_modifier=None,
                           post_invoke_modifier=None,
                           pre_invoke_error=None,
                           post_invoke_error=None):
    """Factory function to create test policies with tracking and optional modifications/errors.

    Args:
        name: Policy name
        execution_tracker: List to track execution order
        pre_invoke_modifier: Function to modify input (receives context.function_input, returns modified value)
        post_invoke_modifier: Function to modify output (receives context.function_output, returns modified value)
        pre_invoke_error: Exception to raise in pre_invoke (for testing error handling)
        post_invoke_error: Exception to raise in post_invoke (for testing error handling)
    """
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.function_policy.interface import FunctionPolicyBase
    from nat.function_policy.interface import PostInvokeContext
    from nat.function_policy.interface import PreInvokeContext

    class TrackingPolicy(FunctionPolicyBase[FunctionPolicyBaseConfig]):

        async def on_pre_invoke(self, context: PreInvokeContext):
            execution_tracker.append(f"{name}_pre")
            if pre_invoke_error:
                raise pre_invoke_error
            if pre_invoke_modifier:
                return pre_invoke_modifier(context.function_input)
            return context.function_input

        async def on_post_invoke(self, context: PostInvokeContext):
            execution_tracker.append(f"{name}_post")
            if post_invoke_error:
                raise post_invoke_error
            if post_invoke_modifier:
                return post_invoke_modifier(context.function_output)
            return context.function_output

    return TrackingPolicy


# ==================== Test DynamicMiddlewareBase Initialization ====================


def test_dynamic_middleware_initialization(mock_builder):
    """Test DynamicMiddlewareBase initializes correctly."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)

    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._config == config
    assert middleware._builder == mock_builder
    assert isinstance(middleware._workflow_inventory, WorkflowInventory)
    assert isinstance(middleware._registered_callables, set)


def test_dynamic_middleware_patches_builder_methods(mock_builder):
    """Test that builder get_* methods are patched during initialization."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)

    original_get_llm = mock_builder.get_llm
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Builder methods should be replaced
    assert mock_builder.get_llm != original_get_llm
    assert middleware._builder_get_llm == original_get_llm


def test_dynamic_middleware_missing_builder_method():
    """Test that missing builder methods raise RuntimeError."""
    mock_builder = Mock()
    # Missing get_llm method
    del mock_builder.get_llm
    config = DynamicMiddlewareConfig()

    with pytest.raises(RuntimeError, match="Builder does not have 'get_llm' method"):
        DynamicFunctionMiddleware(config=config, builder=mock_builder)


# ==================== Middleware Pattern Compliance Tests ====================


async def test_middleware_implements_invoke(mock_builder):
    """Test that DynamicFunctionMiddleware implements function_middleware_invoke."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert hasattr(middleware, 'function_middleware_invoke')
    assert callable(middleware.function_middleware_invoke)


async def test_middleware_implements_stream(mock_builder):
    """Test that DynamicFunctionMiddleware implements function_middleware_stream."""
    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert hasattr(middleware, 'function_middleware_stream')
    assert callable(middleware.function_middleware_stream)


async def test_middleware_invoke_calls_next_with_no_policies(mock_builder):
    """Test that invoke delegates to call_next when no policies are configured."""
    from nat.middleware.middleware import FunctionMiddlewareContext

    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    assert result == expected_output


async def test_middleware_invoke_executes_pre_invoke_policies(mock_builder):
    """Test that pre_invoke policies are executed in order."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.function_policy.interface import FunctionPolicyBase
    from nat.function_policy.interface import PostInvokeContext
    from nat.function_policy.interface import PreInvokeContext
    from nat.middleware.middleware import FunctionMiddlewareContext

    # Create a test policy that modifies input
    class TestPolicy(FunctionPolicyBase[FunctionPolicyBaseConfig]):

        async def on_pre_invoke(self, context: PreInvokeContext):
            # Add a marker to show this policy executed
            if isinstance(context.function_input, dict):
                context.function_input["policy_executed"] = True
            return context.function_input

        async def on_post_invoke(self, context: PostInvokeContext):
            return context.function_output

    policy = TestPolicy(config=FunctionPolicyBaseConfig(), name="test_policy")

    config = DynamicMiddlewareConfig(
        register_workflow_functions=False,
        pre_invoke_policy=[],
    )
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [policy]

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        # Verify policy modified the input
        assert value.get("policy_executed") is True
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    assert result == expected_output


async def test_middleware_invoke_executes_post_invoke_policies(mock_builder):
    """Test that post_invoke policies are executed in order."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.function_policy.interface import FunctionPolicyBase
    from nat.function_policy.interface import PostInvokeContext
    from nat.function_policy.interface import PreInvokeContext
    from nat.middleware.middleware import FunctionMiddlewareContext

    # Create a test policy that modifies output
    class TestPolicy(FunctionPolicyBase[FunctionPolicyBaseConfig]):

        async def on_pre_invoke(self, context: PreInvokeContext):
            return context.function_input

        async def on_post_invoke(self, context: PostInvokeContext):
            # Add a marker to show this policy executed
            if isinstance(context.function_output, dict):
                context.function_output["policy_executed"] = True
            return context.function_output

    policy = TestPolicy(config=FunctionPolicyBaseConfig(), name="test_policy")

    config = DynamicMiddlewareConfig(
        register_workflow_functions=False,
        post_invoke_policy=[],
    )
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._post_invoke_policies = [policy]

    test_input = {"value": "test"}
    function_output = {"result": "success"}

    async def mock_call_next(value):
        return function_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    # Verify policy modified the output
    assert result.get("policy_executed") is True


async def test_middleware_invoke_policy_execution_order(mock_builder):
    """Test that multiple policies execute in the correct order."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.function_policy.interface import FunctionPolicyBase
    from nat.function_policy.interface import PostInvokeContext
    from nat.function_policy.interface import PreInvokeContext
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_order = []

    # Create test policies that track execution order
    class Policy1(FunctionPolicyBase[FunctionPolicyBaseConfig]):

        async def on_pre_invoke(self, context: PreInvokeContext):
            execution_order.append("policy1_pre")
            return context.function_input

        async def on_post_invoke(self, context: PostInvokeContext):
            execution_order.append("policy1_post")
            return context.function_output

    class Policy2(FunctionPolicyBase[FunctionPolicyBaseConfig]):

        async def on_pre_invoke(self, context: PreInvokeContext):
            execution_order.append("policy2_pre")
            return context.function_input

        async def on_post_invoke(self, context: PostInvokeContext):
            execution_order.append("policy2_post")
            return context.function_output

    policy1 = Policy1(config=FunctionPolicyBaseConfig(), name="policy1")
    policy2 = Policy2(config=FunctionPolicyBaseConfig(), name="policy2")

    config = DynamicMiddlewareConfig(
        register_workflow_functions=False,
        pre_invoke_policy=[],
        post_invoke_policy=[],
    )
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [policy1, policy2]
    middleware._post_invoke_policies = [policy1, policy2]

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        execution_order.append("function_call")
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    # Verify execution order: pre1, pre2, function, post1, post2
    assert execution_order == ["policy1_pre", "policy2_pre", "function_call", "policy1_post", "policy2_post"]
    assert result == expected_output


async def test_middleware_stream_calls_next_with_no_policies(mock_builder):
    """Test that stream delegates to call_next when no policies are configured."""
    from nat.middleware.middleware import FunctionMiddlewareContext

    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    test_input = {"value": "test"}

    async def mock_call_next(value):
        yield "chunk1"
        yield "chunk2"
        yield "chunk3"

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    chunks = []
    async for chunk in middleware.function_middleware_stream(test_input, mock_call_next, context):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2", "chunk3"]


# ==================== Policy Error Handling Tests ====================


async def test_middleware_skips_failing_pre_invoke_policy(mock_builder):
    """Test that middleware skips a failing pre-invoke policy and continues execution."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_tracker = []
    FailingPolicy = create_tracking_policy("failing",
                                           execution_tracker,
                                           pre_invoke_error=ValueError("Policy intentionally failed"))
    failing_policy = FailingPolicy(config=FunctionPolicyBaseConfig(), name="failing_policy")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, pre_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [failing_policy]

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        # Should still be called with original input since policy failed
        assert value == test_input
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    # Should complete successfully despite policy failure
    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)
    assert result == expected_output


async def test_middleware_skips_failing_post_invoke_policy(mock_builder):
    """Test that middleware skips a failing post-invoke policy and continues execution."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_tracker = []
    FailingPolicy = create_tracking_policy("failing",
                                           execution_tracker,
                                           post_invoke_error=ValueError("Policy intentionally failed in post-invoke"))
    failing_policy = FailingPolicy(config=FunctionPolicyBaseConfig(), name="failing_policy")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, post_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._post_invoke_policies = [failing_policy]

    test_input = {"value": "test"}
    function_output = {"result": "success"}

    async def mock_call_next(value):
        return function_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    # Should return original output since policy failed
    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)
    assert result == function_output


async def test_middleware_continues_with_working_policies_after_failure(mock_builder):
    """Test that working policies still execute after a failing policy."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_order = []

    # Create policies using factory
    FailingPolicy = create_tracking_policy("failing_policy",
                                           execution_order,
                                           pre_invoke_error=ValueError("Policy intentionally failed"))
    WorkingPolicy1 = create_tracking_policy("working_policy_1",
                                            execution_order,
                                            pre_invoke_modifier=lambda inp: {
                                                **inp, "policy1_executed": True
                                            })
    WorkingPolicy2 = create_tracking_policy("working_policy_2",
                                            execution_order,
                                            pre_invoke_modifier=lambda inp: {
                                                **inp, "policy2_executed": True
                                            })

    failing = FailingPolicy(config=FunctionPolicyBaseConfig(), name="failing")
    working1 = WorkingPolicy1(config=FunctionPolicyBaseConfig(), name="working1")
    working2 = WorkingPolicy2(config=FunctionPolicyBaseConfig(), name="working2")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, pre_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    # Put failing policy in the middle
    middleware._pre_invoke_policies = [working1, failing, working2]

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        # Both working policies should have executed
        assert value.get("policy1_executed") is True
        assert value.get("policy2_executed") is True
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    # Verify execution order
    assert execution_order == ["working_policy_1_pre", "failing_policy_pre", "working_policy_2_pre"]
    assert result == expected_output


async def test_middleware_stream_skips_failing_pre_invoke_policy(mock_builder):
    """Test that streaming skips failing pre-invoke policy and continues."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_tracker = []
    FailingPolicy = create_tracking_policy("failing",
                                           execution_tracker,
                                           pre_invoke_error=ValueError("Policy intentionally failed"))
    failing_policy = FailingPolicy(config=FunctionPolicyBaseConfig(), name="failing_policy")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, pre_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [failing_policy]

    test_input = {"value": "test"}

    async def mock_call_next(value):
        # Should still be called with original input
        assert value == test_input
        yield "chunk1"
        yield "chunk2"

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    # Should stream successfully despite policy failure
    chunks = []
    async for chunk in middleware.function_middleware_stream(test_input, mock_call_next, context):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]


async def test_middleware_stream_skips_failing_post_invoke_policy(mock_builder):
    """Test that streaming skips failing post-invoke policy for chunks."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_tracker = []
    FailingPolicy = create_tracking_policy("failing",
                                           execution_tracker,
                                           post_invoke_error=ValueError("Policy intentionally failed for chunk"))
    failing_policy = FailingPolicy(config=FunctionPolicyBaseConfig(), name="failing_policy")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, post_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._post_invoke_policies = [failing_policy]

    test_input = {"value": "test"}

    async def mock_call_next(value):
        yield "chunk1"
        yield "chunk2"

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    # Should return original chunks since policy fails
    chunks = []
    async for chunk in middleware.function_middleware_stream(test_input, mock_call_next, context):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]


async def test_middleware_skips_disabled_pre_invoke_policy(mock_builder):
    """Test that pre-invoke policies with enabled=False are skipped."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_order = []

    # Create policies using factory
    DisabledPolicy = create_tracking_policy("disabled_policy",
                                            execution_order,
                                            pre_invoke_modifier=lambda inp: {
                                                **inp, "disabled_executed": True
                                            })
    EnabledPolicy = create_tracking_policy("enabled_policy",
                                           execution_order,
                                           pre_invoke_modifier=lambda inp: {
                                               **inp, "enabled_executed": True
                                           })

    # Create disabled policy with enabled=False
    disabled = DisabledPolicy(config=FunctionPolicyBaseConfig(enabled=False), name="disabled")
    enabled = EnabledPolicy(config=FunctionPolicyBaseConfig(enabled=True), name="enabled")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, pre_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [disabled, enabled]

    test_input = {"value": "test"}
    expected_output = {"result": "success"}

    async def mock_call_next(value):
        # Verify disabled policy didn't execute, but enabled did
        assert value.get("disabled_executed") is None
        assert value.get("enabled_executed") is True
        return expected_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    # Verify disabled policy was never called
    assert execution_order == ["enabled_policy_pre"]
    assert result == expected_output


async def test_middleware_skips_disabled_post_invoke_policy(mock_builder):
    """Test that post-invoke policies with enabled=False are skipped."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_order = []

    # Create policies using factory
    DisabledPolicy = create_tracking_policy("disabled_policy",
                                            execution_order,
                                            post_invoke_modifier=lambda out: {
                                                **out, "disabled_executed": True
                                            })
    EnabledPolicy = create_tracking_policy("enabled_policy",
                                           execution_order,
                                           post_invoke_modifier=lambda out: {
                                               **out, "enabled_executed": True
                                           })

    # Create disabled policy with enabled=False
    disabled = DisabledPolicy(config=FunctionPolicyBaseConfig(enabled=False), name="disabled")
    enabled = EnabledPolicy(config=FunctionPolicyBaseConfig(enabled=True), name="enabled")

    config = DynamicMiddlewareConfig(register_workflow_functions=False, post_invoke_policy=[])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._post_invoke_policies = [disabled, enabled]

    test_input = {"value": "test"}
    function_output = {"result": "success"}

    async def mock_call_next(value):
        return function_output

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    result = await middleware.function_middleware_invoke(test_input, mock_call_next, context)

    # Verify disabled policy was never called, but enabled was
    assert execution_order == ["enabled_policy_post"]
    assert result.get("disabled_executed") is None
    assert result.get("enabled_executed") is True


async def test_middleware_skips_disabled_streaming_policies(mock_builder):
    """Test that disabled policies are skipped in streaming mode."""
    from nat.data_models.function_policy import FunctionPolicyBaseConfig
    from nat.middleware.middleware import FunctionMiddlewareContext

    execution_order = []

    # Create policies using factory
    DisabledPrePolicy = create_tracking_policy("disabled_pre",
                                               execution_order,
                                               pre_invoke_modifier=lambda inp: {"should_not_see": "this"})
    DisabledPostPolicy = create_tracking_policy("disabled_post",
                                                execution_order,
                                                post_invoke_modifier=lambda out: "MODIFIED")

    disabled_pre = DisabledPrePolicy(config=FunctionPolicyBaseConfig(enabled=False), name="disabled_pre")
    disabled_post = DisabledPostPolicy(config=FunctionPolicyBaseConfig(enabled=False), name="disabled_post")

    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)
    middleware._pre_invoke_policies = [disabled_pre]
    middleware._post_invoke_policies = [disabled_post]

    test_input = {"value": "test"}

    async def mock_call_next(value):
        # Should receive original input, not modified by disabled pre policy
        assert value == test_input
        yield "chunk1"
        yield "chunk2"

    context = FunctionMiddlewareContext(
        name="test_function",
        config={},
        description="Test function",
        input_schema=None,
        single_output_schema=None,
        stream_output_schema=None,
    )

    # Should return original chunks, not modified by disabled post policy
    chunks = []
    async for chunk in middleware.function_middleware_stream(test_input, mock_call_next, context):
        chunks.append(chunk)

    # Verify disabled policies never executed
    assert execution_order == []
    assert chunks == ["chunk1", "chunk2"]


# ==================== LLM Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,llm_name,expected_name",
    [
        # Explicit LLM reference
        ({
            "llms": ["gpt4"]
        }, "gpt4", "gpt4"),
        # Auto-discovery enabled
        ({
            "register_llms": True
        }, "claude", "claude"),
        # Multiple explicit LLM references (testing first one)
        ({
            "llms": ["gpt4", "claude"]
        }, "gpt4", "gpt4"),
    ])
async def test_discover_and_register_llm(mock_builder, mock_llm_client, config_dict, llm_name, expected_name):
    """Test LLM discovery and registration with various configuration patterns."""

    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_llm = AsyncMock(return_value=mock_llm_client)

    # Mock _get_callable_functions to return expected LLM methods
    expected_methods = {"invoke", "ainvoke", "stream", "astream"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_llm(llm_name, "langchain")

    # Assertions
    assert result == mock_llm_client
    assert len(middleware._workflow_inventory.llms) == 1

    discovered = middleware._workflow_inventory.llms[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_llm_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_llm_skip_if_not_configured(mock_builder, mock_llm_client):
    """Test LLM is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_llms=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_llm = AsyncMock(return_value=mock_llm_client)
    result = await middleware._discover_and_register_llm("test_llm", "langchain")

    assert result == mock_llm_client
    assert len(middleware._workflow_inventory.llms) == 0


async def test_discover_and_register_llm_skip_duplicates(mock_builder, mock_llm_client):
    """Test that duplicate LLMs are not registered twice."""
    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_llm = AsyncMock(return_value=mock_llm_client)

    # Mock _get_callable_functions to return expected LLM methods
    expected_methods = {"invoke", "stream"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_llm("test_llm", "langchain")
    await middleware._discover_and_register_llm("test_llm", "langchain")

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.llms) == 1


def test_should_intercept_llm_with_explicit_config(mock_builder):
    """Test LLM interception logic with explicit configuration."""
    from nat.data_models.component_ref import LLMRef

    config = DynamicMiddlewareConfig(llms=[LLMRef("gpt4")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_llm("gpt4") is True
    assert middleware._should_intercept_llm("other_llm") is False


def test_should_intercept_llm_with_auto_discovery(mock_builder):
    """Test LLM interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_llm("any_llm") is True


def test_should_intercept_llm_prevents_duplicate(mock_builder, mock_llm_client):
    """Test that LLM interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add LLM to inventory manually
    callable_functions = {"invoke", "stream"}
    middleware._workflow_inventory.llms.append(
        DiscoveredComponent(name="existing_llm",
                            component_type=ComponentGroup.LLMS,
                            instance=mock_llm_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_llm.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_llm("existing_llm") is False

    # Verify it's tracked in _registered_callables
    assert "existing_llm.invoke" in middleware._registered_callables
    assert "existing_llm.stream" in middleware._registered_callables


# ==================== Embedder Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,embedder_name,expected_name",
    [
        # Explicit embedder reference
        ({
            "embedders": ["text-embedding"]
        }, "text-embedding", "text-embedding"),
        # Auto-discovery enabled
        ({
            "register_embedders": True
        }, "nv-embed", "nv-embed"),
        # Multiple explicit embedder references (testing first one)
        ({
            "embedders": ["text-embedding", "nv-embed"]
        }, "text-embedding", "text-embedding"),
    ])
async def test_discover_and_register_embedder(mock_builder,
                                              mock_embedder_client,
                                              config_dict,
                                              embedder_name,
                                              expected_name):
    """Test embedder discovery and registration with various configuration patterns."""
    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_embedder = AsyncMock(return_value=mock_embedder_client)

    # Mock _get_callable_functions to return expected embedder methods
    expected_methods = {"embed_query", "embed_documents"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_embedder(embedder_name, "langchain")

    # Assertions
    assert result == mock_embedder_client
    assert len(middleware._workflow_inventory.embedders) == 1

    discovered = middleware._workflow_inventory.embedders[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_embedder_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_embedder_skip_if_not_configured(mock_builder, mock_embedder_client):
    """Test embedder is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_embedders=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_embedder = AsyncMock(return_value=mock_embedder_client)
    result = await middleware._discover_and_register_embedder("test_embedder", "langchain")

    assert result == mock_embedder_client
    assert len(middleware._workflow_inventory.embedders) == 0


async def test_discover_and_register_embedder_skip_duplicates(mock_builder, mock_embedder_client):
    """Test that duplicate embedders are not registered twice."""
    config = DynamicMiddlewareConfig(register_embedders=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_embedder = AsyncMock(return_value=mock_embedder_client)

    # Mock _get_callable_functions to return expected embedder methods
    expected_methods = {"embed_query", "embed_documents"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_embedder("test_embedder", "langchain")
    await middleware._discover_and_register_embedder("test_embedder", "langchain")

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.embedders) == 1


def test_should_intercept_embedder_with_explicit_config(mock_builder):
    """Test embedder interception logic with explicit configuration."""
    from nat.data_models.component_ref import EmbedderRef

    config = DynamicMiddlewareConfig(embedders=[EmbedderRef("text-embedding")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_embedder("text-embedding") is True
    assert middleware._should_intercept_embedder("other_embedder") is False


def test_should_intercept_embedder_with_auto_discovery(mock_builder):
    """Test embedder interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_embedders=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_embedder("any_embedder") is True


def test_should_intercept_embedder_prevents_duplicate(mock_builder, mock_embedder_client):
    """Test that embedder interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_embedders=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add embedder to inventory manually
    callable_functions = {"embed_query", "embed_documents"}
    middleware._workflow_inventory.embedders.append(
        DiscoveredComponent(name="existing_embedder",
                            component_type=ComponentGroup.EMBEDDERS,
                            instance=mock_embedder_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_embedder.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_embedder("existing_embedder") is False

    # Verify it's tracked in _registered_callables
    assert "existing_embedder.embed_query" in middleware._registered_callables
    assert "existing_embedder.embed_documents" in middleware._registered_callables


# ==================== Retriever Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,retriever_name,expected_name",
    [
        # Explicit retriever reference
        ({
            "retrievers": ["milvus"]
        }, "milvus", "milvus"),
        # Auto-discovery enabled
        ({
            "register_retrievers": True
        }, "tavily", "tavily"),
        # Multiple explicit retriever references (testing first one)
        ({
            "retrievers": ["milvus", "tavily"]
        }, "milvus", "milvus"),
    ])
async def test_discover_and_register_retriever(mock_builder,
                                               mock_retriever_client,
                                               config_dict,
                                               retriever_name,
                                               expected_name):
    """Test retriever discovery and registration with various configuration patterns."""
    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_retriever = AsyncMock(return_value=mock_retriever_client)

    # Mock _get_callable_functions to return expected retriever methods
    expected_methods = {"retrieve", "get_relevant_documents"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_retriever(retriever_name, None)

    # Assertions
    assert result == mock_retriever_client
    assert len(middleware._workflow_inventory.retrievers) == 1

    discovered = middleware._workflow_inventory.retrievers[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_retriever_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_retriever_skip_if_not_configured(mock_builder, mock_retriever_client):
    """Test retriever is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_retrievers=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_retriever = AsyncMock(return_value=mock_retriever_client)
    result = await middleware._discover_and_register_retriever("test_retriever", None)

    assert result == mock_retriever_client
    assert len(middleware._workflow_inventory.retrievers) == 0


async def test_discover_and_register_retriever_skip_duplicates(mock_builder, mock_retriever_client):
    """Test that duplicate retrievers are not registered twice."""
    config = DynamicMiddlewareConfig(register_retrievers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_retriever = AsyncMock(return_value=mock_retriever_client)

    # Mock _get_callable_functions to return expected retriever methods
    expected_methods = {"retrieve", "get_relevant_documents"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_retriever("test_retriever", None)
    await middleware._discover_and_register_retriever("test_retriever", None)

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.retrievers) == 1


def test_should_intercept_retriever_with_explicit_config(mock_builder):
    """Test retriever interception logic with explicit configuration."""
    from nat.data_models.component_ref import RetrieverRef

    config = DynamicMiddlewareConfig(retrievers=[RetrieverRef("milvus")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_retriever("milvus") is True
    assert middleware._should_intercept_retriever("other_retriever") is False


def test_should_intercept_retriever_with_auto_discovery(mock_builder):
    """Test retriever interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_retrievers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_retriever("any_retriever") is True


def test_should_intercept_retriever_prevents_duplicate(mock_builder, mock_retriever_client):
    """Test that retriever interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_retrievers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add retriever to inventory manually
    callable_functions = {"retrieve", "get_relevant_documents"}
    middleware._workflow_inventory.retrievers.append(
        DiscoveredComponent(name="existing_retriever",
                            component_type=ComponentGroup.RETRIEVERS,
                            instance=mock_retriever_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_retriever.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_retriever("existing_retriever") is False

    # Verify it's tracked in _registered_callables
    assert "existing_retriever.retrieve" in middleware._registered_callables
    assert "existing_retriever.get_relevant_documents" in middleware._registered_callables


# ==================== Memory Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,memory_name,expected_name",
    [
        # Explicit memory reference
        ({
            "memory": ["redis_memory"]
        }, "redis_memory", "redis_memory"),
        # Auto-discovery enabled
        ({
            "register_memory": True
        }, "postgres_memory", "postgres_memory"),
        # Multiple explicit memory references (testing first one)
        ({
            "memory": ["redis_memory", "postgres_memory"]
        }, "redis_memory", "redis_memory"),
    ])
async def test_discover_and_register_memory(mock_builder, mock_memory_client, config_dict, memory_name, expected_name):
    """Test memory provider discovery and registration with various configuration patterns."""
    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_memory = AsyncMock(return_value=mock_memory_client)

    # Mock _get_callable_functions to return expected memory methods
    expected_methods = {"add", "get", "search"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_memory(memory_name)

    # Assertions
    assert result == mock_memory_client
    assert len(middleware._workflow_inventory.memory) == 1

    discovered = middleware._workflow_inventory.memory[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_memory_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_memory_skip_if_not_configured(mock_builder, mock_memory_client):
    """Test memory provider is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_memory=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_memory = AsyncMock(return_value=mock_memory_client)
    result = await middleware._discover_and_register_memory("test_memory")

    assert result == mock_memory_client
    assert len(middleware._workflow_inventory.memory) == 0


async def test_discover_and_register_memory_skip_duplicates(mock_builder, mock_memory_client):
    """Test that duplicate memory providers are not registered twice."""
    config = DynamicMiddlewareConfig(register_memory=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_memory = AsyncMock(return_value=mock_memory_client)

    # Mock _get_callable_functions to return expected memory methods
    expected_methods = {"add", "get", "search"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_memory("test_memory")
    await middleware._discover_and_register_memory("test_memory")

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.memory) == 1


def test_should_intercept_memory_with_explicit_config(mock_builder):
    """Test memory provider interception logic with explicit configuration."""
    from nat.data_models.component_ref import MemoryRef

    config = DynamicMiddlewareConfig(memory=[MemoryRef("redis_memory")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_memory("redis_memory") is True
    assert middleware._should_intercept_memory("other_memory") is False


def test_should_intercept_memory_with_auto_discovery(mock_builder):
    """Test memory provider interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_memory=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_memory("any_memory") is True


def test_should_intercept_memory_prevents_duplicate(mock_builder, mock_memory_client):
    """Test that memory provider interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_memory=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add memory provider to inventory manually
    callable_functions = {"add", "get", "search"}
    middleware._workflow_inventory.memory.append(
        DiscoveredComponent(name="existing_memory",
                            component_type=ComponentGroup.MEMORY,
                            instance=mock_memory_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_memory.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_memory("existing_memory") is False

    # Verify it's tracked in _registered_callables
    assert "existing_memory.add" in middleware._registered_callables
    assert "existing_memory.get" in middleware._registered_callables
    assert "existing_memory.search" in middleware._registered_callables


# ==================== Object Store Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,store_name,expected_name",
    [
        # Explicit object store reference
        ({
            "object_stores": ["s3_store"]
        }, "s3_store", "s3_store"),
        # Auto-discovery enabled
        ({
            "register_object_stores": True
        }, "azure_store", "azure_store"),
        # Multiple explicit object store references (testing first one)
        ({
            "object_stores": ["s3_store", "azure_store"]
        }, "s3_store", "s3_store"),
    ])
async def test_discover_and_register_object_store(mock_builder,
                                                  mock_object_store_client,
                                                  config_dict,
                                                  store_name,
                                                  expected_name):
    """Test object store discovery and registration with various configuration patterns."""
    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_object_store = AsyncMock(return_value=mock_object_store_client)

    # Mock _get_callable_functions to return expected object store methods
    expected_methods = {"put", "get", "list"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_object_store(store_name)

    # Assertions
    assert result == mock_object_store_client
    assert len(middleware._workflow_inventory.object_stores) == 1

    discovered = middleware._workflow_inventory.object_stores[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_object_store_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_object_store_skip_if_not_configured(mock_builder, mock_object_store_client):
    """Test object store is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_object_stores=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_object_store = AsyncMock(return_value=mock_object_store_client)
    result = await middleware._discover_and_register_object_store("test_store")

    assert result == mock_object_store_client
    assert len(middleware._workflow_inventory.object_stores) == 0


async def test_discover_and_register_object_store_skip_duplicates(mock_builder, mock_object_store_client):
    """Test that duplicate object stores are not registered twice."""
    config = DynamicMiddlewareConfig(register_object_stores=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_object_store = AsyncMock(return_value=mock_object_store_client)

    # Mock _get_callable_functions to return expected object store methods
    expected_methods = {"put", "get", "list"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_object_store("test_store")
    await middleware._discover_and_register_object_store("test_store")

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.object_stores) == 1


def test_should_intercept_object_store_with_explicit_config(mock_builder):
    """Test object store interception logic with explicit configuration."""
    from nat.data_models.component_ref import ObjectStoreRef

    config = DynamicMiddlewareConfig(object_stores=[ObjectStoreRef("s3_store")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_object_store("s3_store") is True
    assert middleware._should_intercept_object_store("other_store") is False


def test_should_intercept_object_store_with_auto_discovery(mock_builder):
    """Test object store interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_object_stores=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_object_store("any_store") is True


def test_should_intercept_object_store_prevents_duplicate(mock_builder, mock_object_store_client):
    """Test that object store interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_object_stores=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add object store to inventory manually
    callable_functions = {"put", "get", "list"}
    middleware._workflow_inventory.object_stores.append(
        DiscoveredComponent(name="existing_store",
                            component_type=ComponentGroup.OBJECT_STORES,
                            instance=mock_object_store_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_store.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_object_store("existing_store") is False

    # Verify it's tracked in _registered_callables
    assert "existing_store.put" in middleware._registered_callables
    assert "existing_store.get" in middleware._registered_callables
    assert "existing_store.list" in middleware._registered_callables


# ==================== Auth Provider Component Tests ====================


@pytest.mark.parametrize(
    "config_dict,auth_name,expected_name",
    [
        # Explicit auth provider reference
        ({
            "auth_providers": ["oauth2"]
        }, "oauth2", "oauth2"),
        # Auto-discovery enabled
        ({
            "register_auth_providers": True
        }, "api_key", "api_key"),
        # Multiple explicit auth provider references (testing first one)
        ({
            "auth_providers": ["oauth2", "api_key"]
        }, "oauth2", "oauth2"),
    ])
async def test_discover_and_register_auth_provider(mock_builder,
                                                   mock_auth_provider_client,
                                                   config_dict,
                                                   auth_name,
                                                   expected_name):
    """Test auth provider discovery and registration with various configuration patterns."""
    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Setup builder to return mock client
    middleware._builder_get_auth_provider = AsyncMock(return_value=mock_auth_provider_client)

    # Mock _get_callable_functions to return expected auth provider methods
    expected_methods = {"authenticate", "validate_token"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Call discovery
    result = await middleware._discover_and_register_auth_provider(auth_name)

    # Assertions
    assert result == mock_auth_provider_client
    assert len(middleware._workflow_inventory.auth_providers) == 1

    discovered = middleware._workflow_inventory.auth_providers[0]
    assert discovered.name == expected_name
    assert discovered.instance == mock_auth_provider_client
    # Verify callable methods match what we mocked
    assert discovered.callable_functions == expected_methods
    assert len(discovered.callable_functions) > 0


async def test_discover_and_register_auth_provider_skip_if_not_configured(mock_builder, mock_auth_provider_client):
    """Test auth provider is not registered if not configured for interception."""
    config = DynamicMiddlewareConfig(register_auth_providers=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_auth_provider = AsyncMock(return_value=mock_auth_provider_client)
    result = await middleware._discover_and_register_auth_provider("test_auth")

    assert result == mock_auth_provider_client
    assert len(middleware._workflow_inventory.auth_providers) == 0


async def test_discover_and_register_auth_provider_skip_duplicates(mock_builder, mock_auth_provider_client):
    """Test that duplicate auth providers are not registered twice."""
    config = DynamicMiddlewareConfig(register_auth_providers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    middleware._builder_get_auth_provider = AsyncMock(return_value=mock_auth_provider_client)

    # Mock _get_callable_functions to return expected auth provider methods
    expected_methods = {"authenticate", "validate_token"}
    middleware._get_callable_functions = Mock(return_value=expected_methods)

    # Register twice
    await middleware._discover_and_register_auth_provider("test_auth")
    await middleware._discover_and_register_auth_provider("test_auth")

    # Should only be registered once in inventory
    assert len(middleware._workflow_inventory.auth_providers) == 1


def test_should_intercept_auth_provider_with_explicit_config(mock_builder):
    """Test auth provider interception logic with explicit configuration."""
    from nat.data_models.component_ref import AuthenticationRef

    config = DynamicMiddlewareConfig(auth_providers=[AuthenticationRef("oauth2")])
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_auth_provider("oauth2") is True
    assert middleware._should_intercept_auth_provider("other_auth") is False


def test_should_intercept_auth_provider_with_auto_discovery(mock_builder):
    """Test auth provider interception logic with auto-discovery enabled."""
    config = DynamicMiddlewareConfig(register_auth_providers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert middleware._should_intercept_auth_provider("any_auth") is True


def test_should_intercept_auth_provider_prevents_duplicate(mock_builder, mock_auth_provider_client):
    """Test that auth provider interception prevents duplicates."""
    config = DynamicMiddlewareConfig(register_auth_providers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Add auth provider to inventory manually
    callable_functions = {"authenticate", "validate_token"}
    middleware._workflow_inventory.auth_providers.append(
        DiscoveredComponent(name="existing_auth",
                            component_type=ComponentGroup.AUTHENTICATION,
                            instance=mock_auth_provider_client,
                            callable_functions=callable_functions))

    # Add entries to _registered_callables to track registration
    for method in callable_functions:
        middleware._registered_callables.add(f"existing_auth.{method}")

    # Should return False because already in inventory
    assert middleware._should_intercept_auth_provider("existing_auth") is False

    # Verify it's tracked in _registered_callables
    assert "existing_auth.authenticate" in middleware._registered_callables
    assert "existing_auth.validate_token" in middleware._registered_callables


# ==================== Workflow Function Tests ====================


@pytest.mark.parametrize("config_dict,function_names,expected_count",
                         [
                             ({
                                 "register_workflow_functions": True
                             }, ["calc_function"], 1),
                             ({
                                 "register_workflow_functions": True
                             }, ["func1", "func2"], 2),
                             ({
                                 "register_workflow_functions": True
                             }, ["api_handler", "data_processor", "validator"], 3),
                         ])
def test_discover_functions_from_builder(mock_builder, mock_function, config_dict, function_names, expected_count):
    """Test function discovery from builder._functions with various configurations."""
    # Add functions to builder
    mock_builder._functions = {
        name: Mock(config=FunctionBaseConfig(), instance=mock_function)
        for name in function_names
    }

    config = DynamicMiddlewareConfig(**config_dict)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Assertions
    assert len(middleware._workflow_inventory.workflow_functions) == expected_count

    discovered_names = [f.name for f in middleware._workflow_inventory.workflow_functions]
    for name in function_names:
        assert name in discovered_names

    # Verify each discovered function
    for discovered_func in middleware._workflow_inventory.workflow_functions:
        assert discovered_func.name in function_names
        assert discovered_func.instance == mock_function
        assert isinstance(discovered_func.config, FunctionBaseConfig)


def test_discover_functions_skip_if_not_configured(mock_builder, mock_function):
    """Test functions are not discovered if register_workflow_functions is False."""
    mock_builder._functions = {"test_func": Mock(config=FunctionBaseConfig(), instance=mock_function)}

    config = DynamicMiddlewareConfig(register_workflow_functions=False)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    assert len(middleware._workflow_inventory.workflow_functions) == 0


def test_discover_functions_skip_duplicates(mock_builder, mock_function):
    """Test duplicate functions are not added to inventory multiple times."""
    # Python dicts have unique keys, so this tests the dict behavior
    mock_builder._functions = {
        "func1": Mock(config=FunctionBaseConfig(), instance=mock_function),
    }

    config = DynamicMiddlewareConfig(register_workflow_functions=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Should only have one entry
    assert len(middleware._workflow_inventory.workflow_functions) == 1
    assert middleware._workflow_inventory.workflow_functions[0].name == "func1"


# ==================== Allowlist Enforcement Tests ====================


async def test_llm_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted LLM functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock LLM with both allowlisted and non-allowlisted functions
    mock_llm = Mock()
    mock_llm.config = {"model": "gpt-4"}
    mock_llm.description = "Test LLM"

    # Allowlisted functions (from COMPONENT_FUNCTION_ALLOWLISTS)
    mock_llm.generate = AsyncMock()
    mock_llm.agenerate = AsyncMock()
    mock_llm.predict = AsyncMock()

    # Non-allowlisted functions (should NOT be registered)
    mock_llm.get_token_count = Mock(return_value=10)  # Not in allowlist
    mock_llm.validate_config = Mock()  # Not in allowlist
    mock_llm.bind = Mock()  # Configuration method, not in allowlist

    config = DynamicMiddlewareConfig(register_llms=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'generate', 'agenerate', 'predict', 'get_token_count', 'validate_config', 'bind'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_llm = AsyncMock(return_value=mock_llm)
    await middleware._discover_and_register_llm("test_llm", None)

    # Get the allowlist for LLMs
    llm_allowlist = COMPONENT_FUNCTION_ALLOWLISTS[ComponentGroup.LLMS]

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['generate', 'agenerate', 'predict']:
        assert f"test_llm.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['get_token_count', 'validate_config', 'bind']:
        assert f"test_llm.{func_name}" not in registered


async def test_embedder_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted embedder functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock embedder with both allowlisted and non-allowlisted functions
    mock_embedder = Mock()
    mock_embedder.config = {"model": "text-embedding-ada-002"}
    mock_embedder.description = "Test Embedder"

    # Allowlisted functions
    mock_embedder.embed_query = AsyncMock()
    mock_embedder.aembed_query = AsyncMock()
    mock_embedder.embed_documents = AsyncMock()

    # Non-allowlisted functions
    mock_embedder.get_dimensions = Mock(return_value=1536)
    mock_embedder.model_dump = Mock()

    config = DynamicMiddlewareConfig(register_embedders=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'embed_query', 'aembed_query', 'embed_documents', 'get_dimensions', 'model_dump'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_embedder = AsyncMock(return_value=mock_embedder)
    await middleware._discover_and_register_embedder("test_embedder", "langchain")

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['embed_query', 'aembed_query', 'embed_documents']:
        assert f"test_embedder.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['get_dimensions', 'model_dump']:
        assert f"test_embedder.{func_name}" not in registered


async def test_retriever_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted retriever functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock retriever with both allowlisted and non-allowlisted functions
    mock_retriever = Mock()
    mock_retriever.config = {"top_k": 5}
    mock_retriever.description = "Test Retriever"

    # Allowlisted functions
    mock_retriever.retrieve = AsyncMock()
    mock_retriever.aretrieve = AsyncMock()
    mock_retriever.search = AsyncMock()

    # Non-allowlisted functions
    mock_retriever.add_documents = AsyncMock()
    mock_retriever.reset_index = AsyncMock()

    config = DynamicMiddlewareConfig(register_retrievers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'retrieve', 'aretrieve', 'search', 'add_documents', 'reset_index'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_retriever = AsyncMock(return_value=mock_retriever)
    await middleware._discover_and_register_retriever("test_retriever", None)

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['retrieve', 'aretrieve', 'search']:
        assert f"test_retriever.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['add_documents', 'reset_index']:
        assert f"test_retriever.{func_name}" not in registered


async def test_memory_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted memory functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock memory with both allowlisted and non-allowlisted functions
    mock_memory = Mock()
    mock_memory.config = {"max_history": 100}
    mock_memory.description = "Test Memory"

    # Allowlisted functions
    mock_memory.save_context = AsyncMock()
    mock_memory.asave_context = AsyncMock()
    mock_memory.clear = AsyncMock()

    # Non-allowlisted functions
    mock_memory.get_stats = Mock()
    mock_memory.export_history = AsyncMock()

    config = DynamicMiddlewareConfig(register_memory=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'save_context', 'asave_context', 'clear', 'get_stats', 'export_history'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_memory = AsyncMock(return_value=mock_memory)
    await middleware._discover_and_register_memory("test_memory")

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['save_context', 'asave_context', 'clear']:
        assert f"test_memory.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['get_stats', 'export_history']:
        assert f"test_memory.{func_name}" not in registered


async def test_object_store_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted object store functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock object store with both allowlisted and non-allowlisted functions
    mock_store = Mock()
    mock_store.config = {"bucket": "test-bucket"}
    mock_store.description = "Test Object Store"

    # Allowlisted functions
    mock_store.get = AsyncMock()
    mock_store.put = AsyncMock()
    mock_store.delete = AsyncMock()

    # Non-allowlisted functions
    mock_store.connect = AsyncMock()
    mock_store.close = AsyncMock()

    config = DynamicMiddlewareConfig(register_object_stores=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'get', 'put', 'delete', 'connect', 'close'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_object_store = AsyncMock(return_value=mock_store)
    await middleware._discover_and_register_object_store("test_store")

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['get', 'put', 'delete']:
        assert f"test_store.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['connect', 'close']:
        assert f"test_store.{func_name}" not in registered


async def test_auth_provider_only_registers_allowlisted_functions(mock_builder):
    """Test that only allowlisted auth provider functions are registered."""
    from nat.data_models.component import ComponentGroup
    from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

    # Create a mock auth provider with both allowlisted and non-allowlisted functions
    mock_auth = Mock()
    mock_auth.config = {"provider": "oauth2"}
    mock_auth.description = "Test Auth Provider"

    # Allowlisted functions
    mock_auth.authenticate = AsyncMock()
    mock_auth.verify = AsyncMock()
    mock_auth.validate_token = AsyncMock()

    # Non-allowlisted functions
    mock_auth.refresh_token = AsyncMock()
    mock_auth.revoke_token = AsyncMock()

    config = DynamicMiddlewareConfig(register_auth_providers=True)
    middleware = DynamicFunctionMiddleware(config=config, builder=mock_builder)

    # Mock _get_callable_functions to return all functions
    with_mock = Mock(return_value={'authenticate', 'verify', 'validate_token', 'refresh_token', 'revoke_token'})
    middleware._get_callable_functions = with_mock

    # Discover and register
    mock_builder._builder_get_auth_provider = AsyncMock(return_value=mock_auth)
    await middleware._discover_and_register_auth_provider("test_auth")

    # Verify only allowlisted functions are registered
    registered = middleware._registered_callables
    for func_name in ['authenticate', 'verify', 'validate_token']:
        assert f"test_auth.{func_name}" in registered

    # Verify non-allowlisted functions are NOT registered
    for func_name in ['refresh_token', 'revoke_token']:
        assert f"test_auth.{func_name}" not in registered


# ==================== Method Registration Tests ====================


def test_register_function_prevents_duplicates(mock_function):
    """Test that duplicate function registration is prevented."""
    from nat.data_models.function import FunctionBaseConfig
    from nat.middleware.utils.workflow_inventory import DiscoveredFunction

    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    # Create a proper DiscoveredFunction
    discovered = DiscoveredFunction(name="test_function", config=FunctionBaseConfig(), instance=mock_function)

    # Register once
    middleware._register_function(discovered)
    call_count_1 = discovered.instance.configure_middleware.call_count

    # Try to register again
    middleware._register_function(discovered)
    call_count_2 = discovered.instance.configure_middleware.call_count

    # Should log but not call configure_middleware again (already registered)
    assert call_count_1 == call_count_2


def test_register_function_error_handling(mock_function):
    """Test error handling when registering a function that's already registered."""
    from nat.data_models.function import FunctionBaseConfig
    from nat.middleware.utils.workflow_inventory import DiscoveredFunction

    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    # Create a proper DiscoveredFunction
    discovered = DiscoveredFunction(name="test_func", config=FunctionBaseConfig(), instance=mock_function)

    # Register once
    middleware._register_function(discovered)

    # Manually mark as registered
    middleware._registered_callables.add(discovered.instance.instance_name)

    # Try to register again - should handle gracefully (log and skip)
    middleware._register_function(discovered)

    # Verify configure_middleware was called only once (not twice)
    assert discovered.instance.configure_middleware.call_count == 1


def test_register_component_function_prevents_duplicates(mock_llm_client):
    """Test that duplicate component function registration is prevented."""
    from nat.middleware.utils.workflow_inventory import DiscoveredComponent

    config = DynamicMiddlewareConfig()
    middleware = DynamicFunctionMiddleware(config=config, builder=Mock(_functions={}))

    # Create a DiscoveredComponent
    discovered = DiscoveredComponent(name="gpt4",
                                     component_type=ComponentGroup.LLMS,
                                     instance=mock_llm_client,
                                     callable_functions={"invoke"})

    # Manually add to registered to simulate first registration
    middleware._registered_callables.add("gpt4.invoke")

    # Try to register again (should log and skip wrapping)
    middleware._register_component_function(discovered, "invoke")

    # Should still be tracked (once)
    assert "gpt4.invoke" in middleware._registered_callables
    assert list(middleware._registered_callables).count("gpt4.invoke") == 1
