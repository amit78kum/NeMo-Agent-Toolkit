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
"""Tests for the function policy interface and context objects."""

from dataclasses import FrozenInstanceError

import pytest

from nat.data_models.function_policy import FunctionPolicyBaseConfig
from nat.function_policy.interface import FunctionPolicyBase
from nat.function_policy.interface import PostInvokeContext
from nat.function_policy.interface import PreInvokeContext
from nat.middleware.middleware import FunctionMiddlewareContext


class SamplePolicyConfig(FunctionPolicyBaseConfig, name="sample_policy"):
    """Sample configuration for function policies."""
    pass


class SamplePolicy(FunctionPolicyBase[SamplePolicyConfig]):
    """Sample policy implementation for testing."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> any:
        return context.function_input

    async def on_post_invoke(self, context: PostInvokeContext) -> any:
        return context.function_output


@pytest.fixture
def mock_function_context():
    """Create a mock FunctionMiddlewareContext for testing."""
    from unittest.mock import Mock
    context = Mock(spec=FunctionMiddlewareContext)
    context.name = "test_function"
    context.description = "Test function"
    return context


# ==================== Test PreInvokeContext ====================


def test_pre_invoke_context_creation(mock_function_context):
    """Test creating a PreInvokeContext."""
    original_input = {"key": "value"}
    function_input = {"key": "modified"}

    context = PreInvokeContext(function_context=mock_function_context,
                               original_input=original_input,
                               function_input=function_input)

    assert context.function_context == mock_function_context
    assert context.original_input == original_input
    assert context.function_input == function_input


def test_pre_invoke_context_function_input_mutable(mock_function_context):
    """Test that PreInvokeContext.function_input can be modified."""
    original_input = {"key": "value"}
    context = PreInvokeContext(function_context=mock_function_context,
                               original_input=original_input,
                               function_input=original_input)

    # Middleware should be able to update function_input
    new_input = {"key": "modified"}
    context.function_input = new_input
    assert context.function_input == new_input
    assert context.original_input == original_input  # Should remain unchanged


def test_pre_invoke_context_preserves_original_input(mock_function_context):
    """Test that original_input is preserved during modifications."""
    original_input = {"key": "value"}
    context = PreInvokeContext(function_context=mock_function_context,
                               original_input=original_input,
                               function_input=original_input)

    # Modify function_input
    context.function_input = {"key": "modified"}

    # Original should be unchanged
    assert context.original_input == {"key": "value"}
    assert context.function_input == {"key": "modified"}


# ==================== Test PostInvokeContext ====================


def test_post_invoke_context_creation(mock_function_context):
    """Test creating a PostInvokeContext."""
    original_input = {"key": "original"}
    function_input = {"key": "modified"}
    function_output = {"result": "success"}

    context = PostInvokeContext(function_context=mock_function_context,
                                original_input=original_input,
                                function_input=function_input,
                                function_output=function_output)

    assert context.function_context == mock_function_context
    assert context.original_input == original_input
    assert context.function_input == function_input
    assert context.function_output == function_output


def test_post_invoke_context_is_frozen(mock_function_context):
    """Test that PostInvokeContext is immutable (frozen)."""
    context = PostInvokeContext(function_context=mock_function_context,
                                original_input={"key": "value"},
                                function_input={"key": "modified"},
                                function_output={"result": "success"})

    # Should not be able to modify any field
    with pytest.raises(FrozenInstanceError):
        context.original_input = {"new": "value"}

    with pytest.raises(FrozenInstanceError):
        context.function_input = {"new": "value"}

    with pytest.raises(FrozenInstanceError):
        context.function_output = {"new": "result"}


def test_post_invoke_context_provides_full_audit_trail(mock_function_context):
    """Test that PostInvokeContext contains complete audit trail."""
    original = {"user": "input"}
    modified = {"user": "sanitized_input"}
    output = {"result": "success"}

    context = PostInvokeContext(function_context=mock_function_context,
                                original_input=original,
                                function_input=modified,
                                function_output=output)

    # All values should be accessible for audit
    assert context.original_input == original
    assert context.function_input == modified
    assert context.function_output == output


# ==================== Test FunctionPolicyBase ====================


async def test_policy_base_initialization():
    """Test FunctionPolicyBase initialization."""
    config = SamplePolicyConfig()
    policy = SamplePolicy(config=config)

    # Builder sets name after creation
    policy.name = "test_policy_instance"

    assert policy.config == config
    assert policy.name == "test_policy_instance"


async def test_policy_base_has_required_methods():
    """Test that FunctionPolicyBase defines required abstract methods."""
    config = SamplePolicyConfig()
    policy = SamplePolicy(config=config)

    # Should have on_pre_invoke method
    assert hasattr(policy, "on_pre_invoke")
    assert callable(policy.on_pre_invoke)

    # Should have on_post_invoke method
    assert hasattr(policy, "on_post_invoke")
    assert callable(policy.on_post_invoke)


async def test_policy_can_access_context_in_pre_invoke(mock_function_context):
    """Test that policy can read PreInvokeContext in on_pre_invoke."""
    config = SamplePolicyConfig()
    policy = SamplePolicy(config=config)

    context = PreInvokeContext(function_context=mock_function_context,
                               original_input={"value": 10},
                               function_input={"value": 10})

    result = await policy.on_pre_invoke(context)
    assert result == {"value": 10}


async def test_policy_can_access_context_in_post_invoke(mock_function_context):
    """Test that policy can read PostInvokeContext in on_post_invoke."""
    config = SamplePolicyConfig()
    policy = SamplePolicy(config=config)

    context = PostInvokeContext(function_context=mock_function_context,
                                original_input={"value": 10},
                                function_input={"value": 11},
                                function_output={"result": 21})

    result = await policy.on_post_invoke(context)
    assert result == {"result": 21}


async def test_policy_enabled_config():
    """Test that policy respects enabled configuration."""
    config = SamplePolicyConfig(enabled=True)
    policy = SamplePolicy(config=config)
    assert policy.config.enabled is True

    config_disabled = SamplePolicyConfig(enabled=False)
    policy_disabled = SamplePolicy(config=config_disabled)
    assert policy_disabled.config.enabled is False
