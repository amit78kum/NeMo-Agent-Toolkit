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

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TypeVar

from nat.middleware.middleware import FunctionMiddlewareContext

PolicyConfigT = TypeVar("PolicyConfigT")


@dataclass
class PreInvokeContext:
    """Context for pre-invoke policy execution.

    Contains the current state of inputs before function execution. Middleware
    updates function_input after each policy runs.

    Attributes:
        function_context: Metadata about the function being intercepted
        original_input: The original input from the user
        function_input: The value that will be passed to the function
    """
    function_context: FunctionMiddlewareContext
    original_input: Any
    function_input: Any


@dataclass(frozen=True)
class PostInvokeContext:
    """Context for post-invoke policy execution.

    Contains the complete execution state after function runs.

    Attributes:
        function_context: Metadata about the function being intercepted
        original_input: The original input from the user
        function_input: The input that was sent to the function
        function_output: The output returned by the function
    """
    function_context: FunctionMiddlewareContext
    original_input: Any
    function_input: Any
    function_output: Any


class FunctionPolicyBase(ABC, Generic[PolicyConfigT]):
    """Base class for function policies.

    Function policies intercept function calls via on_pre_invoke and on_post_invoke hooks.
    Policies receive context objects for inspection and return transformed values.
    Multiple policies execute in sequence, with each policy receiving the output of the previous.

    Type Parameters:
        PolicyConfigT: The configuration type for this policy

    Attributes:
        config: Policy configuration
        name: Registered policy name
    """

    def __init__(self, config: PolicyConfigT, name: str | None = None):
        """Initialize the policy.

        Args:
            config: Policy configuration
            name: Registered name for this policy
        """
        self.config = config
        self.name = name

    @abstractmethod
    async def on_pre_invoke(self, context: PreInvokeContext) -> Any:
        """Called before function execution.

        Receives the current input state via context. Returns the input value
        to pass forward. Returning None preserves the current input unchanged.

        Args:
            context: Pre-invoke execution context

        Returns:
            Transformed input value, or None to preserve current value


        """
        pass

    @abstractmethod
    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        """Called after function execution to transform output.

        Receives the complete execution state via context. Returns the output
        value to pass forward. Returning None preserves the current output unchanged.

        Args:
            context: Post-invoke execution context

        Returns:
            Transformed output value, or None to preserve current value


        """
        pass


__all__ = [
    "FunctionPolicyBase",
    "PreInvokeContext",
    "PostInvokeContext",
    "PolicyConfigT",
]
