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
"""Logging function policy for observing function inputs and outputs."""

import logging
from typing import Any

from nat.function_policy.interface import FunctionPolicyBase
from nat.function_policy.interface import PostInvokeContext
from nat.function_policy.interface import PreInvokeContext
from nat.function_policy.logging.logging_policy_config import LoggingPolicyConfig

logger = logging.getLogger(__name__)


class LoggingPolicy(FunctionPolicyBase[LoggingPolicyConfig]):
    """Policy that logs function inputs and outputs without modification."""

    async def on_pre_invoke(self, context: PreInvokeContext) -> tuple[Any, ...] | None:
        """Log pre-invoke context.

        Args:
            context: Pre-invoke context with function args

        Returns:
            None (args unchanged)
        """
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.log(
            level,
            "[%s] pre_invoke: function=%s, args=%s",
            self.name,
            context.function_context.name,
            self._truncate(context.function_args),
        )
        return None

    async def on_post_invoke(self, context: PostInvokeContext) -> Any:
        """Log post-invoke context.

        Args:
            context: Post-invoke context with function output

        Returns:
            None (output unchanged)
        """
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.log(
            level,
            "[%s] post_invoke: function=%s, output=%s",
            self.name,
            context.function_context.name,
            self._truncate(context.function_output),
        )
        return None

    def _truncate(self, value: Any) -> str:
        """Truncate value string representation if too long."""
        s = repr(value)
        if len(s) > self.config.max_value_length:
            return s[:self.config.max_value_length] + "..."
        return s
