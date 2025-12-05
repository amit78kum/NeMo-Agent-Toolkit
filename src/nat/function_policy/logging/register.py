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
"""Registration for logging function policy."""

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function_policy
from nat.function_policy.logging.logging_policy import LoggingPolicy
from nat.function_policy.logging.logging_policy_config import LoggingPolicyConfig


@register_function_policy(config_type=LoggingPolicyConfig)
async def logging_policy(config: LoggingPolicyConfig, builder: Builder):
    """Build a logging policy from configuration.

    Args:
        config: The logging policy configuration
        builder: The workflow builder (unused but required by component pattern)

    Yields:
        A configured logging policy instance
    """
    yield LoggingPolicy(config=config)
