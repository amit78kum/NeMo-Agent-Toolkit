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

import typing

from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class FunctionPolicyBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for function policies.

    Function policies inject custom logic before/after function execution - modifying inputs,
    outputs, or monitoring behavior.
    """

    enabled: bool = Field(default=True, description="Enable or disable this policy")


FunctionPolicyBaseConfigT = typing.TypeVar("FunctionPolicyBaseConfigT", bound=FunctionPolicyBaseConfig)
