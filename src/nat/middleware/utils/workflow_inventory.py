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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.function import Function
from nat.data_models.component import ComponentGroup
from nat.data_models.function import FunctionBaseConfig

# Default component function allowlists for auto-registration
COMPONENT_FUNCTION_ALLOWLISTS: dict[ComponentGroup, set[str]] = {
    ComponentGroup.LLMS: {
        'generate',
        'agenerate',
        'predict',
        'apredict',
        'call',
        'acall',
        'complete',
        'acomplete',
        'generate_stream',
        'agenerate_stream',
    },
    ComponentGroup.EMBEDDERS: {
        'embed_query',
        'aembed_query',
        'embed_documents',
        'aembed_documents',
        'embed',
        'aembed',
        'embed_text',
        'aembed_text',
    },
    ComponentGroup.RETRIEVERS: {
        'get_relevant_documents',
        'aget_relevant_documents',
        'retrieve',
        'aretrieve',
        'search',
        'asearch',
    },
    ComponentGroup.MEMORY: {
        'load_memory_variables',
        'aload_memory_variables',
        'save_context',
        'asave_context',
        'clear',
        'aclear',
    },
    ComponentGroup.OBJECT_STORES: {
        'get',
        'aget',
        'put',
        'aput',
        'delete',
        'adelete',
        'list',
        'alist',
        'exists',
        'aexists',
    },
    ComponentGroup.AUTHENTICATION: {
        'authenticate',
        'aauthenticate',
        'verify',
        'averify',
        'validate_token',
        'avalidate_token',
    },
}


class DiscoveredComponent(BaseModel):
    """Information about a discovered component and its available functions.

    Attributes:
        name: Component name (e.g., "gpt4", "milvus")
        component_type: Component type
        instance: Component instance
        config: Component configuration
        callable_functions: A set of callable component function names on the instance
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Component name")
    component_type: ComponentGroup = Field(
        description="Component group (llms, embedders, retrievers, memory, object_stores, authentication)")
    instance: Any = Field(description="Component instance")
    config: Any = Field(description="Component configuration", default=None)
    callable_functions: set[str] = Field(description="Set of callable component function names on the instance",
                                         default_factory=set)


class DiscoveredFunction(BaseModel):
    """Information about a discovered workflow function.

    Attributes:
        name: Function name (e.g., "my_api_handler")
        config: Function configuration
        instance: Function instance
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Function name")
    config: FunctionBaseConfig = Field(description="Function configuration")
    instance: Function = Field(description="Function instance")


class WorkflowInventory(BaseModel):
    """Inventory of discovered components and functions.

    This container holds all components and functions discovered from the workflow that
    are available for registration but not explicitly configured in the middleware config.
    It provides a structured view of everything that can be intercepted.
    """

    llms: list[DiscoveredComponent] = Field(
        default_factory=list, description="Discovered LLM components and their functions available for registration")

    embedders: list[DiscoveredComponent] = Field(
        default_factory=list,
        description="Discovered Embedder components and their functions available for registration")

    retrievers: list[DiscoveredComponent] = Field(
        default_factory=list,
        description="Discovered Retriever components and their functions available for registration")

    memory: list[DiscoveredComponent] = Field(
        default_factory=list, description="Discovered Memory components and their functions available for registration")

    object_stores: list[DiscoveredComponent] = Field(
        default_factory=list,
        description="Discovered Object Store components and their functions available for registration")

    auth_providers: list[DiscoveredComponent] = Field(
        default_factory=list,
        description="Discovered Authentication components and their functions available for registration")

    workflow_functions: list[DiscoveredFunction] = Field(
        default_factory=list, description="Discovered workflow functions available for registration")
