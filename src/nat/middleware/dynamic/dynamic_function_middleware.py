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

import inspect
import logging
from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import FunctionRef
from nat.function_policy.interface import FunctionPolicyBase
from nat.function_policy.interface import PostInvokeContext
from nat.function_policy.interface import PreInvokeContext
from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareChain
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS
from nat.middleware.utils.workflow_inventory import DiscoveredComponent
from nat.middleware.utils.workflow_inventory import DiscoveredFunction
from nat.middleware.utils.workflow_inventory import WorkflowInventory

logger = logging.getLogger(__name__)


class DynamicFunctionMiddleware(FunctionMiddleware):
    """Middleware extends FunctionMiddleware to provide dynamic discovery and
    interception of all workflow components, including functions and components, without requiring explicit
    per-component configuration.
    """

    def __init__(self, config: DynamicMiddlewareConfig, builder: Builder):
        """Initialize middleware and discover workflow functions.

        Args:
            config: Middleware configuration
            builder: Workflow builder
        """
        super().__init__()
        self._config = config
        self._builder = builder

        self._pre_invoke_policies: list[FunctionPolicyBase] = [
            builder._function_policies[str(ref)].instance for ref in (config.pre_invoke_policy or [])
        ]
        self._post_invoke_policies: list[FunctionPolicyBase] = [
            builder._function_policies[str(ref)].instance for ref in (config.post_invoke_policy or [])
        ]

        self._registered_callables: set[str] = set()

        self._builder_get_llm: Callable | None = None
        self._builder_get_embedder: Callable | None = None
        self._builder_get_retriever: Callable | None = None
        self._builder_get_memory: Callable | None = None
        self._builder_get_object_store: Callable | None = None
        self._builder_get_auth_provider: Callable | None = None
        self._builder_get_function: Callable | None = None

        self._workflow_inventory: WorkflowInventory = WorkflowInventory()

        self._component_allowlists: dict[ComponentGroup, set[str]] = self._build_component_allowlists()

        self._discover_workflow()

    # ==================== Component Discovery and Registration ====================

    async def _discover_and_register_llm(self, llm_name: str, wrapper_type: Any) -> Any:
        """Intercept LLM creation and register allowlisted component functions with middleware.

        Args:
            llm_name: LLM component name
            wrapper_type: LLM framework wrapper type

        Returns:
            The LLM client instance
        """
        # Call the original get_llm to get the actual LLM client
        llm_client = await self._get_builder_get_llm()(llm_name, wrapper_type)

        if not self._should_intercept_llm(llm_name):
            return llm_client

        if any(client.name == llm_name for client in self._workflow_inventory.llms):
            return llm_client

        all_functions = self._get_callable_functions(llm_client, component_type='llm')

        discovered_component = DiscoveredComponent(name=llm_name,
                                                   component_type=ComponentGroup.LLMS,
                                                   instance=llm_client,
                                                   config=None,
                                                   callable_functions=all_functions)
        self._workflow_inventory.llms.append(discovered_component)

        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on LLM '%s': %s", function_name, llm_name, e)

        return llm_client

    async def _discover_and_register_embedder(self, embedder_name: str, wrapper_type: Any) -> Any:
        """Intercept embedder creation and register allowlisted component functions with middleware.

        Args:
            embedder_name: Embedder component name
            wrapper_type: Embedder framework wrapper type

        Returns:
            The Embedder client instance
        """
        # Call the original get_embedder to get the actual embedder client
        embedder_client = await self._get_builder_get_embedder()(embedder_name, wrapper_type)

        if not self._should_intercept_embedder(embedder_name):
            return embedder_client

        if any(client.name == embedder_name for client in self._workflow_inventory.embedders):
            return embedder_client

        all_functions = self._get_callable_functions(embedder_client, component_type='embedder')

        embedder_config = getattr(embedder_client, 'config', None)
        discovered_component = DiscoveredComponent(name=embedder_name,
                                                   component_type=ComponentGroup.EMBEDDERS,
                                                   instance=embedder_client,
                                                   config=embedder_config,
                                                   callable_functions=all_functions)
        self._workflow_inventory.embedders.append(discovered_component)

        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on embedder '%s': %s",
                             function_name,
                             embedder_name,
                             e)

        return embedder_client

    async def _discover_and_register_retriever(self, retriever_name: str, wrapper_type: Any = None):
        """Intercept retriever creation and register allowlisted component functions with middleware.

        Args:
            retriever_name: Retriever component name
            wrapper_type: Retriever framework wrapper type

        Returns:
            The retriever client instance
        """
        retriever_client = await self._get_builder_get_retriever()(retriever_name, wrapper_type)

        if not self._should_intercept_retriever(retriever_name):
            return retriever_client

        if any(client.name == retriever_name for client in self._workflow_inventory.retrievers):
            return retriever_client

        all_functions = self._get_callable_functions(retriever_client, component_type='retriever')

        retriever_config = getattr(retriever_client, 'config', None)
        discovered_component = DiscoveredComponent(name=retriever_name,
                                                   component_type=ComponentGroup.RETRIEVERS,
                                                   instance=retriever_client,
                                                   config=retriever_config,
                                                   callable_functions=all_functions)
        self._workflow_inventory.retrievers.append(discovered_component)

        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on retriever '%s': %s",
                             function_name,
                             retriever_name,
                             e)

        return retriever_client

    async def _discover_and_register_memory(self, memory_name: str):
        """Intercept memory creation and register allowlisted component functions with middleware.

        Args:
            memory_name: Memory component name

        Returns:
            The memory client instance
        """
        memory_client = await self._get_builder_get_memory_client()(memory_name)

        if not self._should_intercept_memory(memory_name):
            return memory_client

        if any(client.name == memory_name for client in self._workflow_inventory.memory):
            return memory_client

        all_functions = self._get_callable_functions(memory_client, component_type='memory')

        memory_config = getattr(memory_client, 'config', None)
        discovered_component = DiscoveredComponent(name=memory_name,
                                                   component_type=ComponentGroup.MEMORY,
                                                   instance=memory_client,
                                                   config=memory_config,
                                                   callable_functions=all_functions)
        self._workflow_inventory.memory.append(discovered_component)

        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on memory '%s': %s",
                             function_name,
                             memory_name,
                             e)

        return memory_client

    async def _discover_and_register_object_store(self, object_store_name: str) -> Any:
        """Intercept object store creation and register allowlisted component functions with middleware.


        Args:
            object_store_name: Object store component name

        Returns:
            The object store client instance
        """
        store_client = await self._get_builder_get_object_store()(object_store_name)

        if not self._should_intercept_object_store(object_store_name):
            return store_client

        if any(client.name == object_store_name for client in self._workflow_inventory.object_stores):
            return store_client

        all_functions = self._get_callable_functions(store_client, component_type='object_store')

        store_config = getattr(store_client, 'config', None)
        discovered_component = DiscoveredComponent(name=object_store_name,
                                                   component_type=ComponentGroup.OBJECT_STORES,
                                                   instance=store_client,
                                                   config=store_config,
                                                   callable_functions=all_functions)
        self._workflow_inventory.object_stores.append(discovered_component)

        # Register all functions - filtering happens in _register_component_function
        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on object store '%s': %s",
                             function_name,
                             object_store_name,
                             e)

        return store_client

    async def _discover_and_register_auth_provider(self, auth_provider_name: str) -> Any:
        """Intercept auth provider creation and register allowlisted component functions with middleware.


        Args:
            auth_provider_name: Auth provider component name

        Returns:
            The auth provider client instance
        """
        auth_client = await self._get_builder_get_auth_provider()(auth_provider_name)

        if not self._should_intercept_auth_provider(auth_provider_name):
            return auth_client

        if any(client.name == auth_provider_name for client in self._workflow_inventory.auth_providers):
            return auth_client

        all_functions = self._get_callable_functions(auth_client, component_type='auth')

        auth_config = getattr(auth_client, 'config', None)
        discovered_component = DiscoveredComponent(name=auth_provider_name,
                                                   component_type=ComponentGroup.AUTHENTICATION,
                                                   instance=auth_client,
                                                   config=auth_config,
                                                   callable_functions=all_functions)
        self._workflow_inventory.auth_providers.append(discovered_component)

        # Register all functions - filtering happens in _register_component_function
        for function_name in all_functions:
            try:
                self._register_component_function(discovered_component, function_name)
            except Exception as e:
                logger.debug("Failed to register component function '%s' on auth provider '%s': %s",
                             function_name,
                             auth_provider_name,
                             e)

        return auth_client

    async def _discover_and_register_function(self, name: str | FunctionRef) -> Function:
        """Intercept workflow function and register with middleware.

        Args:
            name: Function name or reference

        Returns:
            The function instance
        """
        function = await self._get_builder_get_function()(name)

        if not self._config.register_workflow_functions:
            return function

        func_name = str(name)

        if any(f.name == func_name for f in self._workflow_inventory.workflow_functions):
            return function

        func_config = self._builder.get_function_config(name)
        discovered_function = DiscoveredFunction(name=func_name, config=func_config, instance=function)
        self._workflow_inventory.workflow_functions.append(discovered_function)

        # Register with middleware
        self._register_function(discovered_function)

        return function

    # ==================== Internal Discovery and Registration ====================

    def _discover_workflow(self) -> None:
        """Discover workflow functions and patch builder methods for runtime interception."""
        # Patch all builder for runtime discovery and registration
        self._patch_components()

        # Discover registered functions not listed in the config
        self._discover_functions()

    def _discover_functions(self) -> None:
        """Discover and register workflow functions already in the builder."""
        if not self._config.register_workflow_functions:
            return

        if not hasattr(self._builder, '_functions'):
            return

        # Discover functions already registered
        for func_name, configured_func in self._builder._functions.items():  # type: ignore
            # Skip if already in inventory
            if any(func.name == func_name for func in self._workflow_inventory.workflow_functions):
                continue

            # Add to inventory
            discovered_function = DiscoveredFunction(name=func_name,
                                                     config=configured_func.config,
                                                     instance=configured_func.instance)
            self._workflow_inventory.workflow_functions.append(discovered_function)

            # Register with middleware
            self._register_function(discovered_function)

    # ==================== Helper Methods for Interception ====================

    def _should_intercept_llm(self, llm_name: str) -> bool:
        """Check if LLM should be intercepted based on config.

        Args:
            llm_name: Name of the LLM to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == llm_name for client in self._workflow_inventory.llms):
            return False

        # If register_llms is True, intercept all LLMs
        if self._config.register_llms:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.llms is not None and llm_name in self._config.llms

    def _should_intercept_embedder(self, embedder_name: str) -> bool:
        """Check if embedder should be intercepted based on config.

        Args:
            embedder_name: Name of the embedder to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == embedder_name for client in self._workflow_inventory.embedders):
            return False

        # If register_embedders is True, intercept all embedders
        if self._config.register_embedders:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.embedders is not None and embedder_name in self._config.embedders

    def _should_intercept_retriever(self, retriever_name: str) -> bool:
        """Check if retriever should be intercepted based on config.

        Args:
            retriever_name: Name of the retriever to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == retriever_name for client in self._workflow_inventory.retrievers):
            return False

        # If register_retrievers is True, intercept all retrievers
        if self._config.register_retrievers:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.retrievers is not None and retriever_name in self._config.retrievers

    def _should_intercept_memory(self, memory_name: str) -> bool:
        """Check if memory provider should be intercepted based on config.

        Args:
            memory_name: Name of the memory provider to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == memory_name for client in self._workflow_inventory.memory):
            return False

        # If register_memory is True, intercept all memory providers
        if self._config.register_memory:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.memory is not None and memory_name in self._config.memory

    def _should_intercept_object_store(self, store_name: str) -> bool:
        """Check if object store should be intercepted based on config.

        Args:
            store_name: Name of the object store to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == store_name for client in self._workflow_inventory.object_stores):
            return False

        # If register_object_stores is True, intercept all object stores
        if self._config.register_object_stores:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.object_stores is not None and store_name in self._config.object_stores

    def _should_intercept_auth_provider(self, auth_name: str) -> bool:
        """Check if auth provider should be intercepted based on config.

        Args:
            auth_name: Name of the auth provider to check

        Returns:
            True if should intercept, False otherwise
        """
        # Check if already registered
        if any(client.name == auth_name for client in self._workflow_inventory.auth_providers):
            return False

        # If register_auth_providers is True, intercept all auth providers
        if self._config.register_auth_providers:
            return True

        # Otherwise, only intercept if explicitly configured
        return self._config.auth_providers is not None and auth_name in self._config.auth_providers

    def _register_function(self, discovered: DiscoveredFunction) -> None:
        """Register a discovered workflow function with this middleware.

        Args:
            discovered: A DiscoveredFunction from the workflow inventory
        """
        function = discovered.instance
        registration_key = discovered.name

        if registration_key in self._registered_callables:
            logger.debug("Function '%s' already registered, skipping", registration_key)
            return

        # Add this middleware to the function's existing middleware chain
        existing_middleware = list(function.middleware)
        existing_middleware.append(self)
        function.configure_middleware(existing_middleware)

        self._registered_callables.add(registration_key)

    def _register_component_function(self, discovered: DiscoveredComponent, function_name: str) -> None:
        """Register a specific component function from a discovered component.

        Args:
            discovered: A DiscoveredComponent from the workflow inventory
            function_name: Name of the component function to register
        """
        component = discovered.instance
        component_name = discovered.name

        # Validate function exists
        if not hasattr(component, function_name):
            raise ValueError(f"Component function '{function_name}' does not exist on component '{component_name}'")

        # Validate function is in discovered callable_functions
        if function_name not in discovered.callable_functions:
            raise ValueError(
                f"Component function '{function_name}' was not discovered as callable on '{component_name}'. "
                f"Available functions: {sorted(discovered.callable_functions)}")

        # Check allowlist - only auto-register functions in the allowlist (includes user customizations)
        allowlist = self._component_allowlists.get(discovered.component_type, set())
        if function_name not in allowlist:
            logger.debug("Component function '%s.%s' not in allowlist for %s, skipping auto-registration",
                         component_name,
                         function_name,
                         discovered.component_type.value)
            return

        # Check if already registered
        registration_key = f"{component_name}.{function_name}"
        if registration_key in self._registered_callables:
            logger.debug("Component function '%s' already registered, skipping", registration_key)
            return

        # Wrap it with middleware
        wrapped_function = self._configure_component_function_middleware(discovered, function_name)

        # Replace the function on the component instance
        object.__setattr__(component, function_name, wrapped_function)

        self._registered_callables.add(registration_key)
        logger.debug("Registered component function '%s'", registration_key)

    def _configure_component_function_middleware(self, discovered: DiscoveredComponent, function_name: str) -> Any:
        """Wrap a component function with middleware interception.

        Args:
            discovered: The DiscoveredComponent from the workflow inventory
            function_name: Name of the component function to wrap

        Returns:
            Wrapped component function
        """
        component_instance = discovered.instance
        component_name = discovered.name
        original_function = getattr(component_instance, function_name)

        # Verify function has __name__
        if not hasattr(original_function, '__name__'):
            raise RuntimeError(
                f"Component function '{function_name}' on component '{component_name}' has no __name__ attribute")

        registration_key = f"{component_name}.{function_name}"

        # Check if already registered - return original function to prevent nested wrapping
        if registration_key in self._registered_callables:
            return original_function

        # Extract metadata safely - defaults to None for missing/inaccessible attributes
        component_config = self._extract_component_attributes(discovered, 'config')
        description = self._extract_component_attributes(discovered, 'description')
        input_schema = self._extract_component_attributes(discovered, 'input_schema')
        single_output_schema = self._extract_component_attributes(discovered, 'single_output_schema')
        stream_output_schema = self._extract_component_attributes(discovered, 'streaming_output_schema')

        context = FunctionMiddlewareContext(name=function_name,
                                            config=component_config,
                                            description=description,
                                            input_schema=input_schema,
                                            single_output_schema=single_output_schema,
                                            stream_output_schema=stream_output_schema)

        chain = FunctionMiddlewareChain(middleware=[self], context=context)

        if inspect.isasyncgenfunction(original_function):
            wrapped_function = chain.build_stream(original_function)
        else:
            wrapped_function = chain.build_single(original_function)

        # Track registration
        self._registered_callables.add(registration_key)

        return wrapped_function

    # ==================== Helper Methods ====================

    def _build_component_allowlists(self) -> dict[ComponentGroup, set[str]]:
        """Build component allowlists from config (merged with defaults).

        Returns:
            Dict mapping ComponentGroup enums to sets of allowed function names
        """
        if self._config.allowed_component_functions is None:
            # No custom config, use defaults
            return {k: v.copy() for k, v in COMPONENT_FUNCTION_ALLOWLISTS.items()}

        allowed = self._config.allowed_component_functions
        return {
            ComponentGroup.LLMS: allowed.llms,  # type: ignore[dict-item]
            ComponentGroup.EMBEDDERS: allowed.embedders,  # type: ignore[dict-item]
            ComponentGroup.RETRIEVERS: allowed.retrievers,  # type: ignore[dict-item]
            ComponentGroup.MEMORY: allowed.memory,  # type: ignore[dict-item]
            ComponentGroup.OBJECT_STORES: allowed.object_stores,  # type: ignore[dict-item]
            ComponentGroup.AUTHENTICATION: allowed.authentication,  # type: ignore[dict-item]
        }

    def _extract_component_attributes(self, discovered: DiscoveredComponent, attr_name: str) -> Any:
        """Safely extract an attribute from a discovered component's instance.

        Args:
            discovered: DiscoveredComponent containing the component instance
            attr_name: Name of the attribute to extract from the component instance

        Returns:
            Attribute value or None if it cannot be safely extracted
        """
        try:
            obj = discovered.instance
            # Check class-level attribute to avoid triggering async properties
            class_attr = getattr(type(obj), attr_name, None)
            if isinstance(class_attr, property):
                return None

            value = getattr(obj, attr_name, None)
            if callable(value) or inspect.iscoroutine(value):
                return None
            return value
        except Exception:
            return None

    # ==================== Policy Orchestration ====================

    async def function_middleware_invoke(self,
                                         *args: Any,
                                         call_next: CallNext,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> Any:
        """Execute function with policy orchestration for single-output.

        Runs pre-invoke policies, calls the function, then runs post-invoke policies.
        Each policy can transform the input/output.

        Args:
            *args: Positional arguments
            call_next: Next middleware or function
            context: Function metadata
            **kwargs: Additional function arguments

        Returns:
            The output from the function
        """
        # PRE-INVOKE: Build transformation chain through policies
        pre_context = PreInvokeContext(function_context=context,
                                       original_args=args,
                                       function_args=args,
                                       function_kwargs=kwargs)

        for policy in self._pre_invoke_policies:
            if not policy.config.enabled:
                continue

            try:
                modified_args = await policy.on_pre_invoke(pre_context)

                if modified_args is not None:
                    if len(modified_args) != len(pre_context.function_args):
                        raise ValueError(f"Policy '{policy.name}' returned {len(modified_args)} args, "
                                         f"expected {len(pre_context.function_args)}")
                    pre_context.function_args = modified_args
            except Exception as e:
                logger.error(
                    "Pre-invoke policy '%s' failed for function '%s' - skipping policy and continuing. Error: %s",
                    policy.name,
                    context.name,
                    str(e),
                    exc_info=True)

        # INVOKE: Call actual function with transformed args
        output = await call_next(*pre_context.function_args, **kwargs)

        # POST-INVOKE: Build transformation chain through policies
        post_context = PostInvokeContext(function_context=context,
                                         original_args=args,
                                         function_args=pre_context.function_args,
                                         function_kwargs=kwargs,
                                         function_output=output)
        current_output = output

        for policy in self._post_invoke_policies:
            if not policy.config.enabled:
                continue

            try:
                modified_output = await policy.on_post_invoke(post_context)

                if modified_output is not None:
                    current_output = modified_output
            except Exception as e:
                logger.error(
                    "Post-invoke policy '%s' failed for function '%s' - skipping policy and continuing. Error: %s",
                    policy.name,
                    context.name,
                    str(e),
                    exc_info=True)

        return current_output

    async def function_middleware_stream(self,
                                         *args: Any,
                                         call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        """Execute function with policy orchestration for streaming.

        Pre-invoke policies run once before streaming starts. Post-invoke policies
        apply to each chunk as it streams.

        Args:
            *args: Positional arguments
            call_next: Next middleware or streaming function
            context: Function metadata
            **kwargs: Additional function arguments

        Yields:
            Stream chunks (possibly transformed by policies)
        """
        # PRE-INVOKE: Build transformation chain through policies
        pre_context = PreInvokeContext(function_context=context,
                                       original_args=args,
                                       function_args=args,
                                       function_kwargs=kwargs)

        for policy in self._pre_invoke_policies:
            if not policy.config.enabled:
                continue

            try:
                modified_args = await policy.on_pre_invoke(pre_context)

                if modified_args is not None:
                    if len(modified_args) != len(pre_context.function_args):
                        raise ValueError(f"Policy '{policy.name}' returned {len(modified_args)} args, "
                                         f"expected {len(pre_context.function_args)}")
                    pre_context.function_args = modified_args
            except Exception as e:
                logger.error(
                    "Pre-invoke policy '%s' failed for streaming function '%s' - skipping policy and continuing. Error: %s",
                    policy.name,
                    context.name,
                    str(e),
                    exc_info=True)

        # STREAM: Call function with transformed args and yield chunks
        async for chunk in call_next(*pre_context.function_args, **kwargs):

            # POST-INVOKE: Build transformation chain through policies for each chunk
            post_context = PostInvokeContext(function_context=context,
                                             original_args=args,
                                             function_args=pre_context.function_args,
                                             function_kwargs=kwargs,
                                             function_output=chunk)
            current_chunk = chunk

            for policy in self._post_invoke_policies:
                if not policy.config.enabled:
                    continue

                try:
                    modified_chunk = await policy.on_post_invoke(post_context)

                    if modified_chunk is not None:
                        current_chunk = modified_chunk
                except Exception as e:
                    logger.error(
                        "Post-invoke policy '%s' failed for chunk in streaming function '%s' - skipping policy and continuing. Error: %s",
                        policy.name,
                        context.name,
                        str(e),
                        exc_info=True)
            yield current_chunk

    # ==================== Helper Methods ====================

    def _get_callable_functions(self, instance: Any, component_type: str | None = None) -> set[str]:
        """Get all callable functions from component instance that can be safely wrapped.

        This discovers ALL potentially wrappable component functions without allowlist filtering.
        Safety checks ensure only valid, callable, bound functions are included.

        Args:
            instance: The component instance to introspect
            component_type: Type of component (for logging/metadata, not filtering)

        Returns:
            Set of all valid component function names that could be wrapped
        """
        functions = set()

        for function_name in dir(instance):
            # Skip private/dunder functions
            if function_name.startswith('_'):
                continue

            try:
                # Must pass basic validity checks (no errors)
                if not self._is_valid_wrappable_function(instance, function_name):
                    continue

                # Passed all safety checks - this component function CAN be wrapped
                functions.add(function_name)

            except Exception:
                # Skip functions that raise errors during introspection
                continue

        return functions

    def _is_valid_wrappable_function(self, instance: Any, function_name: str) -> bool:
        """Check if a component function passes all safety checks for wrapping.

        This is the gatekeeper for what CAN be wrapped (not what SHOULD be).

        Args:
            instance: The component instance
            function_name: Name of the component function to check

        Returns:
            True if component function is safe to wrap, False otherwise
        """
        try:
            instance_class = type(instance)

            # Check if function exists
            if not hasattr(instance, function_name):
                return False

            # Get class-level attribute to check type
            class_attr = getattr(instance_class, function_name, None)

            # Skip properties
            if isinstance(class_attr, property):
                return False

            # Skip static/class methods
            if isinstance(class_attr, (staticmethod, classmethod)):
                return False

            # Get instance attribute
            attr = getattr(instance, function_name, None)
            if attr is None or not callable(attr):
                return False

            # Must be a bound method (component function)
            if not inspect.ismethod(attr):
                return False

            # Must be bound to our instance
            if not hasattr(attr, '__self__') or attr.__self__ is not instance:
                return False

            # Must have a valid signature
            try:
                inspect.signature(attr)
            except (ValueError, TypeError):
                return False

            return True

        except Exception:
            return False

    def _patch_components(self):
        """Patch builder getter methods to enable runtime discovery and registration."""
        self._patch_get_llm()
        self._patch_get_embedder()
        self._patch_get_retriever()
        self._patch_get_memory()
        self._patch_get_object_store()
        self._patch_get_auth_provider()
        self._patch_get_function()

    def _patch_get_llm(self):
        """Patch builder.get_llm() for runtime LLM interception."""
        if not hasattr(self._builder, 'get_llm'):
            raise RuntimeError("Builder does not have 'get_llm' method. Cannot patch LLM creation.")

        self._builder_get_llm = self._builder.get_llm
        self._builder.get_llm = self._discover_and_register_llm

    def _patch_get_embedder(self):
        """Patch builder.get_embedder() for runtime embedder interception."""
        if not hasattr(self._builder, 'get_embedder'):
            raise RuntimeError("Builder does not have 'get_embedder' method. Cannot patch embedder creation.")

        self._builder_get_embedder = self._builder.get_embedder
        self._builder.get_embedder = self._discover_and_register_embedder

    def _patch_get_retriever(self):
        """Patch builder.get_retriever() for runtime retriever interception."""
        if not hasattr(self._builder, 'get_retriever'):
            raise RuntimeError("Builder does not have 'get_retriever' method. Cannot patch retriever creation.")

        self._builder_get_retriever = self._builder.get_retriever
        self._builder.get_retriever = self._discover_and_register_retriever

    def _patch_get_memory(self):
        """Patch builder.get_memory_client() for runtime memory provider interception."""
        if not hasattr(self._builder, 'get_memory_client'):
            raise RuntimeError("Builder does not have 'get_memory_client' method. Cannot patch memory creation.")

        self._builder_get_memory = self._builder.get_memory_client
        self._builder.get_memory_client = self._discover_and_register_memory

    def _patch_get_object_store(self):
        """Patch builder.get_object_store_client() for runtime object store interception."""
        if not hasattr(self._builder, 'get_object_store_client'):
            raise RuntimeError("Builder does not have 'get_object_store_client' method. "
                               "Cannot patch object store creation.")

        self._builder_get_object_store = self._builder.get_object_store_client
        self._builder.get_object_store_client = self._discover_and_register_object_store

    def _patch_get_auth_provider(self):
        """Patch builder.get_auth_provider() for runtime auth provider interception."""
        if not hasattr(self._builder, 'get_auth_provider'):
            raise RuntimeError("Builder does not have 'get_auth_provider' method. Cannot patch auth provider creation.")

        self._builder_get_auth_provider = self._builder.get_auth_provider
        self._builder.get_auth_provider = self._discover_and_register_auth_provider

    def _patch_get_function(self):
        """Patch builder.get_function() for runtime function interception."""
        if not hasattr(self._builder, 'get_function'):
            raise RuntimeError("Builder does not have 'get_function' method. Cannot patch function retrieval.")

        self._builder_get_function = self._builder.get_function
        self._builder.get_function = self._discover_and_register_function

    # ==================== Original Method Getters ====================

    def _get_builder_get_llm(self):
        """Return original builder.get_llm method."""
        if self._builder_get_llm is None:
            raise RuntimeError("get_llm has not been patched yet")
        return self._builder_get_llm

    def _get_builder_get_embedder(self):
        """Return original builder.get_embedder method."""
        if self._builder_get_embedder is None:
            raise RuntimeError("get_embedder has not been patched yet")
        return self._builder_get_embedder

    def _get_builder_get_retriever(self):
        """Return original builder.get_retriever method."""
        if self._builder_get_retriever is None:
            raise RuntimeError("get_retriever has not been patched yet")
        return self._builder_get_retriever

    def _get_builder_get_memory_client(self):
        """Return original builder.get_memory_client method."""
        if self._builder_get_memory is None:
            raise RuntimeError("get_memory_client has not been patched yet")
        return self._builder_get_memory

    def _get_builder_get_object_store(self):
        """Return original builder.get_object_store_client method."""
        if self._builder_get_object_store is None:
            raise RuntimeError("get_object_store_client has not been patched yet")
        return self._builder_get_object_store

    def _get_builder_get_auth_provider(self):
        """Return original builder.get_auth_provider method."""
        if self._builder_get_auth_provider is None:
            raise RuntimeError("get_auth_provider has not been patched yet")
        return self._builder_get_auth_provider

    def _get_builder_get_function(self):
        """Return original builder.get_function method."""
        if self._builder_get_function is None:
            raise RuntimeError("get_function has not been patched yet")
        return self._builder_get_function


__all__ = ["DynamicFunctionMiddleware"]
