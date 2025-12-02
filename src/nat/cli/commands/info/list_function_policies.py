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

import click


@click.command(name="function-policies", help="List all available function policies.")
@click.option("-v",
              "--verbose",
              is_flag=True,
              default=False,
              help="Show detailed information including descriptions and config schemas")
def list_function_policies(verbose: bool) -> None:
    """List all registered function policies.

    Args:
        verbose: Show detailed information about each policy
    """
    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Discover and register all plugins to ensure function policies are loaded
    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()
    policies = registry.get_registered_function_policies()

    if not policies:
        click.echo("No function policies found.")
        return

    click.echo(f"\nFound {len(policies)} function policies:\n")

    for policy_info in sorted(policies, key=lambda p: p.config_type.full_type):
        # Basic info
        click.echo(f"  • {policy_info.config_type.full_type}")

        if verbose:
            # Show discovery metadata
            if policy_info.discovery_metadata:
                metadata = policy_info.discovery_metadata
                if metadata.description:
                    click.echo(f"    {metadata.description}")
                if metadata.package:
                    click.echo(f"    Package: {metadata.package}")

            # Show config fields
            config_type = policy_info.config_type
            if hasattr(config_type, 'model_fields'):
                fields = config_type.model_fields
                if fields:
                    click.echo("    Config fields:")
                    for field_name, field_info in fields.items():
                        if not field_name.startswith('_'):
                            desc = field_info.description or "No description"
                            click.echo(f"      - {field_name}: {desc}")

            click.echo()  # Blank line between verbose entries

    if not verbose:
        click.echo("Use --verbose for detailed information")


@click.command(name="function-policy", help="Show detailed information about a specific function policy.")
@click.argument("policy_name", type=str)
def show_function_policy(policy_name: str) -> None:
    """Show detailed information about a specific function policy.

    Args:
        policy_name: The full_type name of the policy (e.g., 'input_logging')
    """
    from nat.cli.type_registry import GlobalTypeRegistry
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Discover and register all plugins
    discover_and_register_plugins(PluginTypes.ALL)

    registry = GlobalTypeRegistry.get()
    policies = registry.get_registered_function_policies()

    # Find the policy
    policy_info = None
    for policy in policies:
        if policy.config_type.full_type == policy_name:
            policy_info = policy
            break

    if policy_info is None:
        click.echo(f"Function policy '{policy_name}' not found.")
        click.echo("\nAvailable policies:")
        for policy in sorted(policies, key=lambda p: p.config_type.full_type):
            click.echo(f"  • {policy.config_type.full_type}")
        return

    # Show detailed information
    click.echo(f"\nFunction Policy: {policy_info.config_type.full_type}")
    click.echo()

    # Discovery metadata
    if policy_info.discovery_metadata:
        metadata = policy_info.discovery_metadata
        if metadata.description:
            click.echo(f"Description: {metadata.description}")
        if metadata.package:
            click.echo(f"Package: {metadata.package}")
        click.echo()

    # Config type information
    click.echo(f"Config Type: {policy_info.config_type.__name__}")
    click.echo(f"Full Type: {policy_info.config_type.full_type}")
    click.echo()

    # Config fields
    config_type = policy_info.config_type
    if hasattr(config_type, 'model_fields'):
        fields = config_type.model_fields
        if fields:
            click.echo("Configuration Fields:")
            for field_name, field_info in fields.items():
                if not field_name.startswith('_'):
                    field_type = field_info.annotation
                    default = field_info.default if field_info.default is not None else "None"
                    desc = field_info.description or "No description"
                    click.echo(f"  • {field_name}")
                    click.echo(f"      Type: {field_type}")
                    click.echo(f"      Default: {default}")
                    click.echo(f"      Description: {desc}")
            click.echo()

    # Usage example
    click.echo("Example Configuration:")
    click.echo(f"""
function_policies:
  my_policy:
    _type: {policy_info.config_type.full_type}
    # ... additional config fields ...

middleware:
  my_middleware:
    _type: nat/dynamic
    pre_invoke_policies:
      - my_policy
    # or use post_invoke_policies:
    # post_invoke_policies:
    #   - my_policy
""")
