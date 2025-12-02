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
"""Tests for function policy CLI commands."""

import pytest
from click.testing import CliRunner

from nat.cli.commands.info.list_function_policies import list_function_policies
from nat.cli.commands.info.list_function_policies import show_function_policy


@pytest.fixture(name="cli_runner")
def fixture_cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


class TestListFunctionPolicies:
    """Test 'nat info function-policies' command."""

    def test_list_function_policies_basic(self, cli_runner):
        """Test basic list without verbose flag."""
        result = cli_runner.invoke(list_function_policies, [])
        assert result.exit_code == 0
        assert "function policies" in result.output.lower()

    def test_list_function_policies_verbose(self, cli_runner):
        """Test list with --verbose flag."""
        result = cli_runner.invoke(list_function_policies, ["--verbose"])
        assert result.exit_code == 0
        # Should show config fields when verbose
        if "function policies" in result.output.lower() and "found" in result.output.lower():
            # Has policies registered, check for verbose output
            assert "config" in result.output.lower() or "field" in result.output.lower() or len(result.output) > 100

    @pytest.mark.parametrize("verbose_flag", [[], ["--verbose"], ["-v"]])
    def test_list_function_policies_verbose_variations(self, cli_runner, verbose_flag):
        """Test verbose flag variations."""
        result = cli_runner.invoke(list_function_policies, verbose_flag)
        assert result.exit_code == 0


class TestShowFunctionPolicy:
    """Test 'nat info function-policy <name>' command."""

    def test_show_function_policy_not_found(self, cli_runner):
        """Test showing details for a non-existent policy."""
        result = cli_runner.invoke(show_function_policy, ["nonexistent.policy"])
        assert "not found" in result.output.lower()
        # Should show available policies
        assert "available" in result.output.lower() or "policies" in result.output.lower()

    def test_show_function_policy_missing_argument(self, cli_runner):
        """Test that command requires policy name argument."""
        result = cli_runner.invoke(show_function_policy, [])
        assert result.exit_code != 0
        # Click will show error about missing argument

    def test_show_function_policy_output_format(self, cli_runner):
        """Test that show command has proper output format for any policy."""
        # First get list of available policies
        list_result = cli_runner.invoke(list_function_policies, [])

        # Extract first policy name if any exist
        if "function policies" in list_result.output.lower():
            lines = list_result.output.split('\n')
            policy_name = None
            for line in lines:
                if '•' in line:
                    # Extract policy name from bullet point
                    policy_name = line.split('•')[1].strip()
                    if policy_name:
                        break

            if policy_name:
                # Test showing that policy
                show_result = cli_runner.invoke(show_function_policy, [policy_name])
                assert show_result.exit_code == 0
                assert policy_name in show_result.output


class TestCLIIntegration:
    """Test CLI integration and error handling."""

    def test_list_handles_no_policies_gracefully(self, cli_runner):
        """Test that list command handles no registered policies."""
        result = cli_runner.invoke(list_function_policies, [])
        assert result.exit_code == 0
        # Should either show policies or a message about none found
        assert len(result.output) > 0

    def test_show_provides_helpful_error(self, cli_runner):
        """Test that show command provides helpful error for invalid policy."""
        result = cli_runner.invoke(show_function_policy, ["invalid_policy_name"])
        assert "not found" in result.output.lower()
        # Should suggest available policies
        assert result.exit_code == 0  # Command runs successfully, just doesn't find the policy
