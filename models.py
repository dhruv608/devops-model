# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Action / Observation pydantic models for SafeSreEnvironment.

Action is a tool-dispatch envelope: ``{"tool": "<method_name>", "args": {...}}``.
The env's step() inspects ``tool`` and routes to the matching method. Read-only
tools (read_log, list_processes, ...) and the mutating tools (execute_bash,
submit_fix) all share this single shape so TRL only has to learn one schema.

Observation carries Linux-shell-flavoured ``stdout`` / ``stderr`` strings so
the model sees output it recognises from training data.
"""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SafeSreAction(Action):
    """One tool call from the agent.

    The envelope is intentionally permissive: ``args`` is a free-form dict so
    each tool can validate its own keys without a global schema dance.

    Examples:

        # 1. Investigate first (no args needed)
        SafeSreAction(tool="list_ports", args={})
        SafeSreAction(tool="list_processes", args={})

        # 2. Read a file
        SafeSreAction(tool="read_log",
                      args={"path": "/var/log/nginx/error.log"})

        # 3. Check a service status
        SafeSreAction(tool="check_service_status",
                      args={"service": "nginx"})

        # 4. Run a bash script (gets AST-classified for safety)
        SafeSreAction(tool="execute_bash",
                      args={"script": "kill -9 4051 && systemctl restart nginx"})

        # 5. Declare the fix complete (terminal action)
        SafeSreAction(tool="submit_fix",
                      args={"claim": "Killed rogue PID 4051, restarted nginx"})

    A catastrophic command is rejected by the env's bash classifier
    BEFORE it executes:

        SafeSreAction(tool="execute_bash", args={"script": "rm -rf /"})
        # -> stdout: "[BLOCKED BY SAFETY HARNESS] command refused: rm -rf /"
    """

    tool: str = Field(
        ...,
        description=(
            "Name of the tool to invoke. Must be one of the env's exposed methods: "
            "read_log, list_processes, check_service_status, check_disk_usage, "
            "list_ports, list_files, execute_bash, submit_fix."
        ),
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific keyword arguments.",
    )


class SafeSreObservation(Observation):
    """One step's response. ``stdout`` / ``stderr`` mimic shell output so the
    model can interpret tool results the way it'd interpret real CLI output.

    ``done`` flips True only when ``submit_fix`` is called or the turn limit
    is hit. ``reward`` stays 0 mid-episode and is filled in at terminal step
    by SafeSreEnvironment._compute_terminal_rewards (Hour 7).
    """

    stdout: str = Field(default="", description="Tool stdout output.")
    stderr: str = Field(default="", description="Tool stderr output (errors).")
    turn_count: int = Field(default=0, description="Number of step() calls so far.")
