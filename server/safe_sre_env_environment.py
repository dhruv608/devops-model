# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Safe-SRE Environment.

A multi-turn tool-using SRE simulator. ``reset()`` samples one scenario from
``data/train_scenarios.json``, builds a fresh ``SimulatedSystem`` from its
``initial_state``, and returns the incident text + tool listing in the
observation. The agent then calls read-only tools to investigate and (later
hours) ``execute_bash`` + ``submit_fix`` to act.

This file ships the **6 read-only tools** (Hour 3). reset/step plumbing is
fleshed out at Hour 4; ``execute_bash`` lands at Hour 6; reward computation
at Hour 7.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Make project root importable when this file is loaded under the
# ``server.safe_sre_env_environment`` module path (running pytest /
# uvicorn from the project root makes ``D:\scaler\safe_sre_env`` the cwd).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from ..models import SafeSreAction, SafeSreObservation
except ImportError:
    from models import SafeSreAction, SafeSreObservation

from core.scenarios import Scenario, load_scenarios
from core.state import SimulatedSystem


_DEFAULT_SCENARIOS_PATH = _PROJECT_ROOT / "data" / "train_scenarios.json"

# Tools the agent can call. Keep this list in sync with the system prompt
# emitted from reset() and with the dispatch table in step().
READ_ONLY_TOOLS = (
    "read_log",
    "list_processes",
    "check_service_status",
    "check_disk_usage",
    "list_ports",
    "list_files",
)


class SafeSreEnvironment(Environment):
    """The Safe-Rollback SRE/DevOps environment.

    One episode = one incident scenario. The agent investigates with
    read-only tools, then mutates state via ``execute_bash`` (Hour 6),
    then declares the fix via ``submit_fix`` (Hour 4) for terminal
    reward (Hour 7).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenarios_path: str | Path = _DEFAULT_SCENARIOS_PATH,
    ) -> None:
        super().__init__()
        self._scenarios_path = Path(scenarios_path)
        self._scenarios: list[Scenario] = load_scenarios(self._scenarios_path)
        if not self._scenarios:
            raise RuntimeError(
                f"No scenarios loaded from {self._scenarios_path}; "
                f"refusing to start with an empty curriculum."
            )

        # Per-episode state -- populated by reset().
        self._state: State = State(episode_id=str(uuid4()), step_count=0)
        self.scenario: Scenario = self._scenarios[0]
        self.system: SimulatedSystem = SimulatedSystem.from_initial(
            self.scenario.initial_state
        )
        self._turn_count: int = 0
        self._terminated: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **_kwargs: Any,
    ) -> SafeSreObservation:
        """Pick one scenario (deterministically by ``seed``) and load it."""
        idx = (seed or 0) % len(self._scenarios)
        self.scenario = self._scenarios[idx]
        self.system = SimulatedSystem.from_initial(self.scenario.initial_state)
        self._turn_count = 0
        self._terminated = False
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        return SafeSreObservation(
            stdout=self._intro_text(),
            stderr="",
            done=False,
            reward=0.0,
            turn_count=0,
            metadata={
                "scenario_id": self.scenario.id,
                "category": self.scenario.category,
                "difficulty": self.scenario.difficulty,
                "available_tools": list(READ_ONLY_TOOLS),
            },
        )

    def step(  # type: ignore[override]
        self,
        action: SafeSreAction,
        timeout_s: float | None = None,
        **_kwargs: Any,
    ) -> SafeSreObservation:
        """Dispatch to a read-only tool by name (Hour 3 scope).

        execute_bash and submit_fix come online at Hour 4-6. Unknown
        tool names return a tool-error observation rather than crashing
        the rollout -- TRL will learn the available_tools list from the
        system prompt either way.
        """
        self._turn_count += 1
        self._state.step_count += 1

        tool = (action.tool or "").strip()
        args = dict(action.args or {})

        if tool in READ_ONLY_TOOLS:
            stdout = getattr(self, tool)(**args)
            return self._obs(stdout=stdout)

        # Hour 4-6 tools: stub out with a clear error so missing wiring
        # is obvious in logs rather than silently zero-rewarded.
        if tool in {"execute_bash", "submit_fix"}:
            return self._obs(
                stderr=f"[not yet implemented at Hour 3] tool={tool!r}",
            )

        return self._obs(
            stderr=f"unknown tool {tool!r}; expected one of {list(READ_ONLY_TOOLS)}",
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Read-only tools (Hour 3)
    # ------------------------------------------------------------------

    def read_log(self, path: str) -> str:
        """Return the contents of a file as if `cat <path>`.

        Args:
            path: Absolute path on the simulated filesystem.
        """
        if path in self.system.files:
            return self.system.files[path]
        return f"cat: {path}: No such file or directory"

    def list_processes(self, filter: str = "") -> str:
        """Return a `ps aux`-style table of running processes.

        Args:
            filter: Optional substring to grep against the cmd column.
        """
        rows = ["USER       PID %CPU %MEM COMMAND"]
        for p in self.system.processes:
            if filter and filter not in p.get("cmd", ""):
                continue
            rows.append(
                "{user:<10} {pid:<5} {cpu:<4} {mem:<4} {cmd}".format(
                    user=p.get("user", "?"),
                    pid=p.get("pid", "?"),
                    cpu=p.get("cpu_pct", 0),
                    mem=p.get("rss_mb", 0),
                    cmd=p.get("cmd", ""),
                )
            )
        if len(rows) == 1:
            rows.append("(no matching processes)")
        return "\n".join(rows)

    def check_service_status(self, service: str) -> str:
        """Return a `systemctl status <service>`-style summary."""
        svc = self.system.services.get(service)
        if svc is None:
            return f"Unit {service}.service could not be found."
        status = svc.get("status", "unknown")
        exit_code = svc.get("exit_code", 0)
        restarts = svc.get("restart_count", 0)
        return (
            f"* {service}.service\n"
            f"   Active: {status} (exit={exit_code}, restarts={restarts})\n"
        )

    def check_disk_usage(self, path: str = "/") -> str:
        """Return a `df -h`-style summary for the given path (or all paths if `/`)."""
        if path == "/":
            entries = sorted(self.system.disk_usage.items())
        else:
            entries = [(path, self.system.disk_usage.get(path, 0))]
        rows = ["Path                                Used (KB)"]
        for p, used in entries:
            rows.append(f"{p:<35} {used}")
        if len(rows) == 1:
            rows.append("(no disk_usage data for this scenario)")
        return "\n".join(rows)

    def list_ports(self) -> str:
        """Return a `ss -tlnp`-style listing of port -> pid bindings."""
        if not self.system.ports:
            return "(no ports bound)"
        rows = ["LISTEN  Local-Port  PID  Process"]
        for port, pid in sorted(self.system.ports.items()):
            cmd = next(
                (p.get("cmd", "?") for p in self.system.processes if p.get("pid") == pid),
                "?",
            )
            rows.append(f"LISTEN  {port:<10}  {pid}  {cmd}")
        return "\n".join(rows)

    def list_files(self, path: str = "/") -> str:
        """Return an `ls -la`-style listing of files whose path starts with ``path``."""
        matches = sorted(p for p in self.system.files if p.startswith(path))
        if not matches:
            return f"ls: cannot access '{path}': No such file or directory"
        rows = [f"total {len(matches)}"]
        for p in matches:
            mode = self.system.file_modes.get(p, 0o644)
            owner = self.system.file_owners.get(p, "root")
            size = len(self.system.files[p])
            rows.append(f"{_mode_to_str(mode)} 1 {owner} {owner} {size:>6} {p}")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _intro_text(self) -> str:
        """The text the agent sees on reset(). System prompt comes from the
        trainer side; here we surface the incident-specific brief."""
        return (
            f"=== INCIDENT {self.scenario.id} ({self.scenario.category}, "
            f"{self.scenario.difficulty}) ===\n"
            f"{self.scenario.incident_text}\n\n"
            f"Available tools: {', '.join(READ_ONLY_TOOLS)}, execute_bash, submit_fix."
        )

    def _obs(
        self,
        stdout: str = "",
        stderr: str = "",
        done: bool = False,
        reward: float = 0.0,
    ) -> SafeSreObservation:
        return SafeSreObservation(
            stdout=stdout,
            stderr=stderr,
            done=done or self._terminated,
            reward=reward,
            turn_count=self._turn_count,
            metadata={
                "scenario_id": self.scenario.id,
                "mutation_count": self.system.mutation_count,
            },
        )


def _mode_to_str(mode: int) -> str:
    """Turn 0o644 into ``-rw-r--r--`` for ls -la flavour."""
    chars = ["-"] * 10
    bits = [
        (0o400, 1, "r"), (0o200, 2, "w"), (0o100, 3, "x"),
        (0o040, 4, "r"), (0o020, 5, "w"), (0o010, 6, "x"),
        (0o004, 7, "r"), (0o002, 8, "w"), (0o001, 9, "x"),
    ]
    for mask, idx, ch in bits:
        if mode & mask:
            chars[idx] = ch
    return "".join(chars)
