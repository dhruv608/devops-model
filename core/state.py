"""In-memory simulated Linux system state.

The agent never touches a real filesystem. All "actions" route into a
SimulatedSystem instance whose dicts model files, services, processes,
ports, and disk usage. Every mutation appends to ``mutation_log`` — that
log is the source of truth for ``reward_minimality`` (blast-radius proxy)
and for the post-episode audit trail in the demo before/after table.

Field shapes match the JSON ``initial_state`` in scenarios (strategy.md
§3.1) so ``SimulatedSystem.from_initial(scenario["initial_state"])`` is
the only thing ``reset()`` needs to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Default Linux file metadata for files loaded from a scenario without
# explicit mode/owner. Matches typical /etc and /var/log entries.
_DEFAULT_MODE = 0o644
_DEFAULT_OWNER = "root"
_DEFAULT_GROUP = "root"


@dataclass
class SimulatedSystem:
    """Mutable state for one episode.

    A scenario's ``initial_state`` dict is loaded once via ``from_initial``
    at ``reset()``; from there every tool the agent calls dispatches into
    one of the mutation methods on this class. Read-only env tools (e.g.
    ``read_log``) just index into the public dicts directly.
    """

    files: dict[str, str] = field(default_factory=dict)
    services: dict[str, dict[str, Any]] = field(default_factory=dict)
    processes: list[dict[str, Any]] = field(default_factory=list)
    ports: dict[int, int] = field(default_factory=dict)
    disk_usage: dict[str, int] = field(default_factory=dict)

    # File metadata kept parallel to ``files`` so the canonical content map
    # stays ``dict[str,str]`` per the strategy doc spec. chmod/chown only
    # touch these two dicts, never ``files``.
    file_modes: dict[str, int] = field(default_factory=dict)
    file_owners: dict[str, str] = field(default_factory=dict)

    mutation_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_initial(cls, state_dict: dict[str, Any]) -> "SimulatedSystem":
        """Build a fresh system from a scenario's ``initial_state`` block.

        Accepts the JSON shape documented in strategy.md §3.1. Missing keys
        default to empty collections; file mode/owner default to 0o644 root.
        """
        sys_ = cls()

        for path, content in (state_dict.get("files") or {}).items():
            sys_.files[path] = content
            sys_.file_modes[path] = _DEFAULT_MODE
            sys_.file_owners[path] = _DEFAULT_OWNER

        sys_.services = dict(state_dict.get("services") or {})
        sys_.processes = list(state_dict.get("processes") or [])
        # JSON keys come back as str; coerce to int for port->pid lookup.
        sys_.ports = {
            int(port): int(pid)
            for port, pid in (state_dict.get("ports") or {}).items()
        }
        sys_.disk_usage = dict(state_dict.get("disk_usage") or {})

        # Optional explicit metadata override blocks.
        for path, mode in (state_dict.get("file_modes") or {}).items():
            sys_.file_modes[path] = mode
        for path, owner in (state_dict.get("file_owners") or {}).items():
            sys_.file_owners[path] = owner

        return sys_

    def _log(self, op: str, **kwargs: Any) -> None:
        """Append one entry to mutation_log. Sequence is implicit by index."""
        self.mutation_log.append({"op": op, "args": kwargs})

    def set_file(self, path: str, content: str) -> None:
        """Create or overwrite a file's content. Preserves existing mode/owner."""
        self.files[path] = content
        self.file_modes.setdefault(path, _DEFAULT_MODE)
        self.file_owners.setdefault(path, _DEFAULT_OWNER)
        self._log("set_file", path=path, content_len=len(content))

    def delete_file(self, path: str) -> bool:
        """Remove a file. Returns True iff the file existed."""
        existed = path in self.files
        self.files.pop(path, None)
        self.file_modes.pop(path, None)
        self.file_owners.pop(path, None)
        self._log("delete_file", path=path, existed=existed)
        return existed

    def kill_pid(self, pid: int) -> bool:
        """Terminate a process; release any ports it held.

        Returns True iff a process with that pid existed.
        """
        before = len(self.processes)
        self.processes = [p for p in self.processes if p.get("pid") != pid]
        killed = len(self.processes) < before

        freed = [port for port, owner_pid in self.ports.items() if owner_pid == pid]
        for port in freed:
            del self.ports[port]

        self._log("kill_pid", pid=pid, killed=killed, freed_ports=freed)
        return killed

    def start_service(self, name: str) -> None:
        """Bring a service to ``active`` (creates entry if missing)."""
        svc = self.services.setdefault(name, {})
        svc["status"] = "active"
        svc["exit_code"] = 0
        self._log("start_service", name=name)

    def restart_service(self, name: str) -> None:
        """Cycle a service to ``active`` and increment ``restart_count``."""
        svc = self.services.setdefault(name, {})
        svc["status"] = "active"
        svc["exit_code"] = 0
        svc["restart_count"] = svc.get("restart_count", 0) + 1
        self._log("restart_service", name=name)

    def chmod(self, path: str, mode: int) -> None:
        """Update a file's mode bits (octal int, e.g. 0o600)."""
        self.file_modes[path] = mode
        self._log("chmod", path=path, mode=mode)

    def chown(self, path: str, owner: str) -> None:
        """Update a file's owner."""
        self.file_owners[path] = owner
        self._log("chown", path=path, owner=owner)

    @property
    def mutation_count(self) -> int:
        """Total mutating operations in this episode (used by reward_minimality)."""
        return len(self.mutation_log)

    @property
    def mutated_paths(self) -> set[str]:
        """Distinct file paths touched by any mutation (audit + minimality)."""
        return {
            entry["args"]["path"]
            for entry in self.mutation_log
            if "path" in entry.get("args", {})
        }
