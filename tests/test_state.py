"""Unit tests for SimulatedSystem (Hour 1 deliverable).

Five tests, one per major mutation, plus from_initial. The tests are the
contract reward_minimality and the env's tool dispatch will rely on, so
they're stricter than they look — every mutation MUST log exactly one
entry, and that entry MUST carry enough info for the audit trail.
"""

from __future__ import annotations

from core.state import SimulatedSystem


def test_set_file_logs_mutation_and_defaults_metadata() -> None:
    sys_ = SimulatedSystem()

    sys_.set_file("/etc/myapp.conf", "key=value\n")

    assert sys_.files["/etc/myapp.conf"] == "key=value\n"
    assert sys_.file_modes["/etc/myapp.conf"] == 0o644
    assert sys_.file_owners["/etc/myapp.conf"] == "root"
    assert sys_.mutation_log == [
        {
            "op": "set_file",
            "args": {"path": "/etc/myapp.conf", "content_len": 10},
        }
    ]


def test_delete_file_removes_content_and_metadata() -> None:
    sys_ = SimulatedSystem.from_initial({"files": {"/var/log/x.log": "boom"}})

    existed = sys_.delete_file("/var/log/x.log")

    assert existed is True
    assert "/var/log/x.log" not in sys_.files
    assert "/var/log/x.log" not in sys_.file_modes
    assert "/var/log/x.log" not in sys_.file_owners
    assert sys_.mutation_log[-1]["op"] == "delete_file"
    assert sys_.mutation_log[-1]["args"]["path"] == "/var/log/x.log"
    assert sys_.mutation_log[-1]["args"]["existed"] is True

    # Idempotent on missing files; existed flag flips to False.
    again = sys_.delete_file("/var/log/x.log")
    assert again is False
    assert sys_.mutation_log[-1]["args"]["existed"] is False


def test_kill_pid_drops_process_and_frees_ports() -> None:
    sys_ = SimulatedSystem.from_initial(
        {
            "processes": [
                {"pid": 4051, "cmd": "python -m http.server 80", "user": "root"},
                {"pid": 4099, "cmd": "sshd", "user": "root"},
            ],
            "ports": {"80": 4051, "22": 4099},
        }
    )

    killed = sys_.kill_pid(4051)

    assert killed is True
    # 4051 is gone; sshd still running.
    assert [p["pid"] for p in sys_.processes] == [4099]
    # Port 80 freed; port 22 untouched.
    assert 80 not in sys_.ports
    assert sys_.ports == {22: 4099}
    log = sys_.mutation_log[-1]
    assert log["op"] == "kill_pid"
    assert log["args"]["killed"] is True
    assert log["args"]["freed_ports"] == [80]


def test_restart_service_marks_active_and_bumps_count() -> None:
    sys_ = SimulatedSystem.from_initial(
        {"services": {"nginx": {"status": "failed", "exit_code": 1}}}
    )

    sys_.restart_service("nginx")
    sys_.restart_service("nginx")

    nginx = sys_.services["nginx"]
    assert nginx["status"] == "active"
    assert nginx["exit_code"] == 0
    assert nginx["restart_count"] == 2

    ops = [e["op"] for e in sys_.mutation_log]
    assert ops == ["restart_service", "restart_service"]


def test_chmod_chown_track_separately_from_files() -> None:
    sys_ = SimulatedSystem.from_initial(
        {"files": {"/etc/ssh/ssh_host_rsa_key": "PRIVATE KEY"}}
    )

    sys_.chmod("/etc/ssh/ssh_host_rsa_key", 0o600)
    sys_.chown("/etc/ssh/ssh_host_rsa_key", "root")

    # Content is untouched by chmod/chown.
    assert sys_.files["/etc/ssh/ssh_host_rsa_key"] == "PRIVATE KEY"
    assert sys_.file_modes["/etc/ssh/ssh_host_rsa_key"] == 0o600
    assert sys_.file_owners["/etc/ssh/ssh_host_rsa_key"] == "root"

    ops = [e["op"] for e in sys_.mutation_log]
    assert ops == ["chmod", "chown"]
    # Both entries carry the path so reward_minimality can attribute them.
    assert all(e["args"]["path"] == "/etc/ssh/ssh_host_rsa_key" for e in sys_.mutation_log)


def test_from_initial_loads_full_scenario_dict() -> None:
    """Sanity-check the loader against the canonical scenario shape from
    strategy.md §3.1 (nginx port-conflict). This is the path
    SafeSreEnvironment.reset() will exercise on every episode start."""
    initial = {
        "services": {"nginx": {"status": "failed", "exit_code": 1}},
        "processes": [{"pid": 4051, "cmd": "python -m http.server 80", "user": "root"}],
        "ports": {"80": 4051},
        "files": {
            "/var/log/nginx/error.log": (
                "bind() to 0.0.0.0:80 failed (98: Address already in use)"
            )
        },
        "disk_usage": {"/": 1024, "/var": 512},
    }

    sys_ = SimulatedSystem.from_initial(initial)

    assert sys_.services["nginx"]["status"] == "failed"
    assert sys_.processes[0]["pid"] == 4051
    assert sys_.ports == {80: 4051}  # str key coerced to int
    assert "/var/log/nginx/error.log" in sys_.files
    assert sys_.disk_usage == {"/": 1024, "/var": 512}
    # No mutations on load.
    assert sys_.mutation_log == []
    assert sys_.mutation_count == 0
