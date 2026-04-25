"""End-to-end tests for SafeSreEnvironment (Hour 3).

Cover the reset() contract and each of the 6 read-only tools as both a
direct method call and a step()-dispatched call. step()'s mutating-tool
branches stay stubbed at this hour -- we just assert they don't crash and
return a clearly-flagged not-implemented stderr so a future regression
silently swallowing them is detected.
"""

from __future__ import annotations

from models import SafeSreAction
from server.safe_sre_env_environment import (
    ALL_TOOLS,
    READ_ONLY_TOOLS,
    SafeSreEnvironment,
)


def _new_env() -> SafeSreEnvironment:
    return SafeSreEnvironment()


def test_reset_loads_scenario_zero_by_default() -> None:
    env = _new_env()
    obs = env.reset(seed=0)

    assert obs.done is False
    assert obs.reward == 0.0
    assert obs.turn_count == 0
    assert env.scenario.id == "nginx_port_conflict_001"
    # The available_tools list flowing into the agent must include every
    # tool the env exposes (read-only + execute_bash + submit_fix) so the
    # system prompt the trainer composes lists them all.
    assert obs.metadata["available_tools"] == list(ALL_TOOLS)
    # Sanity: the read-only set is a strict subset of all tools.
    assert set(READ_ONLY_TOOLS).issubset(set(ALL_TOOLS))
    # Initial observation should mention the scenario id and category.
    assert "nginx_port_conflict_001" in obs.stdout
    assert "service_failure" in obs.stdout


def test_reset_seed_indexes_scenarios_deterministically() -> None:
    env = _new_env()
    seeds_to_ids: dict[int, str] = {}
    for seed in range(15):
        env.reset(seed=seed)
        seeds_to_ids[seed] = env.scenario.id
    # 15 seeds, 15 distinct scenarios.
    assert len(set(seeds_to_ids.values())) == 15


def test_read_log_returns_existing_file_content() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.read_log("/var/log/nginx/error.log")
    assert "Address already in use" in out


def test_read_log_missing_file_returns_cat_error() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.read_log("/var/log/does_not_exist.log")
    assert "No such file" in out


def test_list_processes_shows_rogue_python_for_scenario_zero() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.list_processes()
    assert "python -m http.server 80" in out
    assert "4051" in out
    assert "4099" in out  # sshd also present


def test_list_processes_filter_narrows_output() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.list_processes(filter="sshd")
    assert "sshd" in out
    assert "python" not in out


def test_check_service_status_reports_failed_and_active() -> None:
    env = _new_env()
    env.reset(seed=0)
    nginx = env.check_service_status("nginx")
    sshd = env.check_service_status("sshd")
    assert "failed" in nginx
    assert "active" in sshd


def test_check_service_status_unknown_service_returns_not_found() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.check_service_status("not_a_service")
    assert "could not be found" in out


def test_list_ports_surfaces_pid_and_cmd() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.list_ports()
    assert "80" in out
    assert "4051" in out
    assert "python" in out


def test_list_files_filters_by_path_prefix() -> None:
    env = _new_env()
    env.reset(seed=0)
    out = env.list_files("/var/log")
    assert "/var/log/nginx/error.log" in out
    # Path under /etc shouldn't appear under /var/log filter.
    assert "/etc" not in out


def test_step_dispatches_read_only_tools_by_name() -> None:
    env = _new_env()
    env.reset(seed=0)
    obs = env.step(SafeSreAction(tool="read_log", args={"path": "/var/log/nginx/error.log"}))
    assert obs.done is False
    assert obs.turn_count == 1
    assert "Address already in use" in obs.stdout


def test_step_with_unknown_tool_returns_stderr_no_exception() -> None:
    env = _new_env()
    env.reset(seed=0)
    obs = env.step(SafeSreAction(tool="rm_rf_slash", args={}))
    assert obs.stderr
    assert "unknown tool" in obs.stderr.lower()
    assert obs.turn_count == 1


def test_step_execute_bash_still_stub_until_hour_6() -> None:
    """execute_bash arrives at Hour 6. Until then it must return a clear
    not-implemented stderr so a missed wiring is loud."""
    env = _new_env()
    env.reset(seed=0)
    obs = env.step(SafeSreAction(tool="execute_bash", args={"script": "echo hi"}))
    assert "not yet implemented" in obs.stderr.lower()


def test_state_property_advances_step_count() -> None:
    env = _new_env()
    env.reset(seed=0)
    assert env.state.step_count == 0
    env.step(SafeSreAction(tool="list_ports"))
    env.step(SafeSreAction(tool="list_processes"))
    assert env.state.step_count == 2


def test_submit_fix_terminates_episode_with_zero_placeholder_reward() -> None:
    env = _new_env()
    env.reset(seed=0)
    env.step(SafeSreAction(tool="list_processes"))  # one investigation step
    obs = env.step(SafeSreAction(tool="submit_fix", args={"claim": "killed pid 4051"}))

    assert obs.done is True
    assert env._terminated is True
    assert env._claim == "killed pid 4051"
    # Hour 4 placeholder: all 5 reward attrs remain 0; total = 0.
    assert obs.reward == 0.0
    assert env._total_reward() == 0.0
    # All five reward attrs exist (Hour 7 will fill values; smoke train
    # would AttributeError without these).
    for attr in ("safety_reward", "correctness_reward",
                 "minimality_reward", "format_reward", "investigation_reward"):
        assert hasattr(env, attr), attr


def test_step_after_termination_is_idempotent() -> None:
    env = _new_env()
    env.reset(seed=0)
    env.step(SafeSreAction(tool="submit_fix", args={"claim": "done"}))
    turn_before = env._turn_count
    obs2 = env.step(SafeSreAction(tool="list_processes"))
    # No turn advance, still terminal, reward unchanged.
    assert env._turn_count == turn_before
    assert obs2.done is True


def test_first_action_was_read_only_tracks_correctly() -> None:
    env = _new_env()
    env.reset(seed=0)
    env.step(SafeSreAction(tool="list_processes"))
    assert env._first_action_was_read_only is True

    env2 = _new_env()
    env2.reset(seed=0)
    env2.step(SafeSreAction(tool="execute_bash", args={"script": "anything"}))
    # execute_bash is mutating (even though stubbed at Hour 4, the ordering
    # signal is what reward_investigation uses).
    assert env2._first_action_was_read_only is False


def test_turn_limit_auto_terminates_at_max_turns() -> None:
    """Twelve calls succeed; the 12th completes and flags done; further
    calls return terminal state idempotently."""
    from server.safe_sre_env_environment import MAX_TURNS

    env = _new_env()
    env.reset(seed=0)

    # 11 read-only tool calls do not yet trigger termination.
    for _ in range(MAX_TURNS - 1):
        obs = env.step(SafeSreAction(tool="list_ports"))
        assert obs.done is False

    # 12th call triggers the limit branch and terminates.
    obs = env.step(SafeSreAction(tool="list_ports"))
    assert obs.done is True
    assert env._terminated is True
    assert "turn limit" in obs.stderr.lower()
    assert env._turn_count == MAX_TURNS
