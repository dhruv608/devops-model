"""End-to-end manual walkthrough on scenario 0 (Hour 4 deliverable).

Run: ``PYTHONPATH=. python demo/walkthrough_hour_4.py``

Walks the env through reset -> 3 read-only tool calls -> submit_fix and
prints every observation so we can eyeball the lifecycle. No mutating
tools yet (execute_bash arrives at Hour 6); the point here is to confirm
that reset/step/submit_fix and the 12-turn limit all behave as designed.
"""

from __future__ import annotations

from models import SafeSreAction
from server.safe_sre_env_environment import SafeSreEnvironment


def banner(title: str) -> None:
    print(f"\n{'=' * 6} {title} {'=' * 6}")


def show(label: str, obs) -> None:
    print(f"\n--- after {label} ---")
    print(f"  done       = {obs.done}")
    print(f"  reward     = {obs.reward}")
    print(f"  turn_count = {obs.turn_count}")
    if obs.stdout:
        # Indent for readability; truncate long blocks.
        body = obs.stdout if len(obs.stdout) < 600 else obs.stdout[:580] + " ...[truncated]"
        for line in body.splitlines():
            print(f"  | {line}")
    if obs.stderr:
        print(f"  stderr: {obs.stderr}")


def main() -> None:
    env = SafeSreEnvironment()

    banner("RESET seed=0")
    obs = env.reset(seed=0)
    print(f"  scenario_id = {env.scenario.id}")
    print(f"  category    = {env.scenario.category}")
    print(f"  difficulty  = {env.scenario.difficulty}")
    show("reset", obs)

    banner("TOOL: list_processes")
    obs = env.step(SafeSreAction(tool="list_processes"))
    show("list_processes", obs)

    banner("TOOL: list_ports")
    obs = env.step(SafeSreAction(tool="list_ports"))
    show("list_ports", obs)

    banner("TOOL: read_log /var/log/nginx/error.log")
    obs = env.step(
        SafeSreAction(tool="read_log", args={"path": "/var/log/nginx/error.log"})
    )
    show("read_log", obs)

    banner("TOOL: submit_fix")
    obs = env.step(
        SafeSreAction(
            tool="submit_fix",
            args={
                "claim": (
                    "PID 4051 (rogue python -m http.server) was holding port 80; "
                    "would kill -9 4051 and systemctl restart nginx."
                )
            },
        )
    )
    show("submit_fix", obs)

    banner("EPISODE SUMMARY")
    print(f"  terminated      = {env._terminated}")
    print(f"  claim           = {env._claim}")
    print(f"  total turns     = {env._turn_count}")
    print(f"  mutation count  = {env.system.mutation_count}")
    print("  reward attrs (Hour 4 placeholders, Hour 7 fills these in):")
    for attr in (
        "safety_reward",
        "correctness_reward",
        "minimality_reward",
        "format_reward",
        "investigation_reward",
    ):
        print(f"    {attr:<22} = {getattr(env, attr)}")
    print(f"  total_reward    = {env._total_reward()}")


if __name__ == "__main__":
    main()
