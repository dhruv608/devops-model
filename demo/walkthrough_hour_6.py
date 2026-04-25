"""Hour 6 walkthrough: full investigate -> fix -> submit on scenario 0.

Run: ``PYTHONPATH=. python demo/walkthrough_hour_6.py``

Unlike walkthrough_hour_4 (read-only + stub submit), this one drives the
agent loop end-to-end: read a couple of investigation tools, then run the
real fix script through execute_bash, then submit_fix. Used to eyeball
the playbook Hour 6 CHECK (`services.nginx.status == 'active'` after).
"""

from __future__ import annotations

from models import SafeSreAction
from server.safe_sre_env_environment import SafeSreEnvironment


def show(label: str, obs) -> None:
    print(f"\n--- after {label} ---")
    print(f"  done={obs.done}  reward={obs.reward}  turn={obs.turn_count}")
    body = obs.stdout if len(obs.stdout) < 600 else obs.stdout[:580] + " ...[truncated]"
    for line in body.splitlines() or [""]:
        print(f"  | {line}")
    if obs.stderr:
        print(f"  stderr: {obs.stderr}")


def main() -> None:
    env = SafeSreEnvironment()
    env.reset(seed=0)

    print(f"=== scenario {env.scenario.id} ({env.scenario.category}) ===")
    print(f"  pre-fix: nginx.status = {env.system.services['nginx']['status']}")
    print(f"  pre-fix: ports = {env.system.ports}")

    show(
        "list_ports",
        env.step(SafeSreAction(tool="list_ports")),
    )
    show(
        "read_log /var/log/nginx/error.log",
        env.step(
            SafeSreAction(
                tool="read_log",
                args={"path": "/var/log/nginx/error.log"},
            )
        ),
    )
    show(
        "execute_bash 'kill -9 4051 && systemctl restart nginx'",
        env.step(
            SafeSreAction(
                tool="execute_bash",
                args={"script": "kill -9 4051 && systemctl restart nginx"},
            )
        ),
    )

    print("\n--- mid-fix system state ---")
    print(f"  nginx.status = {env.system.services['nginx']['status']}")
    print(f"  ports        = {env.system.ports}")
    print(f"  mutations    = {[e['op'] for e in env.system.mutation_log]}")

    show(
        "submit_fix",
        env.step(
            SafeSreAction(
                tool="submit_fix",
                args={
                    "claim": (
                        "PID 4051 (rogue python http.server) was holding port 80. "
                        "Killed it and restarted nginx; nginx is now active on 80."
                    )
                },
            )
        ),
    )

    print("\n=== EPISODE SUMMARY ===")
    print(f"  terminated      = {env._terminated}")
    print(f"  total turns     = {env._turn_count}")
    print(f"  mutations       = {env.system.mutation_count}")
    print(f"  catastrophic    = {len(env._catastrophic_commands)}")
    print(f"  risky           = {len(env._risky_commands)}")
    print(f"  parse_failures  = {env._parse_failures}")
    print("  reward attrs (filled at Hour 7):")
    for attr in (
        "safety_reward",
        "correctness_reward",
        "minimality_reward",
        "format_reward",
        "investigation_reward",
    ):
        print(f"    {attr:<22} = {getattr(env, attr)}")


if __name__ == "__main__":
    main()
