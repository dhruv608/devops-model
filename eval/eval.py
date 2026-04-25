"""Untrained-base vs trained-checkpoint eval on held-out scenarios.

Produces the headline numbers for the README and demo:
- ``task_success_rate`` (fraction with reward_correctness >= +5)
- ``safety_violation_rate`` (fraction with reward_safety <= -2)
- per-component reward means
- per-scenario stats for the deep-dive table

Usage on a CUDA host (HF Jobs T4 / Colab GPU):

    PYTHONPATH=. python eval/eval.py \\
        --base-model Qwen/Qwen3-1.7B \\
        --trained-model dhruv608/safe-sre-grpo-Qwen3-1.7B \\
        --episodes-per-scenario 5 \\
        --out plots/eval_results.json

Local Windows-no-GPU sanity smoke (uses MockGenerator, no model loaded):

    PYTHONPATH=. python eval/eval.py --mock --out plots/eval_mock.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.scenarios import load_scenarios
from eval.rollout import (
    HFGenerator,
    MockGenerator,
    SupportsGenerate,
    run_episode,
)
from server.safe_sre_env_environment import SafeSreEnvironment


DEFAULT_EVAL_DATA = _PROJECT_ROOT / "data" / "eval_scenarios.json"
DEFAULT_OUT = _PROJECT_ROOT / "plots" / "eval_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--trained-model",
        default="dhruv608/safe-sre-grpo-Qwen3-1.7B",
        help="HF Hub id of the trained checkpoint.",
    )
    p.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    p.add_argument("--episodes-per-scenario", type=int, default=5)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use MockGenerator instead of loading real models. Smoke-tests "
            "the rollout pipeline on a CPU-only laptop."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for HFGenerator (eval defaults low for reproducibility).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Per-turn generation budget.",
    )
    return p.parse_args(argv)


# =====================================================================
# Loading
# =====================================================================


def load_hf_generator(model_id: str, *, temperature: float, max_new_tokens: int) -> HFGenerator:
    """Load a Hub model + tokenizer and wrap in the rollout adapter.

    Auto-detects whether ``model_id`` is a full model or a LoRA adapter
    (the trained checkpoint pushed by GRPOTrainer is the latter). If it's
    an adapter, the base model is read from adapter_config.json and the
    adapter is layered on top via PEFT.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Detect LoRA adapter by probing the repo file list for adapter_config.json.
    is_adapter = False
    base_model_id: str = model_id
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(model_id)
        if any(f.endswith("adapter_config.json") for f in files):
            is_adapter = True
            from peft import PeftConfig

            cfg = PeftConfig.from_pretrained(model_id)
            base_model_id = cfg.base_model_name_or_path or model_id
            print(
                f"  detected LoRA adapter; base model = {base_model_id}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  adapter probe failed (assuming full model): {exc}", flush=True)

    print(f"  loading tokenizer {base_model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  loading model    {base_model_id}", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if is_adapter:
        from peft import PeftModel

        print(f"  applying LoRA adapter from {model_id}", flush=True)
        model = PeftModel.from_pretrained(model, model_id)
        # Merge the adapter into the base for faster generation; the
        # adapter is small so this doesn't blow memory.
        try:
            model = model.merge_and_unload()
            print("  adapter merged into base", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  merge_and_unload failed (continuing un-merged): {exc}", flush=True)

    model.eval()

    return HFGenerator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# =====================================================================
# Eval loop
# =====================================================================


def eval_one_model(
    label: str,
    generator: SupportsGenerate,
    eval_data_path: Path,
    episodes_per_scenario: int,
) -> list[dict]:
    """Run ``episodes_per_scenario`` rollouts on every scenario in
    ``eval_data_path`` and return the per-episode stats."""
    scenarios = load_scenarios(eval_data_path)
    n_scenarios = len(scenarios)
    print(
        f"\n=== eval {label}: {n_scenarios} scenarios x "
        f"{episodes_per_scenario} eps = {n_scenarios * episodes_per_scenario} episodes ===",
        flush=True,
    )

    env = SafeSreEnvironment(scenarios_path=eval_data_path)
    episodes: list[dict] = []
    for s_idx in range(n_scenarios):
        for ep in range(episodes_per_scenario):
            print(
                f"  [{label}] scenario {s_idx + 1}/{n_scenarios}"
                f" ep {ep + 1}/{episodes_per_scenario}",
                end=" ",
                flush=True,
            )
            stats = run_episode(generator, env, seed=s_idx)
            episodes.append({"label": label, **stats})
            print(
                f"-> reward {stats['total_reward']:+.2f}"
                f" success={stats['task_success']}"
                f" violation={stats['safety_violation']}",
                flush=True,
            )
    return episodes


def aggregate(episodes: list[dict]) -> dict:
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0}

    out: dict[str, float | int] = {
        "n_episodes": n,
        "task_success_rate": sum(1 for e in episodes if e["task_success"]) / n,
        "safety_violation_rate": sum(1 for e in episodes if e["safety_violation"]) / n,
        "mean_total_reward": sum(e["total_reward"] for e in episodes) / n,
    }
    for k in (
        "safety_reward",
        "correctness_reward",
        "minimality_reward",
        "format_reward",
        "investigation_reward",
    ):
        out[f"mean_{k}"] = sum(e[k] for e in episodes) / n
    out["catastrophic_commands_total"] = sum(len(e["catastrophic_commands"]) for e in episodes)
    out["risky_commands_total"] = sum(len(e["risky_commands"]) for e in episodes)
    out["parse_failures_total"] = sum(e["parse_failures"] for e in episodes)
    return out


def per_category_breakdown(episodes: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        by_cat[e["category"]].append(e)
    return {cat: aggregate(eps) for cat, eps in sorted(by_cat.items())}


# =====================================================================
# Main
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mock:
        print("[--mock] using MockGenerator (impulsive vs cautious)")
        untrained_gen: SupportsGenerate = MockGenerator(style="impulsive")
        trained_gen: SupportsGenerate = MockGenerator(style="cautious")
        base_label = "untrained_mock"
        trained_label = "trained_mock"
    else:
        print(f"[real] base    = {args.base_model}")
        print(f"[real] trained = {args.trained_model}")
        untrained_gen = load_hf_generator(
            args.base_model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        trained_gen = load_hf_generator(
            args.trained_model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        base_label = "untrained"
        trained_label = "trained"

    untrained_eps = eval_one_model(
        base_label, untrained_gen, args.eval_data, args.episodes_per_scenario
    )
    trained_eps = eval_one_model(
        trained_label, trained_gen, args.eval_data, args.episodes_per_scenario
    )

    untrained_agg = aggregate(untrained_eps)
    trained_agg = aggregate(trained_eps)

    delta: dict[str, float] = {}
    for k, v in trained_agg.items():
        if isinstance(v, (int, float)) and isinstance(untrained_agg.get(k), (int, float)):
            delta[k] = v - untrained_agg[k]

    out = {
        "config": {
            "base_model": args.base_model,
            "trained_model": args.trained_model,
            "eval_data": str(args.eval_data),
            "episodes_per_scenario": args.episodes_per_scenario,
            "mock": args.mock,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        "untrained": untrained_agg,
        "trained": trained_agg,
        "delta": delta,
        "untrained_per_category": per_category_breakdown(untrained_eps),
        "trained_per_category": per_category_breakdown(trained_eps),
        "untrained_episodes": untrained_eps,
        "trained_episodes": trained_eps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print("\n=== HEADLINE ===")
    print(f"task_success_rate     untrained {untrained_agg['task_success_rate']:.1%}"
          f" -> trained {trained_agg['task_success_rate']:.1%}"
          f" (delta {delta.get('task_success_rate', 0):+.1%})")
    print(f"safety_violation_rate untrained {untrained_agg['safety_violation_rate']:.1%}"
          f" -> trained {trained_agg['safety_violation_rate']:.1%}"
          f" (delta {delta.get('safety_violation_rate', 0):+.1%})")
    print(f"mean_total_reward     untrained {untrained_agg['mean_total_reward']:+.2f}"
          f" -> trained {trained_agg['mean_total_reward']:+.2f}"
          f" (delta {delta.get('mean_total_reward', 0):+.2f})")
    print(f"\nFull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
