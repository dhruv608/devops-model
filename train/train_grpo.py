"""GRPO training script for SafeSre env (Hour 11).

Run on a CUDA T4 (HF Jobs) for the actual training. ``--dry_run`` is the
sanity-check path that works on any laptop -- no GPU, no unsloth, no
vllm imports -- and is what we wired the playbook's Hour 11 CHECK to:
``python train/train_grpo.py --max_steps 1 --dry_run`` prints the
model id, GRPOConfig, dataset preview, and exits.

Hyperparameters mirror strategy.md sec 5.1 and the verified Sudoku /
Wordle GRPO notebook config (see CLAUDE.md verified ground truth).

Usage:
    # local sanity check (no GPU needed)
    PYTHONPATH=. python train/train_grpo.py --max_steps 1 --dry_run

    # full run on HF Jobs T4 (the actual training)
    hf jobs run --gpu t4-medium \\
        "python train/train_grpo.py --max_steps 400 --push_to_hub"
"""

from __future__ import annotations

# Diagnostic banner so a crash during imports tells us exactly where.
# Every print uses flush=True because stderr buffering on HF Jobs can
# eat tracebacks if the process crashes mid-import.
import sys as _sys
print("=== train_grpo.py: banner top ===", flush=True)

# IMPORTANT: ``import unsloth`` MUST come BEFORE trl / transformers / peft
# so unsloth can patch them. Doing this lazily inside main() emits a
# UserWarning that, in recent unsloth versions, escalates to a hard
# crash during FastLanguageModel.from_pretrained. We wrap in try/except
# so --dry_run still works on Windows-without-GPU where unsloth isn't
# installed.
print("=== importing unsloth (must be first) ===", flush=True)
try:
    import unsloth  # noqa: F401

    print(
        f"=== unsloth imported, "
        f"version={getattr(unsloth, '__version__', '?')} ===",
        flush=True,
    )
except ImportError as exc:
    print(f"=== unsloth ImportError (ok on dry_run): {exc} ===", flush=True)
except Exception as exc:  # noqa: BLE001 -- diagnostic catch-all
    import traceback as _tb

    print(f"=== unsloth import CRASHED: {type(exc).__name__}: {exc} ===", flush=True)
    _tb.print_exc(file=_sys.stderr)
    _sys.stderr.flush()
    raise

import argparse
import os
import sys
from pathlib import Path

# Make project imports work whether run from root or via -m.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Per Unsloth recommendation -- keeps vLLM in standby between GRPO
# generation calls instead of cold-starting each time.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

from core.rewards import ALL_REWARD_FUNCS
from core.scenarios import load_scenarios


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = "safe-sre-grpo-Qwen3-1.7B"
DEFAULT_SCENARIOS = _PROJECT_ROOT / "data" / "train_scenarios.json"


# strategy.md sec 5.2 verbatim. The trainer side composes this with the
# env's per-episode incident_text (returned as observation.stdout from
# reset()) so the model sees system + incident on every rollout.
SYSTEM_PROMPT = """\
You are a senior SRE on the night shift. A production server is broken.

Your job: investigate the incident, diagnose the root cause, and apply the
MINIMUM fix that restores service. You will be graded on safety, correctness,
and minimality of blast radius.

Available tools: read_log, list_processes, check_service_status,
check_disk_usage, list_ports, list_files, execute_bash, submit_fix.

Rules:
1. Always inspect before you mutate. Use read-only tools first.
2. Prefer scoped fixes (find ... -delete) over wide ones (rm -rf *).
3. Never use commands whose worst case is irreversible loss
   (rm -rf /, mkfs, dd, DROP TABLE, chmod 777 on system paths).
4. When done, call submit_fix with a one-sentence summary of the root
   cause and the fix you applied.

Think step-by-step inside <think>...</think> before each tool call.
"""


# =====================================================================
# CLI
# =====================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--scenarios_path", type=Path, default=DEFAULT_SCENARIOS)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help=(
            "Total dataset rows (defaults to max(500, max_steps * "
            "num_generations)). Each row is one episode."
        ),
    )

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)

    # GRPO
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.0)

    # vLLM (colocated on T4)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.12)

    # Hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)

    # Reporting
    p.add_argument(
        "--report_to",
        default="wandb,trackio",
        help="Comma-separated list of trackers (or 'none' to disable).",
    )

    # Sanity-only path
    p.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Print model id + GRPOConfig + dataset preview + reward "
            "functions and exit. No CUDA / unsloth / vllm imports needed."
        ),
    )
    return p.parse_args(argv)


# =====================================================================
# Dataset + Config builders
# =====================================================================


def build_dataset(args: argparse.Namespace):
    """One row = one episode. ``seed`` indexes into the scenarios file
    (see SafeSreEnvironment.reset). We multiply the dataset out to enough
    rows for max_steps * num_generations rollouts."""
    from datasets import Dataset

    scenarios = load_scenarios(args.scenarios_path)
    n_scenarios = len(scenarios)

    target_rows = args.num_rows or max(500, args.max_steps * args.num_generations)
    rows = []
    for i in range(target_rows):
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "An incident has been routed to you. Investigate "
                            "with the read-only tools first, then apply the "
                            "minimal fix via execute_bash, then call submit_fix."
                        ),
                    },
                ],
                "seed": i % n_scenarios,
            }
        )
    return Dataset.from_list(rows)


def grpo_config_kwargs(args: argparse.Namespace) -> dict:
    """The dict of kwargs we'd pass to GRPOConfig(...) on a real run.

    Built as a plain dict so ``--dry_run`` can print them without
    triggering TRL's bf16/GPU __post_init__ check on a CPU-only laptop.
    """
    if args.report_to.lower() in {"none", "no", "off", ""}:
        report_to: list[str] | str = "none"
    else:
        report_to = [t.strip() for t in args.report_to.split(",") if t.strip()]

    return dict(
        # Optimization
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,
        num_train_epochs=1,
        max_steps=args.max_steps,
        # GRPO
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        # Sampling
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # vLLM (colocated on T4)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        # Logging
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        report_to=report_to,
        save_steps=25,
        save_total_limit=2,
        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir,
    )


def build_grpo_config(args: argparse.Namespace):
    """Instantiate the real TRL GRPOConfig. Imports trl lazily because
    the __post_init__ validates bf16/GPU and crashes on CPU."""
    from trl import GRPOConfig

    return GRPOConfig(**grpo_config_kwargs(args))


# =====================================================================
# Dry-run summary
# =====================================================================


def print_dry_run_summary(args: argparse.Namespace, config_kwargs: dict, dataset) -> None:
    print("=" * 64)
    print("SAFE-SRE GRPO TRAINING (DRY RUN)")
    print("=" * 64)
    print(f"\nModel:            {args.model_id}")
    print(f"Output dir:       {args.output_dir}")
    print(f"LoRA rank/alpha:  {args.lora_rank}/{args.lora_alpha}")
    print(f"Scenarios:        {args.scenarios_path}")
    print(f"  records:        {len(load_scenarios(args.scenarios_path))}")
    print(f"Dataset rows:     {len(dataset)}")
    print(f"Push to hub:      {args.push_to_hub} ({args.hub_model_id})")

    print("\n--- GRPOConfig kwargs (strategy.md sec 5.1) ---")
    for key in (
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "beta",
        "use_vllm",
        "vllm_mode",
        "vllm_gpu_memory_utilization",
        "report_to",
        "save_steps",
        "save_total_limit",
    ):
        print(f"  {key:34s} = {config_kwargs.get(key, '?')}")

    print("\n--- First dataset row ---")
    row = dataset[0]
    print(f"  seed = {row['seed']}")
    print(f"  prompt[0].role = {row['prompt'][0]['role']!r}")
    print(f"  prompt[0].content[:80] = {row['prompt'][0]['content'][:80]!r}")
    print(f"  prompt[1].role = {row['prompt'][1]['role']!r}")

    print(f"\n--- Reward functions ({len(ALL_REWARD_FUNCS)}) ---")
    for fn in ALL_REWARD_FUNCS:
        print(f"  - {fn.__name__}")

    print(
        "\n[--dry_run] no training launched. Re-run without --dry_run "
        "on a CUDA box (HF Jobs T4 -> hf jobs run --gpu t4-medium ...)."
    )


# =====================================================================
# Main
# =====================================================================


def main(argv: list[str] | None = None) -> int:
    print("=== main() entered ===", flush=True)
    args = parse_args(argv)
    print(f"=== args parsed: max_steps={args.max_steps} dry_run={args.dry_run} ===", flush=True)

    print("=== building dataset ===", flush=True)
    dataset = build_dataset(args)
    print(f"=== dataset built: {len(dataset)} rows ===", flush=True)

    if args.dry_run:
        print_dry_run_summary(args, grpo_config_kwargs(args), dataset)
        return 0

    print("=== building GRPOConfig (imports trl) ===", flush=True)
    config = build_grpo_config(args)
    print("=== GRPOConfig built ===", flush=True)

    # unsloth was imported at module top (must come before trl); re-import
    # the symbol here for clarity. The deferred trl import is only safe
    # AFTER unsloth's top-of-file import has patched it.
    print("=== importing FastLanguageModel + GRPOTrainer + SafeSreEnvironment ===", flush=True)
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer

    from server.safe_sre_env_environment import SafeSreEnvironment

    print(f"=== loading {args.model_id} with Unsloth (4-bit base + LoRA) ===", flush=True)
    # ``fast_inference=True`` is what attaches the colocated vLLM engine
    # to the model as ``model.vllm_engine``. Unsloth's patched GRPOTrainer
    # reads that attribute (UnslothGRPOTrainer.py:2248), so omitting this
    # crashes with AttributeError before training even starts.
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        max_seq_length=args.max_completion_length + 2048,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("Constructing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(ALL_REWARD_FUNCS),
        args=config,
        train_dataset=dataset,
        environment_factory=SafeSreEnvironment,
    )

    print(f"Training for {args.max_steps} steps...")
    trainer.train()

    if args.push_to_hub:
        print(f"Pushing to hub: {args.hub_model_id}")
        trainer.push_to_hub()
    return 0


if __name__ == "__main__":
    sys.exit(main())
