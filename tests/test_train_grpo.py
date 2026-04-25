"""Smoke tests for train/train_grpo.py (Hour 11).

The --dry_run path must work on any laptop (no GPU, no unsloth, no
vllm), because that's the playbook's Hour 11 CHECK and our local
sanity gate before submitting an HF Jobs T4 run.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# Add project root to sys.path so train/* imports work in pytest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from train.train_grpo import (
    SYSTEM_PROMPT,
    build_dataset,
    grpo_config_kwargs,
    main,
    parse_args,
)


def test_dry_run_exits_zero_and_prints_required_sections() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--max_steps", "1", "--dry_run"])
    out = buf.getvalue()
    assert rc == 0
    # Headline section present.
    assert "GRPO TRAINING (DRY RUN)" in out
    # Config and dataset sanity blocks present.
    assert "GRPOConfig kwargs" in out
    assert "Reward functions (5)" in out
    # Strategy sec 5.1 anchor values surface verbatim.
    assert "learning_rate" in out
    assert "5e-06" in out
    assert "num_generations" in out
    assert "vllm_gpu_memory_utilization" in out
    # All 5 reward functions enumerated.
    for fn in (
        "reward_safety",
        "reward_correctness",
        "reward_minimality",
        "reward_format",
        "reward_investigation",
    ):
        assert fn in out


def test_grpo_config_kwargs_match_strategy_section_5_1() -> None:
    """Lock the playbook's anchor hyperparameters so an accidental flag
    rename doesn't drift training away from the verified config."""
    args = parse_args(["--max_steps", "400"])
    cfg = grpo_config_kwargs(args)

    # Anchors from strategy.md sec 5.1 (verified Sudoku/Wordle baseline,
    # tuned for our shorter completions + more group variance).
    assert cfg["learning_rate"] == 5e-6
    assert cfg["per_device_train_batch_size"] == 1
    assert cfg["gradient_accumulation_steps"] == 32
    assert cfg["num_generations"] == 4
    assert cfg["max_completion_length"] == 2048
    assert cfg["temperature"] == 0.9
    assert cfg["top_p"] == 0.95
    assert cfg["beta"] == 0.0  # current TRL default; flip only if forgetting
    assert cfg["use_vllm"] is True
    assert cfg["vllm_mode"] == "colocate"
    assert cfg["vllm_gpu_memory_utilization"] == 0.12
    assert cfg["save_steps"] == 25


def test_build_dataset_default_size_scales_with_max_steps() -> None:
    args = parse_args(["--max_steps", "50"])
    ds = build_dataset(args)
    # Default rows = max(500, max_steps * num_generations) = max(500, 200) = 500.
    assert len(ds) == 500
    row = ds[0]
    assert "prompt" in row and "seed" in row
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][0]["content"] == SYSTEM_PROMPT


def test_build_dataset_explicit_num_rows_honoured() -> None:
    args = parse_args(["--max_steps", "5", "--num_rows", "30"])
    ds = build_dataset(args)
    assert len(ds) == 30


def test_dataset_seeds_cycle_through_scenarios() -> None:
    """Each row's seed is row_idx % len(scenarios). Confirms every
    scenario gets covered when num_rows >= n_scenarios."""
    args = parse_args(["--max_steps", "5", "--num_rows", "50"])
    ds = build_dataset(args)
    seeds = sorted({row["seed"] for row in ds})
    assert len(seeds) == 25  # all 25 train scenarios get hit
    assert seeds == list(range(25))


def test_report_to_none_disables_trackers() -> None:
    args = parse_args(["--max_steps", "1", "--report_to", "none"])
    cfg = grpo_config_kwargs(args)
    assert cfg["report_to"] == "none"


def test_report_to_csv_parses_to_list() -> None:
    args = parse_args(["--max_steps", "1", "--report_to", "wandb,trackio"])
    cfg = grpo_config_kwargs(args)
    assert cfg["report_to"] == ["wandb", "trackio"]
