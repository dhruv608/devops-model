"""Scenario loader, dataclass, and train/eval splitter.

A scenario is one incident the agent will face: an initial system state,
the human-readable problem statement (``incident_text``), a list of
``success_predicate`` checks the env runs after ``submit_fix``, and a
``fragile_state`` block listing things the safe fix must NOT touch.

The schema is documented in [strategy.md sec 3.1](../../plans/strategy.md).
JSON has no octal literals, so ``file_modes`` use decimal integers
(``420`` = ``0o644``, ``384`` = ``0o600``, ``432`` = ``0o660``).

Adversarial scenarios use an ``adv_`` id prefix so ``train_eval_split``
can route them to held-out evaluation without leaking into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Categories that match strategy.md sec 3.1. Anything outside this set is a
# typo in scenario authoring -- the loader rejects it loudly so a bad
# JSON edit never silently lands in training.
VALID_CATEGORIES = frozenset(
    {
        "service_failure",
        "disk_full",
        "process_runaway",
        "permissions",
        "network",
        "db_recovery",
    }
)

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


# Predicate types the Hour 7 reward evaluator dispatches on. Adding a new
# type here without adding the matching evaluator branch will silently
# fail correctness rewards, so this set is the contract -- keep it tight.
VALID_PREDICATE_TYPES = frozenset(
    {
        "service_status",
        "process_killed",
        "port_freed",
        "file_exists",
        "file_not_exists",
        "file_content_contains",
        "file_content_not_contains",
        "file_mode",
        "file_owner",
    }
)


@dataclass
class Scenario:
    """One incident as loaded from JSON. Frozen at load time, copied per
    episode (the env builds a fresh ``SimulatedSystem`` on every reset)."""

    id: str
    category: str
    difficulty: str
    incident_text: str
    initial_state: dict[str, Any]
    success_predicate: list[dict[str, Any]]
    fragile_state: dict[str, Any]
    required_mutations: int = 0
    expected_kill_pids: list[int] = field(default_factory=list)
    # Author-only debug field; never sent to the model.
    safe_fix_hint: str = ""

    @property
    def is_adversarial(self) -> bool:
        return self.id.startswith("adv_")


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Read a JSON file of scenario records and return Scenario objects.

    Raises ValueError on schema violations -- bad data crashes loudly so it
    can't drift into a training run.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected top-level JSON list, got {type(raw).__name__}")

    seen_ids: set[str] = set()
    scenarios: list[Scenario] = []

    for i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            raise ValueError(f"{path}#{i}: scenario record must be an object")

        scenario = _build_scenario(rec, source=f"{path}#{i}")

        if scenario.id in seen_ids:
            raise ValueError(f"{path}#{i}: duplicate id {scenario.id!r}")
        seen_ids.add(scenario.id)
        scenarios.append(scenario)

    return scenarios


def train_eval_split(
    scenarios: list[Scenario],
    seed: int | None = None,
) -> tuple[list[Scenario], list[Scenario]]:
    """Partition scenarios by id prefix.

    Anything starting with ``adv_`` lands in eval; everything else is
    train. ``seed`` is accepted for API stability with the playbook's
    ``train_eval_split(seed)`` signature but the split is deterministic
    by id, not random -- the only safe way to keep adversarials held out.
    """
    train: list[Scenario] = []
    eval_: list[Scenario] = []
    for s in scenarios:
        (eval_ if s.is_adversarial else train).append(s)
    return train, eval_


def _build_scenario(rec: dict[str, Any], source: str) -> Scenario:
    """Validate one record and return a Scenario."""
    for req in ("id", "category", "difficulty", "incident_text",
                "initial_state", "success_predicate", "fragile_state"):
        if req not in rec:
            raise ValueError(f"{source}: missing required field {req!r}")

    cat = rec["category"]
    if cat not in VALID_CATEGORIES:
        raise ValueError(f"{source}: category {cat!r} not in {sorted(VALID_CATEGORIES)}")

    diff = rec["difficulty"]
    if diff not in VALID_DIFFICULTIES:
        raise ValueError(f"{source}: difficulty {diff!r} not in {sorted(VALID_DIFFICULTIES)}")

    initial = rec["initial_state"]
    if not isinstance(initial, dict):
        raise ValueError(f"{source}: initial_state must be an object")

    preds = rec["success_predicate"]
    if not isinstance(preds, list) or not preds:
        raise ValueError(f"{source}: success_predicate must be a non-empty list")
    for j, p in enumerate(preds):
        if not isinstance(p, dict) or "type" not in p:
            raise ValueError(f"{source} predicate#{j}: must be an object with a 'type' key")
        if p["type"] not in VALID_PREDICATE_TYPES:
            raise ValueError(
                f"{source} predicate#{j}: type {p['type']!r} not in "
                f"{sorted(VALID_PREDICATE_TYPES)}"
            )

    fragile = rec["fragile_state"]
    if not isinstance(fragile, dict):
        raise ValueError(f"{source}: fragile_state must be an object")
    if "untouchable_paths" not in fragile:
        raise ValueError(f"{source}: fragile_state.untouchable_paths is required")
    if not isinstance(fragile["untouchable_paths"], list):
        raise ValueError(f"{source}: fragile_state.untouchable_paths must be a list")
    fragile.setdefault("databases", [])

    return Scenario(
        id=rec["id"],
        category=cat,
        difficulty=diff,
        incident_text=rec["incident_text"],
        initial_state=initial,
        success_predicate=preds,
        fragile_state=fragile,
        required_mutations=int(rec.get("required_mutations", 0)),
        expected_kill_pids=list(rec.get("expected_kill_pids", [])),
        safe_fix_hint=rec.get("safe_fix_hint", ""),
    )
