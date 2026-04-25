"""Validate the 15 Hour-2 scenarios load cleanly and meet the strategy spec.

The schema validator in core/scenarios.py is meant to be paranoid -- these
tests both confirm it accepts our hand-written JSON and that it rejects
the most likely authoring mistakes (bad category, missing predicate type,
missing fragile_state). If you add a new scenario type or predicate, add
a test below or this file will go quiet exactly when you'd want a scream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.scenarios import (
    Scenario,
    VALID_CATEGORIES,
    VALID_PREDICATE_TYPES,
    load_scenarios,
    train_eval_split,
)
from core.state import SimulatedSystem


DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_JSON = DATA_DIR / "train_scenarios.json"
EVAL_JSON = DATA_DIR / "eval_scenarios.json"


def test_train_set_size_after_hour_10_expansion() -> None:
    scenarios = load_scenarios(TRAIN_JSON)
    assert len(scenarios) == 25
    assert all(isinstance(s, Scenario) for s in scenarios)
    # No adv_ ids in train: those live in eval (held-out) by design.
    assert all(not s.is_adversarial for s in scenarios)


def test_eval_set_size_and_adversarial_majority() -> None:
    scenarios = load_scenarios(EVAL_JSON)
    assert len(scenarios) == 8
    advs = [s for s in scenarios if s.is_adversarial]
    # Playbook Hour 10 spec: at least 6 adversarials in held-out eval.
    assert len(advs) >= 6


def test_train_eval_id_sets_disjoint() -> None:
    """Hour 10 contract: train and eval must not share any scenario id.
    A leaked scenario in training trivialises adversarial generalisation
    measurement (strategy.md sec 4 #11)."""
    train_ids = {s.id for s in load_scenarios(TRAIN_JSON)}
    eval_ids = {s.id for s in load_scenarios(EVAL_JSON)}
    assert train_ids.isdisjoint(eval_ids), train_ids & eval_ids


def test_category_coverage_across_train_set() -> None:
    """All 6 strategy categories must be represented in train so the
    agent doesn't get a category-shaped blind spot."""
    cats = {s.category for s in load_scenarios(TRAIN_JSON)}
    assert cats == {
        "service_failure",
        "disk_full",
        "process_runaway",
        "permissions",
        "network",
        "db_recovery",
    }


def test_every_scenario_has_valid_predicate_types_and_fragile_block() -> None:
    """Both train AND eval must satisfy the schema. Loader catches
    schema bugs; this asserts the assertions actually fired."""
    for path in (TRAIN_JSON, EVAL_JSON):
        for s in load_scenarios(path):
            assert s.category in VALID_CATEGORIES, (path, s.id)
            assert s.success_predicate, (path, s.id)
            for pred in s.success_predicate:
                assert pred["type"] in VALID_PREDICATE_TYPES, (path, s.id, pred)
            assert isinstance(s.fragile_state.get("untouchable_paths"), list), (path, s.id)
            assert len(s.incident_text) >= 30, (path, s.id)


def test_every_initial_state_loads_into_simulated_system() -> None:
    """Round-trip every scenario (train + eval) through
    SimulatedSystem.from_initial. Catches malformed initial_state
    before it costs us a training run."""
    for path in (TRAIN_JSON, EVAL_JSON):
        for s in load_scenarios(path):
            sys_ = SimulatedSystem.from_initial(s.initial_state)
            assert sys_.mutation_log == [], (path, s.id)
            # Sanity: at least one of files / services / processes is non-empty
            # for every scenario -- a totally blank initial state is a bug.
            assert sys_.files or sys_.services or sys_.processes, (path, s.id)


def test_train_eval_split_within_train_finds_no_advs() -> None:
    """train_scenarios.json is now strictly non-adversarial; the
    splitter run on it should produce all-train, zero-eval."""
    scenarios = load_scenarios(TRAIN_JSON)
    train, eval_ = train_eval_split(scenarios, seed=0)
    assert len(train) == 25
    assert len(eval_) == 0


def test_loader_rejects_unknown_category(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{
        "id": "x",
        "category": "not_a_category",
        "difficulty": "easy",
        "incident_text": "x" * 50,
        "initial_state": {"files": {"/x": "y"}},
        "success_predicate": [{"type": "file_exists", "path": "/x"}],
        "fragile_state": {"untouchable_paths": []},
    }]))

    with pytest.raises(ValueError, match="category"):
        load_scenarios(bad)


def test_loader_rejects_unknown_predicate_type(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{
        "id": "x",
        "category": "service_failure",
        "difficulty": "easy",
        "incident_text": "x" * 50,
        "initial_state": {"files": {"/x": "y"}},
        "success_predicate": [{"type": "service_telepathy", "service": "nginx"}],
        "fragile_state": {"untouchable_paths": []},
    }]))

    with pytest.raises(ValueError, match="predicate"):
        load_scenarios(bad)


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad = tmp_path / "dup.json"
    rec = {
        "id": "duplicate",
        "category": "service_failure",
        "difficulty": "easy",
        "incident_text": "x" * 50,
        "initial_state": {"files": {"/x": "y"}},
        "success_predicate": [{"type": "file_exists", "path": "/x"}],
        "fragile_state": {"untouchable_paths": []},
    }
    bad.write_text(json.dumps([rec, rec]))

    with pytest.raises(ValueError, match="duplicate id"):
        load_scenarios(bad)
