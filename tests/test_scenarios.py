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


TRAIN_JSON = Path(__file__).parent.parent / "data" / "train_scenarios.json"


def test_load_returns_15_scenarios() -> None:
    scenarios = load_scenarios(TRAIN_JSON)
    assert len(scenarios) == 15
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_category_distribution_matches_playbook_hour_2() -> None:
    """Playbook Hour 2: 3 service / 3 disk / 2 process / 2 perm / 2 net / 2 db
    plus 1 adversarial (which counts under disk_full per category but uses
    the adv_ id prefix). Lock the breakdown so future edits notice."""
    scenarios = load_scenarios(TRAIN_JSON)
    counts: dict[str, int] = {}
    for s in scenarios:
        counts[s.category] = counts.get(s.category, 0) + 1

    assert counts == {
        "service_failure": 3,
        "disk_full": 4,
        "process_runaway": 2,
        "permissions": 2,
        "network": 2,
        "db_recovery": 2,
    }
    # disk_full = 3 non-adv + 1 adv per playbook spec.
    advs = [s for s in scenarios if s.is_adversarial]
    assert len(advs) == 1
    assert advs[0].category == "disk_full"


def test_every_scenario_has_valid_predicate_types_and_fragile_block() -> None:
    """Per playbook Hour 2 step 3: assert every scenario has a valid
    initial_state, success_predicate, and fragile_state. Loader catches
    schema bugs; this test asserts the assertions actually fired."""
    scenarios = load_scenarios(TRAIN_JSON)
    for s in scenarios:
        assert s.category in VALID_CATEGORIES, s.id
        assert s.success_predicate, s.id
        for pred in s.success_predicate:
            assert pred["type"] in VALID_PREDICATE_TYPES, (s.id, pred)
        assert isinstance(s.fragile_state.get("untouchable_paths"), list), s.id
        # Every scenario should have a non-empty incident_text >= 30 chars
        # so the model has something to read.
        assert len(s.incident_text) >= 30, s.id


def test_every_initial_state_loads_into_simulated_system() -> None:
    """Round-trip every scenario through SimulatedSystem.from_initial.
    If any scenario's initial_state is malformed (port keys, file shapes,
    etc.) this is where it surfaces -- before it costs us a training run."""
    scenarios = load_scenarios(TRAIN_JSON)
    for s in scenarios:
        sys_ = SimulatedSystem.from_initial(s.initial_state)
        # No mutations on load.
        assert sys_.mutation_log == [], s.id
        # Sanity: at least one of files / services / processes is non-empty
        # for every scenario -- a totally blank initial state is a bug.
        assert sys_.files or sys_.services or sys_.processes, s.id


def test_train_eval_split_routes_adversarials_to_eval() -> None:
    scenarios = load_scenarios(TRAIN_JSON)
    train, eval_ = train_eval_split(scenarios, seed=0)
    # 14 train + 1 eval (the lone adv_).
    assert len(train) == 14
    assert len(eval_) == 1
    assert all(s.id.startswith("adv_") for s in eval_)
    assert all(not s.id.startswith("adv_") for s in train)
    # Disjoint.
    train_ids = {s.id for s in train}
    eval_ids = {s.id for s in eval_}
    assert train_ids.isdisjoint(eval_ids)


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
