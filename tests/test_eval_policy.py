"""Tests for checkpoint-or-Nash evaluation targets."""

from __future__ import annotations

import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from roll.utils.eval_policy import (  # noqa: E402
    PolicySpecError,
    assign_components,
    combine_metric,
    nash_mixture,
    parse_policy_spec,
    resolve,
)


def _state():
    """A 3x3 game whose equilibrium is a genuine mixture, not a pure strategy."""
    population = {f"A{i}": {"path": f"/ck/A{i}"} for i in range(3)}
    population.update({f"D{i}": {"path": f"/ck/D{i}"} for i in range(3)})
    population["D0"]["path"] = None          # base model carries no adapter
    # rock-paper-scissors payoffs -> uniform 1/3 mixture on both sides
    values = {
        ("A0", "D0"): 0.0, ("A0", "D1"): -1.0, ("A0", "D2"): 1.0,
        ("A1", "D0"): 1.0, ("A1", "D1"): 0.0, ("A1", "D2"): -1.0,
        ("A2", "D0"): -1.0, ("A2", "D1"): 1.0, ("A2", "D2"): 0.0,
    }
    cells = {f"{a}__{d}": {"value": v} for (a, d), v in values.items()}
    return {"population": population, "cells": cells}


# --- spec parsing -----------------------------------------------------------

def test_label_path_and_nash_specs():
    assert parse_policy_spec("D5") == {"kind": "checkpoint", "label": "D5", "path": None}
    assert parse_policy_spec("/ck/D5")["path"] == "/ck/D5"
    assert parse_policy_spec("nash:defender") == {
        "kind": "nash", "role": "defender", "generation": None
    }
    assert parse_policy_spec("nash:attacker@5")["generation"] == 5


def test_bad_specs_are_rejected():
    with pytest.raises(PolicySpecError):
        parse_policy_spec("")
    # a role we do not have is a checkpoint label, and must fail at resolve time
    with pytest.raises(PolicySpecError):
        resolve("nope", _state())


# --- mixtures ---------------------------------------------------------------

def test_nash_mixture_is_the_real_equilibrium():
    support = nash_mixture(_state(), "defender")
    assert {c["label"] for c in support} == {"D0", "D1", "D2"}
    for component in support:
        assert component["probability"] == pytest.approx(1 / 3, abs=1e-6)
    assert sum(c["probability"] for c in support) == pytest.approx(1.0)


def test_mixture_can_be_taken_at_an_earlier_generation():
    # generation 1 -> leading 2x2 submatrix only
    support = nash_mixture(_state(), "defender", generation=1)
    assert {c["label"] for c in support} <= {"D0", "D1"}


def test_missing_cell_is_an_explicit_error_not_a_silent_mixture():
    state = _state()
    del state["cells"]["A2__D1"]
    with pytest.raises(PolicySpecError, match="missing"):
        nash_mixture(state, "defender")


def test_base_component_keeps_its_none_path():
    support = nash_mixture(_state(), "defender")
    base = [c for c in support if c["label"] == "D0"][0]
    assert base["path"] is None          # serving side must treat this as base


def test_resolve_checkpoint_looks_up_the_population_path():
    resolved = resolve("D1", _state())
    assert resolved["kind"] == "checkpoint"
    assert resolved["components"] == [
        {"label": "D1", "path": "/ck/D1", "probability": 1.0}
    ]


# --- expectation mode -------------------------------------------------------

def test_combine_metric_is_the_weighted_mean():
    components = [
        {"label": "D1", "probability": 0.6},
        {"label": "D2", "probability": 0.4},
    ]
    assert combine_metric({"D1": 0.10, "D2": 0.20}, components) == pytest.approx(0.14)


def test_combine_metric_refuses_to_guess_a_missing_component():
    components = [
        {"label": "D1", "probability": 0.5},
        {"label": "D2", "probability": 0.5},
    ]
    with pytest.raises(PolicySpecError, match="D2"):
        combine_metric({"D1": 0.1}, components)


def test_single_component_expectation_equals_the_checkpoint_metric():
    resolved = resolve("D1", _state())
    assert combine_metric({"D1": 0.42}, resolved["components"]) == pytest.approx(0.42)


# --- sample mode ------------------------------------------------------------

def test_assignment_is_deterministic_and_reproducible():
    components = [
        {"label": "D1", "probability": 0.5},
        {"label": "D2", "probability": 0.5},
    ]
    first = assign_components(200, components)
    assert first == assign_components(200, components)
    # a prefix of a longer run matches the shorter run: parallel/partial
    # evaluation cannot shift the assignment
    assert assign_components(50, components) == first[:50]


def test_assignment_tracks_the_mixture_weights():
    components = [
        {"label": "D1", "probability": 0.75},
        {"label": "D2", "probability": 0.25},
    ]
    counts = collections.Counter(assign_components(4000, components))
    assert counts[0] / 4000 == pytest.approx(0.75, abs=0.02)


def test_different_seeds_give_different_assignments():
    components = [
        {"label": "D1", "probability": 0.5},
        {"label": "D2", "probability": 0.5},
    ]
    a = assign_components(100, components, seed="one")
    b = assign_components(100, components, seed="two")
    assert a != b


def test_assignment_requires_normalized_probabilities():
    with pytest.raises(PolicySpecError):
        assign_components(10, [{"label": "D1", "probability": 0.4}])
