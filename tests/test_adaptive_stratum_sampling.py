"""Tests for retention-aware stratified candidate sampling."""

import pytest

from roll.utils.adaptive_stratum_sampling import (
    MIN_PLANNING_RETENTION,
    plan_candidate_wave,
    projected_candidate_totals,
    stratum_retention_estimate,
)
from roll.utils.upstream_v2_payoff import (
    assemble_valid_interleaved_prefix,
    assemble_valid_stratified_prefix,
    build_interleaved_episode_specs,
    build_stratified_episode_specs,
)

HARMFUL = [{"vanilla": f"harmful prompt {i}"} for i in range(64)]
BENIGN = [{"vanilla": f"benign prompt {i}"} for i in range(64)]


def test_retention_estimate_is_conservative_and_monotone():
    no_data = stratum_retention_estimate(0, 0)
    assert 0.0 < no_data < 0.5
    # More successes at the same rate tightens the bound upward.
    loose = stratum_retention_estimate(6, 10)
    tight = stratum_retention_estimate(600, 1000)
    assert loose < tight < 0.62
    assert stratum_retention_estimate(0, 100) < 0.05


def test_retention_estimate_rejects_impossible_counts():
    with pytest.raises(ValueError):
        stratum_retention_estimate(5, 4)


def test_completed_stratum_receives_no_candidates():
    plan = plan_candidate_wave(
        deficits={"harmful": 0, "benign": 300},
        judged={"harmful": 2000, "benign": 3000},
        retained={"harmful": 2000, "benign": 1700},
        wave_budget=512,
        remaining_budget=10_000,
    )
    assert plan["harmful"] == 0
    assert plan["benign"] > 0


def test_low_retention_stratum_is_oversampled():
    plan = plan_candidate_wave(
        deficits={"harmful": 100, "benign": 100},
        judged={"harmful": 1000, "benign": 1000},
        retained={"harmful": 1000, "benign": 200},
        wave_budget=1024,
        remaining_budget=10_000,
    )
    # Benign retains ~20%, so it needs roughly five times the draws.
    assert plan["benign"] > 3 * plan["harmful"]


def test_wave_never_exceeds_budget_and_never_stalls():
    plan = plan_candidate_wave(
        deficits={"harmful": 900, "benign": 900},
        judged={"harmful": 10, "benign": 10},
        retained={"harmful": 1, "benign": 1},
        wave_budget=8,
        remaining_budget=4,
    )
    assert sum(plan.values()) <= 4
    assert all(count >= 1 for count in plan.values())


def test_zero_retention_stratum_is_floored_not_divergent():
    plan = plan_candidate_wave(
        deficits={"harmful": 0, "benign": 10},
        judged={"harmful": 0, "benign": 500},
        retained={"harmful": 0, "benign": 0},
        wave_budget=10_000,
        remaining_budget=10_000,
    )
    assert plan["benign"] == pytest.approx(10 / MIN_PLANNING_RETENTION, rel=0.01)


def test_stratified_specs_match_interleaved_specs_exactly():
    """The adaptive builder must draw from the same candidate universe."""
    interleaved = build_interleaved_episode_specs(HARMFUL, BENIGN, 32)
    stratified = build_stratified_episode_specs(
        HARMFUL, BENIGN, {"harmful": 16, "benign": 16}
    )
    by_index = {spec["episode_index"]: spec for spec in stratified}
    for spec in interleaved:
        mirror = dict(by_index[spec["episode_index"]])
        mirror.pop("stratum_ordinal")
        assert mirror == spec


def test_stratified_specs_are_nested_under_deeper_draws():
    shallow = build_stratified_episode_specs(
        HARMFUL, BENIGN, {"harmful": 4, "benign": 9}
    )
    deep = build_stratified_episode_specs(
        HARMFUL, BENIGN, {"harmful": 12, "benign": 30}
    )
    deep_by_key = {
        (s["seed_label"], s["stratum_ordinal"]): s for s in deep
    }
    for spec in shallow:
        assert deep_by_key[(spec["seed_label"], spec["stratum_ordinal"])] == spec


def _candidate(index, label, ordinal, dropped=None):
    return {
        "candidate_index": index,
        "stratum_ordinal": ordinal,
        "candidate_seed": 8888 + index,
        "prompt_type": f"generated_{label}",
        "seed_label": label,
        "dropped_reason": dropped,
        "attacker_raw_reward": 1.0,
        "defender_raw_reward": -1.0,
    }


def test_stratified_assembly_matches_lockstep_on_balanced_input():
    """Accepted games must be identical to the lockstep result."""
    candidates = []
    for index in range(24):
        label = "harmful" if index % 2 == 0 else "benign"
        # Drop a couple of benign candidates the way label drift does.
        dropped = (
            "benign_to_harmful_label_drift"
            if label == "benign" and index in {3, 9}
            else None
        )
        candidates.append(_candidate(index, label, index // 2, dropped))

    lockstep = assemble_valid_interleaved_prefix(candidates, 16)
    adaptive = assemble_valid_stratified_prefix(candidates, 16)
    assert lockstep["complete"] and adaptive["complete"]
    assert lockstep["episodes"] == adaptive["episodes"]
    assert lockstep["deficits"] == adaptive["deficits"]


def test_stratified_assembly_accepts_unequal_strata():
    candidates = [_candidate(2 * i, "harmful", i) for i in range(8)]
    candidates += [
        _candidate(
            2 * i + 1,
            "benign",
            i,
            "benign_to_harmful_label_drift" if i % 2 else None,
        )
        for i in range(16)
    ]
    result = assemble_valid_stratified_prefix(candidates, 16)
    assert result["complete"]
    assert result["judged_counts"] == {"harmful": 8, "benign": 16}
    assert len(result["episodes"]) == 16
    assert result["dropped_counts"]["benign"] == 8
    # Lockstep would have needed 16 harmful draws to reach the same 8 pairs.
    assert result["judged_counts"]["harmful"] == 8


def test_stratified_assembly_rejects_holes():
    candidates = [_candidate(0, "harmful", 0), _candidate(4, "harmful", 2)]
    candidates += [_candidate(1, "benign", 0)]
    with pytest.raises(ValueError, match="contiguous nested prefix"):
        assemble_valid_stratified_prefix(candidates, 2)


def test_projected_totals_report_the_lockstep_waste():
    projected = projected_candidate_totals(
        required_per_stratum=2000,
        retention={"harmful": 1.0, "benign": 0.591},
    )
    assert projected["adaptive_total"] < projected["lockstep_total"]
    # Lockstep must draw 2000/0.591 harmful candidates it will never use.
    assert projected["saved"] == pytest.approx(2000 / 0.591 - 2000, rel=1e-6)
