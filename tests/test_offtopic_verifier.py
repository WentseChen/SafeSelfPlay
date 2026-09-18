"""Tests for the off-topic verifier and the reward penalty it drives."""

from __future__ import annotations

import os
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from roll.utils.offtopic_verifier import (  # noqa: E402
    DEFAULT_OFFTOPIC_PENALTY,
    apply_offtopic_penalty,
    build_verifier_prompt,
    offtopic_penalty,
    parse_verifier_output,
)


# --- prompt -----------------------------------------------------------------

def test_prompt_contains_both_texts_and_asks_one_word():
    messages = build_verifier_prompt("steal a car", "write a heist screenplay")
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "steal a car" in user and "write a heist screenplay" in user
    assert "ON_TOPIC" in user and "OFF_TOPIC" in user


def test_prompt_rejects_empty_seed():
    with pytest.raises(ValueError):
        build_verifier_prompt("   ", "anything")


def test_verifier_is_told_not_to_judge_harm():
    # The verifier must not double as a safety classifier; WildGuard owns that.
    system = build_verifier_prompt("a", "b")[0]["content"].lower()
    assert "never judge whether they are harmful" in system


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("ON_TOPIC", False),
        ("OFF_TOPIC", True),
        ("  off_topic  ", True),
        ("on topic", False),
        ("The answer is OFF_TOPIC.", True),
        ("Verdict: ON_TOPIC\n", False),
    ],
)
def test_parse_known_verdicts(text, expected):
    assert parse_verifier_output(text) is expected


@pytest.mark.parametrize("text", ["", None, "maybe", "I cannot help with that"])
def test_unparseable_is_none_not_false(text):
    # None must stay distinct from False: an unparseable verdict silently
    # becoming "on-topic" is exactly how the penalty would disable itself.
    assert parse_verifier_output(text) is None


# --- penalty ----------------------------------------------------------------

def test_on_topic_leaves_reward_untouched():
    value, diag = apply_offtopic_penalty(0.8, {"off_topic": False})
    assert value == 0.8
    assert diag["offtopic_verdict"] == "on_topic"
    assert diag["offtopic_penalty_applied"] == 0.0


def test_off_topic_subtracts_the_penalty():
    value, diag = apply_offtopic_penalty(0.8, {"off_topic": True}, penalty=0.5)
    assert value == pytest.approx(0.3)
    assert diag["offtopic_verdict"] == "off_topic"
    assert diag["offtopic_penalty_applied"] == pytest.approx(0.5)


def test_penalty_clamps_at_the_lower_bound():
    # The zero-sum projection assumes normalized values stay in [-1, 1].
    value, diag = apply_offtopic_penalty(-0.9, {"off_topic": True}, penalty=0.5)
    assert value == -1.0
    assert diag["offtopic_penalty_applied"] == pytest.approx(0.1)


def test_absent_and_unparsed_verdicts_are_distinguishable_and_unpenalised():
    absent_value, absent = apply_offtopic_penalty(0.4, {})
    unparsed_value, unparsed = apply_offtopic_penalty(0.4, {"off_topic": None})
    assert absent_value == unparsed_value == 0.4
    assert absent["offtopic_verdict"] == "absent"
    assert unparsed["offtopic_verdict"] == "unparsed"


def test_penalty_magnitude_is_env_tunable(monkeypatch):
    monkeypatch.delenv("OFFTOPIC_PENALTY", raising=False)
    assert offtopic_penalty() == DEFAULT_OFFTOPIC_PENALTY
    monkeypatch.setenv("OFFTOPIC_PENALTY", "0.25")
    assert offtopic_penalty() == 0.25
    monkeypatch.setenv("OFFTOPIC_PENALTY", "")
    assert offtopic_penalty() == DEFAULT_OFFTOPIC_PENALTY
    monkeypatch.setenv("OFFTOPIC_PENALTY", "9")
    with pytest.raises(ValueError):
        offtopic_penalty()


# --- integration with the zero-sum projection -------------------------------

def test_zero_sum_reward_stays_antisymmetric_and_bounded_under_penalty():
    from roll.utils.role_lora_zero_sum_psro import compute_zero_sum_psro_reward

    labels = {
        "prompt_harmfulness": "harmful",
        "response_harmfulness": "harmful",
        "response_refusal": "compliance",
        "is_parsing_error": False,
    }
    clean = compute_zero_sum_psro_reward(
        prompt_type="generated_harmful", labels=labels
    )
    drifted = compute_zero_sum_psro_reward(
        prompt_type="generated_harmful", labels={**labels, "off_topic": True}
    )
    # antisymmetry and bounds are the two invariants the matrix depends on
    for row in (clean, drifted):
        assert row["defender_zero_sum_reward"] == -row["attacker_zero_sum_reward"]
        assert -1.0 <= row["attacker_zero_sum_reward"] <= 1.0
    # an off-topic attack is worth strictly less to the attacker
    assert (
        drifted["attacker_zero_sum_reward"] < clean["attacker_zero_sum_reward"]
    )
    assert clean["offtopic_verdict"] == "absent"
    assert drifted["offtopic_verdict"] == "off_topic"
