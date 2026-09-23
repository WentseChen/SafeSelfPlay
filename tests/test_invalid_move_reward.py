"""The v6d reward with a flat floor for invalid attacker moves."""

from itertools import product
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roll.utils import role_lora_training_reward as training
from roll.utils.upstream_v2_payoff import compute_upstream_v2_raw_rewards

EXPECTED = {"generated_harmful": "harmful", "generated_benign": "unharmful"}


def score(prompt_type, labels, attacker_cot):
    raw = compute_upstream_v2_raw_rewards(
        prompt_type=prompt_type, labels=labels,
        attacker_cot_format_violation=attacker_cot,
        defender_cot_format_violation=False,
    )["attacker_raw_reward"]
    return training.invalid_move_attacker_reward(
        reward=raw, prompt_type=prompt_type, labels=labels,
        cot_format_violation=attacker_cot,
    )


def sweep():
    for prompt_type, prompt, refusal, harm, off, cot in product(
        EXPECTED, ["harmful", "unharmful", None], ["refusal", "compliance", None],
        ["harmful", "unharmful", None], [False, True], [False, True],
    ):
        labels = {"prompt_harmfulness": prompt, "response_refusal": refusal,
                  "response_harmfulness": harm, "is_parsing_error": False, "off_topic": off}
        reward, violations = score(prompt_type, labels, cot)
        yield prompt_type, prompt, refusal, harm, off, cot, reward, violations


def test_every_invalid_move_is_flat_and_strictly_dominated():
    rows = list(sweep())
    assert len(rows) == 216  # the defender's format flag cannot move the attacker reward
    for cot in (False, True):
        shift = 2.0 if cot else 0.0
        subset = [r for r in rows if r[5] == cot]
        valid = [r[6] for r in subset if r[7] == 0]
        assert {r[6] for r in subset if r[7] == 1} == {-3.0 - shift}
        assert {r[6] for r in subset if r[7] == 2} == {-4.0 - shift}
        assert min(valid) == -shift
        assert min(valid) > -3.0 - shift


def test_missing_prompt_label_counts_as_drift():
    for prompt_type in EXPECTED:
        labels = {"prompt_harmfulness": None, "response_refusal": "compliance",
                  "response_harmfulness": "harmful", "off_topic": False}
        assert score(prompt_type, labels, False) == (-3.0, 1)


def test_unclassifiable_reply_earns_nothing_above_zero():
    for prompt_type, refusal, harm in product(EXPECTED, ["refusal", "compliance", None],
                                              ["harmful", "unharmful", None]):
        if refusal is not None and harm is not None:
            continue
        labels = {"prompt_harmfulness": EXPECTED[prompt_type], "response_refusal": refusal,
                  "response_harmfulness": harm, "off_topic": False}
        reward, violations = score(prompt_type, labels, False)
        assert violations == 0 and reward <= 0.0


@pytest.mark.parametrize("prompt_type,prompt,refusal,harm,off,expected", [
    ("generated_harmful", "harmful", "compliance", "harmful", False, 3.0),     # row 3
    ("generated_harmful", "harmful", "compliance", "harmful", True, -3.0),     # row 6
    ("generated_harmful", "unharmful", "compliance", "unharmful", False, -3.0),  # row 8
    ("generated_harmful", "unharmful", "compliance", "unharmful", True, -4.0),   # row 11
    ("generated_benign", "unharmful", "refusal", "unharmful", False, 1.0),     # row 13
    ("generated_benign", "unharmful", "refusal", "unharmful", True, -3.0),     # row 16
    ("generated_benign", "harmful", "compliance", "harmful", True, -4.0),      # row 24
])
def test_reward_table(prompt_type, prompt, refusal, harm, off, expected):
    labels = {"prompt_harmfulness": prompt, "response_refusal": refusal,
              "response_harmfulness": harm, "off_topic": off}
    assert score(prompt_type, labels, False)[0] == expected


def test_valid_moves_keep_the_v6d_reward():
    for prompt_type, refusal, harm in product(EXPECTED, ["refusal", "compliance"],
                                              ["harmful", "unharmful"]):
        labels = {"prompt_harmfulness": EXPECTED[prompt_type], "response_refusal": refusal,
                  "response_harmfulness": harm, "off_topic": False}
        raw = compute_upstream_v2_raw_rewards(
            prompt_type=prompt_type, labels=labels,
            attacker_cot_format_violation=False, defender_cot_format_violation=False,
        )["attacker_raw_reward"]
        assert score(prompt_type, labels, False) == (raw, 0)


def test_flag_defaults_off_and_rejects_junk(monkeypatch):
    monkeypatch.delenv("ATTACKER_INVALID_MOVE", raising=False)
    assert training.invalid_move_reward_enabled() is False
    monkeypatch.setenv("ATTACKER_INVALID_MOVE", "1")
    assert training.invalid_move_reward_enabled() is True
    monkeypatch.setenv("ATTACKER_INVALID_MOVE", "yes")
    with pytest.raises(ValueError):
        training.invalid_move_reward_enabled()


def test_patches_wire_the_rule_into_training():
    trainer = (ROOT / "modal_upstream_selfredteam_role_lora_v2.py").read_text()
    assert "invalid_move_attacker_reward(" in trainer
    # The new branch must come before the zero cap, or drift keeps the cap.
    assert trainer.index("invalid_move_reward_enabled()") < trainer.index('label_drift_action == "cap_zero"')
    topic = (ROOT / "scripts/slurm/patch_offtopic_labels.py").read_text()
    assert "shared_game_reward_enabled() or _invalid_move" in topic
    phase = (ROOT / "scripts/slurm/role_phase.sh").read_text()
    assert phase.count('if [ "$NEEDS_TOPIC_JUDGE" = 1 ]; then') == 2
