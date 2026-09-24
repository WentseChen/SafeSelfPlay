"""Defender format floor: a broken format never outranks a formatted reply."""

import itertools
from pathlib import Path

import pytest

from roll.utils.role_lora_training_reward import (
    DEFENDER_FORMAT_FLOOR_REWARD,
    defender_format_floor_enabled,
    defender_format_floor_reward,
)

ROOT = Path(__file__).resolve().parents[1]
# Additive Self-RedTeam defender terms: harmfulness +-1, refusal +-1, format +-1.
TERMS = (-1.0, 1.0)


def additive(harm, refusal, violation):
    return harm + refusal + (-1.0 if violation else 1.0)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEFENDER_FORMAT_FLOOR", raising=False)
    assert defender_format_floor_enabled() is False


def test_rejects_bad_value(monkeypatch):
    monkeypatch.setenv("DEFENDER_FORMAT_FLOOR", "yes")
    with pytest.raises(ValueError):
        defender_format_floor_enabled()


def test_every_violation_ranks_below_every_formatted_reply():
    formatted = [defender_format_floor_reward(reward=additive(h, r, False), cot_format_violation=False)
                 for h, r in itertools.product(TERMS, TERMS)]
    broken = [defender_format_floor_reward(reward=additive(h, r, True), cot_format_violation=True)
              for h, r in itertools.product(TERMS, TERMS)]
    assert max(broken) < min(formatted)
    assert set(broken) == {DEFENDER_FORMAT_FLOOR_REWARD}
    # Without the floor a broken-format refusal (+1) beats a formatted compliance (-1).
    assert additive(1.0, 1.0, True) > additive(-1.0, -1.0, False)


def test_formatted_reward_untouched():
    for h, r in itertools.product(TERMS, TERMS):
        value = additive(h, r, False)
        assert defender_format_floor_reward(reward=value, cot_format_violation=False) == value
    assert defender_format_floor_reward(reward=2.5, cot_format_violation=None) == 2.5


def test_game_loop_patch_applies_floor_to_defender_turns():
    source = (ROOT / "modal_upstream_selfredteam_role_lora_v2.py").read_text()
    assert "defender_format_floor_enabled,\\n" in source
    block = source[source.index('turn_states["game_role"] == "defender"\n                    and defender_format_floor_enabled()'):]
    assert block.index("defender_format_floor_reward(") < block.index("turn_states['reward'] = reward")


def test_recipe_records_flag_only_when_enabled():
    text = (ROOT / "docker/llama8b_entrypoint.py").read_text()
    assert 'if env.get("DEFENDER_FORMAT_FLOOR", "0") != "0":' in text
    assert "DEFENDER_FORMAT_FLOOR" in (ROOT / "docker/run_llama8b.sh").read_text()
