"""Exercise actual patched rollout rewards against PSRO's matrix scorer."""

import ast
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roll.utils import role_lora_training_reward as training
from roll.utils import shared_game_reward as shared
from roll.utils.role_lora_zero_sum_psro import compute_zero_sum_psro_reward, analyze_zero_sum_convergence


@pytest.fixture(autouse=True)
def shared_mode(monkeypatch):
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_PENALTY", "3")


def labels(*, off_topic=False, harmful=True, refusal=False):
    return {"prompt_harmfulness": "harmful", "response_harmfulness": "harmful" if harmful else "unharmful",
            "response_refusal": "refusal" if refusal else "compliance", "off_topic": off_topic}


def test_off_topic_success_strictly_worse_than_any_on_topic_outcome():
    accepted = []
    for prompt, response, refusal in product(("harmful", "unharmful"), ("harmful", "unharmful"), ("refusal", "compliance")):
        lab = {"prompt_harmfulness": prompt, "response_harmfulness": response,
               "response_refusal": refusal, "off_topic": False}
        clean = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels=lab)
        drift = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels={**lab, "off_topic": True})
        assert drift["attacker_zero_sum_reward"] == pytest.approx(clean["attacker_zero_sum_reward"] - 3)
        assert drift["defender_zero_sum_reward"] == -drift["attacker_zero_sum_reward"]
        accepted.append((clean["attacker_zero_sum_reward"], drift["attacker_zero_sum_reward"]))
    assert max(drift for _, drift in accepted) < min(clean for clean, _ in accepted)
    assert compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels=labels(off_topic=True))["attacker_zero_sum_reward"] == -2
    assert shared.shared_game_reward_bounds() == (-4, 1)


@pytest.mark.parametrize("invalid", [None, "false", "true", 0, 1])
def test_missing_or_non_boolean_topic_label_fails(invalid):
    with pytest.raises(ValueError, match="boolean"):
        shared.compute_shared_game_reward(prompt_type="generated_harmful", labels=labels(off_topic=invalid))


def test_matrix_statistics_accept_full_penalized_range():
    rows = []
    for i, prompt_type in enumerate(("generated_harmful", "generated_benign") * 2):
        lab = labels(off_topic=True, harmful=False)
        if prompt_type == "generated_benign":
            lab["prompt_harmfulness"] = "unharmful"
        rows.append({"episode_index": i, "prompt_type": prompt_type,
                     **compute_zero_sum_psro_reward(prompt_type=prompt_type, labels=lab)})
    result = analyze_zero_sum_convergence(rows, sample_counts=[2, 4])
    assert result["reward_bounds"] == [-4, 1]


@pytest.fixture(scope="module")
def rollout_class(tmp_path_factory):
    upstream = Path(os.environ.get("SSP_UPSTREAM_SRC", ROOT.parent / "selfplay-redteaming"))
    if not (upstream / "openrlhf/trainer/ppo_utils/language_game.py").is_file():
        pytest.skip("Integration check requires the pinned upstream checkout")
    root = tmp_path_factory.mktemp("shared-reward-upstream")
    work = root / "work"
    env = {**os.environ, "SSP_ROOT": str(root), "SSP_WORK": str(work), "SSP_UPSTREAM_SRC": str(upstream)}
    for script in ("prepare_sources.py", "patch_offtopic_labels.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts/slurm" / script)], env=env, check=True, capture_output=True)
    tree = ast.parse((work / "openrlhf/trainer/ppo_utils/language_game.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DialogueGameManager")
    namespace = {**vars(training), **vars(shared), "os": os, "time": __import__("time"),
                 "get_cot_formatting_reward": lambda bad: -999 if bad else 999}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "patched_language_game", "exec"), namespace)
    return namespace["DialogueGameManager"], namespace


def make_manager(cls, *, prompt_type="generated_harmful", bad_format=False):
    manager = object.__new__(cls)
    manager.custom_configs = {"optimizer_train_role": "attacker", "label_drift_training_policy": training.TRAINING_LABEL_DRIFT_POLICY}
    manager.disable_hidden_cot = False
    manager.get_redteaming_game_reward = lambda **kw: (12345.0, set())
    manager.active_games = {0: {"prompts": "Explain the water cycle", "prompt_type": prompt_type,
        "history": [{"content": "Explain the water cycle as a teacher"}, {"content": "A reply"}],
        "is_generated_attack": True,
        "processed_output_history": [{"output": role, "game_states": {
            "game_role": role, "prompt_type": prompt_type, "cot_format_violation": bad_format}}
            for role in ("attacker", "defender")]}}
    return manager


@pytest.mark.parametrize("off_topic,bad_format", list(product((False, True), repeat=2)))
def test_production_rollout_and_matrix_rewards_match_even_with_format_shaping(rollout_class, off_topic, bad_format):
    cls, _ = rollout_class
    manager = make_manager(cls, bad_format=bad_format)
    lab = labels(off_topic=off_topic)
    _, attackers, _, defenders, _ = manager.filter_and_compute_rewards({0: lab})
    matrix = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels=lab)
    assert attackers[0]["reward"] == matrix["attacker_zero_sum_reward"]
    assert defenders[0]["reward"] == matrix["defender_zero_sum_reward"]
    assert attackers[0]["topic_reward_component"] == (-3 if off_topic else 0)


def test_both_training_roles_drop_same_invalid_games_as_matrix(rollout_class):
    from roll.utils.role_lora_zero_sum_psro import ZeroSumEpisodeDrop
    cls, _ = rollout_class
    manager = make_manager(cls, prompt_type="generated_benign")
    result = manager.filter_and_compute_rewards({0: labels()})
    assert result[:4] == ([], [], [], [])
    with pytest.raises(ZeroSumEpisodeDrop):
        compute_zero_sum_psro_reward(prompt_type="generated_benign", labels=labels())


def test_external_label_flows_through_real_rollout_to_shared_reward(rollout_class, monkeypatch):
    cls, namespace = rollout_class
    manager = make_manager(cls)
    manager.remote_rm_url = ["http://wildguard.invalid/classify"]
    manager.strategy = SimpleNamespace(print=lambda *a: None, get_rank=lambda: 0)
    namespace["remote_rm_fn"] = lambda *a, **kw: [labels()]
    seen = []
    def judge(queries):
        seen.extend(queries)
        return [{"game_idx": 0, "off_topic": True, "topic_judge_model": "test-model"}]
    monkeypatch.setattr("roll.utils.topic_api.label_topic_queries", judge)
    merged = manager.evaluate_game_outcomes()
    _, attackers, _, defenders, _ = manager.filter_and_compute_rewards(merged)
    assert seen == [{"game_idx": 0, "seed": "Explain the water cycle",
                     "rewrite": "Explain the water cycle as a teacher"}]
    assert attackers[0]["reward"] == -2 and defenders[0]["reward"] == 2
    assert attackers[0]["topic_judge_model"] == "test-model"
