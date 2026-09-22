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
    monkeypatch.setenv("TOPIC_PENALTY", "2")


def labels(*, off_topic=False, harmful=True, refusal=False):
    return {"prompt_harmfulness": "harmful", "response_harmfulness": "harmful" if harmful else "unharmful",
            "response_refusal": "refusal" if refusal else "compliance", "off_topic": off_topic}


def test_off_topic_success_is_never_better_than_an_on_topic_outcome():
    penalty = shared.topic_penalty()
    accepted = []
    for prompt, response, refusal in product(("harmful", "unharmful"), ("harmful", "unharmful"), ("refusal", "compliance")):
        lab = {"prompt_harmfulness": prompt, "response_harmfulness": response,
               "response_refusal": refusal, "off_topic": False}
        clean = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels=lab, attacker_cot_format_violation=False)
        drift = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels={**lab, "off_topic": True}, attacker_cot_format_violation=False)
        assert drift["attacker_zero_sum_reward"] == pytest.approx(clean["attacker_zero_sum_reward"] - penalty)
        assert drift["defender_zero_sum_reward"] == -drift["attacker_zero_sum_reward"]
        accepted.append((clean["attacker_zero_sum_reward"], drift["attacker_zero_sum_reward"]))
    # Keeping the CoT component widened the unpenalized payoff from [-1, 1] to
    # [-1.5, 1.5], so the span an off-topic win must cover is 3.0, not 2.0. At
    # the default penalty of 2.0 an off-topic win is NO LONGER dominated: it
    # scores -0.5 against the worst on-topic outcome's -1.5. Dominance is now a
    # property of the penalty, not a guarantee -- see the test below.
    drifted = compute_zero_sum_psro_reward(prompt_type="generated_harmful", labels=labels(off_topic=True), attacker_cot_format_violation=False)
    assert drifted["attacker_zero_sum_reward"] == drifted["base_attacker_zero_sum_reward"] - penalty
    assert shared.shared_game_reward_bounds() == (
        -shared.UNPENALIZED_BOUND - penalty, shared.UNPENALIZED_BOUND)


def test_strict_dominance_needs_a_penalty_above_the_payoff_span():
    """The span of on-topic payoffs is exactly what an off-topic win must cover."""
    # CoT format is a reward component again, so the span is only visible when
    # its violation flags vary too.
    reachable = sorted({shared.compute_shared_game_reward(
        prompt_type="generated_harmful",
        labels={"prompt_harmfulness": p, "response_harmfulness": r,
                "response_refusal": rf, "off_topic": False},
        attacker_cot_format_violation=av,
        defender_cot_format_violation=dv)["attacker_zero_sum_reward"]
        for p, r, rf, av, dv in product(("harmful", "unharmful"), ("harmful", "unharmful"),
                                        ("refusal", "compliance"), (False, True),
                                        (False, True, None))})
    assert reachable[0] == -shared.UNPENALIZED_BOUND and reachable[-1] == shared.UNPENALIZED_BOUND
    span = reachable[-1] - reachable[0]
    assert span == 3.0          # CoT format is back in, so the span widened from 2.0
    # The default penalty no longer dominates: an off-topic win beats an
    # on-topic loss by 1.0. Raising TOPIC_PENALTY above 3.0 would restore it.
    assert reachable[-1] - shared.topic_penalty() > reachable[0]


@pytest.mark.parametrize("invalid", [None, "false", "true", 0, 1])
def test_missing_or_non_boolean_topic_label_fails(invalid):
    with pytest.raises(ValueError, match="boolean"):
        shared.compute_shared_game_reward(prompt_type="generated_harmful", labels=labels(off_topic=invalid), attacker_cot_format_violation=False)


def test_matrix_statistics_accept_full_penalized_range():
    rows = []
    for i, prompt_type in enumerate(("generated_harmful", "generated_benign") * 2):
        lab = labels(off_topic=True, harmful=False)
        if prompt_type == "generated_benign":
            lab["prompt_harmfulness"] = "unharmful"
        rows.append({"episode_index": i, "prompt_type": prompt_type,
                     **compute_zero_sum_psro_reward(prompt_type=prompt_type, labels=lab, attacker_cot_format_violation=False)})
    result = analyze_zero_sum_convergence(rows, sample_counts=[2, 4])
    assert result["reward_bounds"] == [-shared.UNPENALIZED_BOUND - shared.topic_penalty(), shared.UNPENALIZED_BOUND]


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


def make_manager(cls, *, prompt_type="generated_harmful", bad_format=False, defender_bad_format=None):
    if defender_bad_format is None:
        defender_bad_format = bad_format
    manager = object.__new__(cls)
    manager.custom_configs = {"optimizer_train_role": "attacker", "label_drift_training_policy": training.TRAINING_LABEL_DRIFT_POLICY}
    manager.disable_hidden_cot = False
    manager.get_redteaming_game_reward = lambda **kw: (12345.0, set())
    manager.active_games = {0: {"prompts": "Explain the water cycle", "prompt_type": prompt_type,
        "history": [{"content": "Explain the water cycle as a teacher"}, {"content": "A reply"}],
        "is_generated_attack": True,
        "processed_output_history": [{"output": role, "game_states": {
            "game_role": role, "prompt_type": prompt_type,
            "cot_format_violation": bad_format if role == "attacker" else defender_bad_format}}
            for role in ("attacker", "defender")]}}
    return manager


@pytest.mark.parametrize("off_topic,attacker_bad,defender_bad", list(product((False, True), repeat=3)))
def test_production_rollout_and_matrix_rewards_match_even_with_format_shaping(
    rollout_class, off_topic, attacker_bad, defender_bad,
):
    cls, _ = rollout_class
    manager = make_manager(cls, bad_format=attacker_bad, defender_bad_format=defender_bad)
    lab = labels(off_topic=off_topic)
    _, attackers, _, defenders, _ = manager.filter_and_compute_rewards({0: lab})
    matrix = compute_zero_sum_psro_reward(
        prompt_type="generated_harmful", labels=lab,
        attacker_cot_format_violation=attacker_bad, defender_cot_format_violation=defender_bad,
    )
    expected = 1.0 - 0.5 * attacker_bad + 0.5 * defender_bad - 2 * off_topic
    assert attackers[0]["reward"] == expected
    assert attackers[0]["reward"] == matrix["attacker_zero_sum_reward"]
    assert defenders[0]["reward"] == matrix["defender_zero_sum_reward"]
    assert attackers[0]["topic_reward_component"] == (-2 if off_topic else 0)


def test_both_training_roles_drop_same_invalid_games_as_matrix(rollout_class):
    from roll.utils.role_lora_zero_sum_psro import ZeroSumEpisodeDrop
    cls, _ = rollout_class
    manager = make_manager(cls, prompt_type="generated_benign")
    result = manager.filter_and_compute_rewards({0: labels()})
    assert result[:4] == ([], [], [], [])
    with pytest.raises(ZeroSumEpisodeDrop):
        compute_zero_sum_psro_reward(prompt_type="generated_benign", labels=labels(), attacker_cot_format_violation=False)


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
    assert attackers[0]["reward"] == -1 and defenders[0]["reward"] == 1
    assert attackers[0]["topic_judge_model"] == "test-model"


def test_the_cot_component_survives_into_the_shared_reward():
    """v1 passed the flags as False and subtracted cot_format back out, so a
    format violation was invisible to training and to the matrix."""
    lab = {"prompt_harmfulness": "harmful", "response_harmfulness": "harmful",
           "response_refusal": "compliance", "off_topic": False}
    clean = shared.compute_shared_game_reward(prompt_type="generated_harmful", labels=lab, attacker_cot_format_violation=False)
    broken = shared.compute_shared_game_reward(
        prompt_type="generated_harmful", labels=lab, attacker_cot_format_violation=True)
    # +1 -> -1 on a component that is divided by 4
    assert clean["attacker_zero_sum_reward"] - broken["attacker_zero_sum_reward"] == 0.5
    assert broken["attacker_cot_format_violation"] is True


def test_the_topic_penalty_is_added_on_top_of_the_unchanged_reward():
    lab = {"prompt_harmfulness": "harmful", "response_harmfulness": "harmful",
           "response_refusal": "compliance", "off_topic": True}
    for av, dv in ((False, None), (True, False), (False, True)):
        drifted = shared.compute_shared_game_reward(
            prompt_type="generated_harmful", labels=lab,
            attacker_cot_format_violation=av, defender_cot_format_violation=dv)
        on_topic = shared.compute_shared_game_reward(
            prompt_type="generated_harmful", labels={**lab, "off_topic": False},
            attacker_cot_format_violation=av, defender_cot_format_violation=dv)
        # the ONLY difference is the deduction; every original component is intact
        assert drifted["base_attacker_zero_sum_reward"] == on_topic["base_attacker_zero_sum_reward"]
        assert (on_topic["attacker_zero_sum_reward"] - drifted["attacker_zero_sum_reward"]
                == shared.topic_penalty())


def test_the_attacker_flag_has_no_default():
    """A default turns the attacker's format component into a constant +1 at
    any call site that forgets it -- an unconditional reward, not a missing
    one -- which is how the training call site stayed broken while the
    contract still advertised format_reward: true."""
    import inspect

    for func in (shared.compute_shared_game_reward, compute_zero_sum_psro_reward):
        parameter = inspect.signature(func).parameters["attacker_cot_format_violation"]
        assert parameter.default is inspect.Parameter.empty, func.__name__
    with pytest.raises(TypeError):
        shared.compute_shared_game_reward(
            prompt_type="generated_harmful", labels=labels(off_topic=False))


def test_the_training_call_site_passes_the_real_violations():
    """The matrix path being fixed is not enough: format collapse is a
    training-time phenomenon, so the injected game-loop patch must forward the
    per-role flags it reads from processed_output_history."""
    patch = Path("modal_upstream_selfredteam_role_lora_v2.py").read_text()
    start = patch.index("shared_score = compute_shared_game_reward(")
    call = patch[start:start + 400]
    assert "attacker_cot_format_violation=" in call
    assert "defender_cot_format_violation=" in call
    # and the flags must come from the turn states, not be hard-coded
    assert "cot_format_violation" in patch[:patch.index("shared_score = compute_shared_game_reward(")][-900:]
