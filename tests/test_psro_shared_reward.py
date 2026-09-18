"""Exercise the actual PSRO coordinator and local trainer interface without GPUs."""

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["legacy", "shared_binary"])
def test_full_eight_round_psro_saves_matrix_without_fabricating_benchmarks(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GAME_REWARD_MODE", mode)
    monkeypatch.setenv("TOPIC_PENALTY", "3")
    monkeypatch.setenv("TOPIC_API_MODEL", "test-fixed-model")
    monkeypatch.setenv("TOPIC_API_KEY", "test-key")
    sys.path.insert(0, str(ROOT / "scripts/slurm"))
    import cold_psro_common as common
    import cold_psro_local as local
    from roll.utils.role_lora_cold_psro import next_action

    psro, payoff = common.load_modules()
    monkeypatch.setattr(psro, "PSRO_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(psro, "_verify_population_identity", lambda *a, **kw: None)
    calls = []
    signature = inspect.signature(local.make_local_oracle_trainer(psro, SimpleNamespace()))
    def train(**kwargs):
        # Check the actual coordinator -> local trainer API, then simulate GPU work.
        signature.bind(**kwargs)
        assert kwargs["cold_start"] is True
        assert kwargs["role_start_label"] == kwargs["target"][0] + "0"
        calls.append(kwargs["target"])
        return {"path": str(tmp_path / kwargs["target"]), "adapter_sha256": kwargs["target"]}
    def cell(**kwargs):
        return {"value": (int(kwargs["attacker_label"][1:]) - int(kwargs["defender_label"][1:])) / 10,
                "episodes": kwargs["episodes"], "manifest_path": "/simulated/cell.json"}
    monkeypatch.setattr(psro, "train_cold_psro_oracle", common._bind_local(train))
    monkeypatch.setattr(psro, "evaluate_zero_sum_psro_cell", common._bind_local(cell))
    root = tmp_path / "psro8"
    kwargs = {"run_suffix": "psro8", "generations": 8}
    for _ in range(100):
        psro.advance_cold_psro(**kwargs)
        state = psro._read_json(root / "state.json")
        kwargs = psro._advance_kwargs(state["contract"], "psro8")
        action = next_action(state["population"], state["cells"], state["evaluations"], generations=8)
        if action["kind"] == "evaluation":
            break
    else:
        pytest.fail("eight-round training failed to finish")
    assert len(state["cells"]) == 81
    assert calls == [f"{r}{i}" for i in range(1, 9) for r in "AD"]
    local.finalize_local_training(psro, root, state)
    local.finalize_local_training(psro, root, state)  # resume/finalization is idempotent
    final = psro._read_json(root / "matrices/final_g8.json")
    assert final["shape"] == [9, 9]
    assert sum(final["meta_solution"]["defender_strategy"]) == pytest.approx(1)
    persisted = psro._read_json(root / "state.json")
    assert persisted["training_completed"] is True
    assert persisted["completed"] is False
    assert persisted["evaluations"] == {}
    if mode == "shared_binary":
        assert persisted["contract"]["shared_game_reward"]["topic_penalty"] == 3
        monkeypatch.setenv("TOPIC_PENALTY", "2")
        with pytest.raises(RuntimeError, match="contract|non-extensible"):
            psro.advance_cold_psro(**kwargs)
