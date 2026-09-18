from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roll.utils.history_selfplay import history_schedule


ROOT = Path(__file__).resolve().parents[1]


def load_driver():
    spec = importlib.util.spec_from_file_location("baseline_driver", ROOT / "scripts/slurm/history_selfplay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_eight_rounds_use_every_available_opponent_equally():
    phases = history_schedule("uniform", 8)
    assert [p["target"] for p in phases] == [f"{r}{i}" for i in range(1, 9) for r in "AD"]
    a3 = phases[4]
    assert a3["opponents"] == ["D0", "D1", "D2"]
    assert a3["probabilities"] == [1 / 3] * 3
    assert a3["initialized_from"] == "A0"
    assert phases[-1]["opponents"] == [f"A{i}" for i in range(9)]
    assert phases[-1]["probabilities"] == [1 / 9] * 9
    assert phases[-1]["initialized_from"] == "D0"


def test_naive_six_rounds_inherit_same_role_and_face_latest_opponent():
    phases = history_schedule("naive", 6)
    assert len(phases) == 12
    assert [(p["target"], p["initialized_from"], p["opponents"]) for p in phases] == [
        (f"{r}{i}", f"{r}{i - 1}", [f"D{i - 1}" if r == "A" else f"A{i}"])
        for i in range(1, 7) for r in "AD"
    ]
    assert all(p["probabilities"] == [1.0] for p in phases)


@pytest.mark.parametrize("method,rounds", [("uniform", 8), ("naive", 6)])
def test_baseline_resume_after_interruption_preserves_schedule(tmp_path, monkeypatch, method, rounds):
    driver = load_driver()
    monkeypatch.setenv("INIT_KL_COEF", "0")
    cfg = SimpleNamespace(run_root=tmp_path / "run", suffix="run")
    def base(label):
        return {"path": None, "adapter_sha256": "base:model"}
    psro = SimpleNamespace(
        BASE_MODEL="model", _base_population_identity=base,
        _read_json=lambda path: json.loads(path.read_text()),
        _write_json_atomic=lambda path, value: path.write_text(json.dumps(value)),
        _verify_population_identity=lambda *a, **kw: None,
    )
    calls = []
    class Interrupted(Exception):
        pass
    def train(**kw):
        if kw["target"] == "A3" and not calls.count("interrupted"):
            calls.append("interrupted")
            raise Interrupted
        calls.append(kw["target"])
        pool = kw["opponent_pool"]
        assert sum(float(e["probability"]) for e in pool) == pytest.approx(1)
        return {"path": f"/adapters/{kw['target']}", "adapter_sha256": kw["target"]}
    with pytest.raises(Interrupted):
        driver.run_baseline(psro, cfg, train, method=method, rounds=rounds, steps=100)
    partial = json.loads((cfg.run_root / "state.json").read_text())
    assert partial["stage"] == "training_A3"
    result = driver.run_baseline(psro, cfg, train, method=method, rounds=rounds, steps=100)
    expected = [p["target"] for p in history_schedule(method, rounds)]
    assert [x for x in calls if x != "interrupted"] == expected
    assert result["completed"]
    assert len(result["population"]) == 2 + rounds * 2
    driver.run_baseline(psro, cfg, train, method=method, rounds=rounds, steps=100)
    assert len(calls) == 2 * rounds + 1  # a completed run performs no training
    with pytest.raises(ValueError, match="settings changed"):
        driver.run_baseline(psro, cfg, train, method=method, rounds=rounds, steps=200)


def test_naive_rejects_base_kl_before_starting_training(tmp_path, monkeypatch):
    driver = load_driver()
    monkeypatch.setenv("INIT_KL_COEF", "0.01")
    with pytest.raises(ValueError, match="INIT_KL_COEF=0"):
        driver.run_baseline(SimpleNamespace(), SimpleNamespace(run_root=tmp_path),
                            None, method="naive", rounds=6, steps=100)


@pytest.mark.parametrize("mode", ["cold", "hot"])
@pytest.mark.parametrize("early", [False, True])
def test_local_trainer_uses_prescribed_start_and_discards_inherited_overrides(tmp_path, monkeypatch, mode, early):
    driver = load_driver()
    import cold_psro_local as local
    monkeypatch.setenv("INIT_KL_COEF", "0")
    monkeypatch.setenv("PSRO_MIDRUN_RESUME", "0")
    root = tmp_path / "psro/run"
    parent = root / "training/A1/ckpt/global_step100_hf"
    parent.mkdir(parents=True)
    (parent / "adapter_model.safetensors").write_bytes(b"test")
    (root / "state.json").write_text(json.dumps({"population": {"A1": {"path": str(parent)}}}))
    psro = SimpleNamespace(
        PSRO_OUTPUT_ROOT=root.parent, BASE_MODEL="base", TRAINING_LABEL_DRIFT_POLICY="test",
        _normalize_fixed_opponent_pool=lambda pool: pool,
        _read_json=lambda path: json.loads(path.read_text()),
        _write_json_atomic=lambda path, obj: path.write_text(json.dumps(obj)),
        _adapter_checkpoint=lambda path: (path / "adapter_model.safetensors").is_file(),
        _adapter_identity=lambda path: {"path": str(path), "adapter_sha256": "test-sha"},
    )
    cfg = SimpleNamespace(init_mode=mode, workspace=tmp_path, work=tmp_path, hf_home=tmp_path,
                          sft_dir=tmp_path, python="python", gpus=4, train_judge_port=5311, ray_port=6311,
                          base_env=lambda: {"START_ADAPTER": "/unrelated", "OPPONENT": "/unrelated"})
    calls = []
    def fake_training(command, **kwargs):
        env = kwargs["env"]
        calls.append(env)
        assert "OPPONENT" not in env
        assert env.get("START_ADAPTER") == (str(parent) if mode == "hot" else None)
        output = Path(env["SSP_OUT"]) / env["TAG"] / "ckpt/global_step100_hf"
        output.mkdir(parents=True)
        (output / "adapter_model.safetensors").write_bytes(b"trained")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(local.subprocess, "run", fake_training)
    if early:
        monkeypatch.setenv("LADDER_EARLY_STOP", "1")
        def supervise(env, log, criterion, checkpoint_dir):
            calls.append(env)
            for step in range(1, 81):
                criterion.observe(step, {criterion.metric: 0.5, criterion.ceiling_metric: 0.05})
            assert criterion.triggered
            for step in (80, 90):
                output = checkpoint_dir / f"global_step{step}_hf"
                output.mkdir(parents=True)
                (output / "adapter_model.safetensors").write_bytes(b"trained")
            return 0, "converged at step 80"
        monkeypatch.setattr(local, "_supervise_training", supervise)
    else:
        monkeypatch.setattr(local, "_build_criterion", lambda role: None)
    trainer = driver.make_local_oracle_trainer(psro, cfg)
    kwargs = dict(run_root=str(root), target="A2", role="attacker", opponent_pool=[], steps=100,
                  lora_rank=64, lora_alpha=64, learning_rate=1e-5, sft_stop_after_step=30,
                  sft_batches_per_step=1, save_steps=10, actor_lr_scheduler="constant_with_warmup",
                  lr_warmup_ratio=0.05, training_seed=8888, wandb_identity="test")
    result = trainer(**kwargs)
    assert result["path"].endswith(f"training/A2/ckpt/global_step{80 if early else 100}_hf")
    assert trainer(**kwargs) == result
    assert len(calls) == 1
    manifest = json.loads((root / "training/A2/manifest.json").read_text())
    assert manifest["role_start"] == (str(parent) if mode == "hot" else "base")
    assert manifest["kl_penalty"] == 0
    assert manifest["steps_completed"] == (80 if early else 100)
    assert manifest["early_stopping"]["enabled"] is early
