"""CPU regressions for source drift, unavailable services and launcher cleanup."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from roll.utils.role_lora_cold_psro import run_contract_settings


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/slurm" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", [
    "modal_role_lora_zero_sum_psro.py", "modal_role_lora_zero_sum_psro_hot.py",
])
def test_resume_and_extension_accept_old_source_hashes_but_reject_settings(tmp_path, filename):
    # Execute the real coordinator functions without importing Modal/GPU images.
    tree = ast.parse((ROOT / filename).read_text())
    names = {"advance_cold_psro", "_contracts_match_for_generation_extension"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    state_path = tmp_path / "run/state.json"
    persisted = []

    class Persisted(Exception):
        pass

    def stop_after_validation(root, state):
        persisted.append(copy.deepcopy(state))
        raise Persisted

    namespace = {
        "Any": Any, "run_contract_settings": run_contract_settings,
        "_EXTENSIBLE_CONTRACT_FIELDS": {"generations", "final_matrix_shape", "post_training_evaluation"},
        "_safe_component": lambda value, **kw: value,
        "PSRO_OUTPUT_ROOT": tmp_path, "BASE_MODEL": "base",
        "ZERO_SUM_REWARD_VERSION": "reward-v1", "TRAINING_LABEL_DRIFT_POLICY": "drift-v1",
        "_psro_implementation_hashes": lambda: {"coordinator": "new", "trainer": "new"},
        "output_vol": SimpleNamespace(reload=lambda: None),
        "_read_json": lambda path: json.loads(path.read_text()),
        "_base_population_identity": lambda label: {"label": label},
        "_persist_psro_inventory": stop_after_validation,
        "_sha256_file": lambda path: "artifact-hash",
        "shared_game_reward_enabled": lambda: False,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    advance = namespace["advance_cold_psro"]
    with pytest.raises(Persisted):
        advance(run_suffix="run")
    old = persisted[-1]
    old["contract"]["implementation_sha256"] = {"coordinator": "old", "reward": "old-reward"}
    state_path.write_text(json.dumps(old))
    with pytest.raises(Persisted):
        advance(run_suffix="run")
    assert persisted[-1]["last_implementation_sha256"] == {"coordinator": "new", "trainer": "new"}

    old["completed"] = True
    state_path.write_text(json.dumps(old))
    assert advance(run_suffix="run")["completed"] is True
    with pytest.raises(Persisted):
        advance(run_suffix="run", generations=6)
    assert persisted[-1]["extension_history"][-1]["from_generations"] == 5
    for settings in ({"attacker_learning_rate": 2e-5}, {"matrix_episodes": 2000}):
        with pytest.raises(RuntimeError, match="non-extensible"):
            advance(run_suffix="run", **settings)
    namespace["ZERO_SUM_REWARD_VERSION"] = "reward-v2"
    with pytest.raises(RuntimeError, match="non-extensible"):
        advance(run_suffix="run")


@pytest.mark.parametrize("fails", [False, True])
def test_optional_verifier_has_one_bounded_request_and_preserves_labels(monkeypatch, fails):
    patch = load_script("patch_offtopic_labels")
    source = '''import time
class Game:
    def evaluate(self, batch_labels_dict):
        if isinstance(batch_labels_dict, list):
            batch_labels_dict = {idx: label for idx, label in enumerate(batch_labels_dict)}

        return batch_labels_dict
'''
    patched, changed = patch.apply(source)
    assert changed
    assert patch.apply(patched) == (patched, False)
    legacy = patched.replace(patch.BOUNDED_CALL, patch.LEGACY_CALL)
    assert patch.apply(legacy) == (patched, True)
    response = Mock()
    response.json.return_value = {"labels": [{"game_idx": 0, "off_topic": True}]}
    post = Mock(return_value=response, side_effect=TimeoutError if fails else None)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))
    monkeypatch.setenv("OFFTOPIC_URL", "http://localhost/verify_topic")
    monkeypatch.delenv("OFFTOPIC_TIMEOUT_SECONDS", raising=False)
    namespace = {}
    exec(patched, namespace)
    game = namespace["Game"]()
    game.active_games = {0: {"prompts": "seed", "history": [{"content": "rewrite"}]}}
    game.strategy = SimpleNamespace(print=Mock(), get_rank=lambda: 0)
    labels = game.evaluate({0: {"prompt_harmfulness": "harmful"}})
    post.assert_called_once()
    assert post.call_args.kwargs["timeout"] == (3, 15.0)
    assert labels[0]["prompt_harmfulness"] == "harmful"
    assert ("off_topic" in labels[0]) is not fails


def executable(path, content):
    path.write_text("#!/usr/bin/env bash\n" + content)
    path.chmod(0o755)


@pytest.mark.parametrize("judge_ready,verifier_ready", [(False, False), (True, False), (True, True)])
def test_role_phase_health_failure_and_service_cleanup(tmp_path, judge_ready, verifier_ready):
    work, sft, commands = (tmp_path / name for name in ("work", "sft_data", "bin"))
    for path in (work, sft, commands):
        path.mkdir()
    (sft / "attacker_rewrite_1180_rl_continuation.jsonl").write_text("{}\n")
    pids = tmp_path / "service_pids"
    trained = tmp_path / "trained"
    wrapper = commands / "python"
    executable(wrapper, '''
if [[ "$*" == *serve_wildguard* || "$*" == *serve_offtopic.py* ]]; then
  echo "$$" >> "$MOCK_PIDS"
  exec /bin/sleep 3600
fi
if [[ "$*" == *train_ppo_ray* ]]; then
  echo "$OFFTOPIC_URL" > "$MOCK_TRAINED"
  exit 0
fi
exec ''' + shlex.quote(sys.executable) + ' "$@"\n')
    executable(commands / "curl", '''
/bin/sleep 0.05
if [[ "$*" == *classify* ]]; then
  [ "$MOCK_JUDGE_READY" = 1 ] || exit 7
  echo '{"labels": [{}]}'
else
  [ "$MOCK_VERIFIER_READY" = 1 ] || exit 7
  echo '{"ok":true}'
fi
''')
    executable(commands / "ray", "exit 0\n")
    executable(commands / "pgrep", "exit 1\n")
    executable(commands / "sleep", "exit 0\n")
    environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
        SSP_ROOT=str(tmp_path), SSP_PYTHON=str(wrapper), SSP_WORK=str(work),
        SSP_SFT_DIR=str(sft), SSP_OUT=str(tmp_path / "psro"),
        BASE_MODEL_PATH=str(tmp_path), JUDGE_MODEL_PATH=str(tmp_path),
        ROLE="attacker", GEN="1", TAG="A1", OFFTOPIC_ENABLE="1", DRY_RUN="0",
        JUDGE_STARTUP_ATTEMPTS="1", OFFTOPIC_STARTUP_ATTEMPTS="1",
        RAY_BIN=str(commands / "ray"), RAY_STOP_SCOPE="temp_dir", RAY_PORT=str(50000 + os.getpid() % 10000),
        MOCK_JUDGE_READY=str(int(judge_ready)), MOCK_VERIFIER_READY=str(int(verifier_ready)),
        MOCK_PIDS=str(pids), MOCK_TRAINED=str(trained))
    environment.pop("WANDB_API_KEY", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/role_phase.sh")],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) is judge_ready, result.stdout + result.stderr
    assert trained.exists() is judge_ready
    if not judge_ready:
        assert "judge did not become ready" in result.stderr
        assert "judge up" not in result.stdout
    else:
        assert bool(trained.read_text().strip()) is verifier_ready
    assert pids.exists()
    for pid in pids.read_text().splitlines():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid), 0)


@pytest.mark.parametrize("backend", ["local", "anthropic", "managed_local"])
def test_shared_binary_configures_the_topic_judge_before_the_reward_contract(tmp_path, backend):
    """The reward contract resolves the topic judge, so TOPIC_API_* must already
    be set when it runs -- not later, at the service launch.  When they were set
    at the launch site the default `local` backend died with "Set TOPIC_API_MODEL
    and TOPIC_API_KEY" before any service started, so this runs the real
    contract check (the wrapper execs a real interpreter for `-c`)."""
    work, sft, commands = (tmp_path / name for name in ("work", "sft_data", "bin"))
    for path in (work, sft, commands):
        path.mkdir()
    (sft / "attacker_rewrite_1180_rl_continuation.jsonl").write_text("{}\n")
    trained = tmp_path / "trained"
    wrapper = commands / "python"
    executable(wrapper, '''
if [[ "$*" == *serve_wildguard* || "$*" == *serve_topic_judge.py* ]]; then
  if [[ "$*" == *serve_topic_judge.py* && "${TOPIC_JUDGE_MANAGED:-0}" == 1 ]]; then
    echo unexpected > "$MOCK_TRAINED.duplicate"
    exit 99
  fi
  exec /bin/sleep 3600
fi
if [[ "$*" == *train_ppo_ray* ]]; then
  printf '%s\\n' "$TOPIC_API_PROVIDER" "$TOPIC_API_MODEL" "$TOPIC_API_BASE_URL" > "$MOCK_TRAINED"
  exit 0
fi
exec ''' + shlex.quote(sys.executable) + ' "$@"\n')
    executable(commands / "curl", '''
/bin/sleep 0.05
if [[ "$*" == *classify* ]]; then echo '{"labels": [{}]}'; else echo '{"ok":true}'; fi
''')
    executable(commands / "ray", "exit 0\n")
    executable(commands / "pgrep", "exit 1\n")
    executable(commands / "sleep", "exit 0\n")
    environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
        SSP_ROOT=str(tmp_path), SSP_PYTHON=str(wrapper), SSP_WORK=str(work),
        SSP_SFT_DIR=str(sft), SSP_OUT=str(tmp_path / "psro"),
        BASE_MODEL_PATH=str(tmp_path), JUDGE_MODEL_PATH=str(tmp_path),
        ROLE="attacker", GEN="1", TAG="A1", DRY_RUN="0",
        GAME_REWARD_MODE="shared_binary", TOPIC_JUDGE_BACKEND="local" if backend == "managed_local" else backend,
        JUDGE_STARTUP_ATTEMPTS="1", TOPIC_JUDGE_STARTUP_ATTEMPTS="1",
        RAY_BIN=str(commands / "ray"), RAY_STOP_SCOPE="temp_dir",
        RAY_PORT=str(50000 + os.getpid() % 10000), MOCK_TRAINED=str(trained))
    for name in ("WANDB_API_KEY", "TOPIC_API_PROVIDER", "TOPIC_API_BASE_URL",
                 "TOPIC_API_MODEL", "TOPIC_API_KEY", "TOPIC_API_KEY_FILE"):
        environment.pop(name, None)
    if backend == "anthropic":
        environment["TOPIC_API_KEY"] = "private-test-key"
    if backend == "managed_local":
        environment.update(TOPIC_JUDGE_MANAGED="1", TOPIC_API_BASE_URL="http://127.0.0.1:5410/v1",
                           TOPIC_API_MODEL="frozen-local-model")
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/role_phase.sh")],
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Set TOPIC_API_MODEL" not in result.stderr
    provider, model, base_url = trained.read_text().splitlines()
    if backend == "local":
        # no credential, and the base URL points at the locally started shim
        assert (provider, model) == ("openai_compatible", "local-topic-judge")
        assert base_url.endswith("/v1")
        assert "local topic judge up" in result.stdout
    elif backend == "managed_local":
        assert (provider, model, base_url) == ("openai_compatible", "frozen-local-model", "http://127.0.0.1:5410/v1")
        assert "reusing managed local topic judge" in result.stdout
        assert not Path(str(trained) + ".duplicate").exists()
    else:
        assert (provider, model) == ("anthropic", "claude-opus-5")
        # the endpoint is the client's default; the phase starts no local service
        assert base_url in ("", "https://api.anthropic.com")
        assert "hosted topic judge: claude-opus-5" in result.stdout
        assert "local topic judge up" not in result.stdout


def test_anthropic_backend_refuses_to_start_without_a_key(tmp_path):
    """A missing key must stop the phase, not fall through to an unlabeled run."""
    work, sft, commands = (tmp_path / name for name in ("work", "sft_data", "bin"))
    for path in (work, sft, commands):
        path.mkdir()
    (sft / "attacker_rewrite_1180_rl_continuation.jsonl").write_text("{}\n")
    wrapper = commands / "python"
    executable(wrapper, "exec " + shlex.quote(sys.executable) + ' "$@"\n')
    environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
        SSP_ROOT=str(tmp_path), SSP_PYTHON=str(wrapper), SSP_WORK=str(work),
        SSP_SFT_DIR=str(sft), SSP_OUT=str(tmp_path / "psro"),
        BASE_MODEL_PATH=str(tmp_path), JUDGE_MODEL_PATH=str(tmp_path),
        ROLE="attacker", GEN="1", TAG="A1", DRY_RUN="1",
        GAME_REWARD_MODE="shared_binary", TOPIC_JUDGE_BACKEND="anthropic")
    for name in ("TOPIC_API_KEY", "TOPIC_API_KEY_FILE", "WANDB_API_KEY"):
        environment.pop(name, None)
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/role_phase.sh")],
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "needs TOPIC_API_KEY" in result.stderr


def test_offtopic_fastapi_endpoint_accepts_request_body(monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient

    service = load_script("serve_offtopic")
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
    llm = SimpleNamespace(generate=lambda prompts, params: [
        SimpleNamespace(outputs=[SimpleNamespace(text="YES")]) for _ in prompts
    ])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda model: tokenizer)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=lambda **kw: llm, SamplingParams=lambda **kw: kw))
    monkeypatch.setenv("OFFTOPIC_MODEL", "fake-model")
    captured = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.append(app))
    assert service.main() == 0
    response = TestClient(captured[0]).post("/verify_topic", json={
        "queries": [{"game_idx": 0, "seed": "seed", "rewrite": "rewrite"}]
    })
    assert response.status_code == 200, response.text
    assert response.json()["labels"][0]["game_idx"] == 0


@pytest.mark.parametrize("script", ["submit_cold_psro.sh", "submit_single_oracle.sh"])
def test_slurm_submission_does_not_require_wandb(tmp_path, script):
    commands = tmp_path / "bin"
    commands.mkdir()
    executable(commands / "sbatch", 'echo "mock submission: $*"\n')
    state = tmp_path / "psro/run/state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{}")
    environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
        WORKSPACE=str(tmp_path), PSRO_ROOT=str(state.parent.parent),
        LOGDIR=str(tmp_path / "logs"), SUFFIX="run", TAG="D2",
        SOURCE_ORACLE="D2", STEPS="100", JUDGE_MODEL_PATH=str(tmp_path))
    environment.pop("WANDB_API_KEY", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm" / script)],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "mock submission:" in result.stdout
    assert "W&B disabled" in result.stderr


@pytest.mark.parametrize("explicit_resume", [False, True])
def test_default_kl_never_anchors_to_partial_adapter(tmp_path, monkeypatch, explicit_resume):
    driver = load_script("cold_psro_local")
    run = tmp_path / "psro/run"
    partial = run / "training/A1/ckpt/global_step10_hf"
    partial.mkdir(parents=True)
    weights = partial / "adapter_model.safetensors"
    weights.write_bytes(b"existing checkpoint")
    psro = SimpleNamespace(
        PSRO_OUTPUT_ROOT=run.parent, BASE_MODEL="base",
        _normalize_fixed_opponent_pool=lambda pool: pool,
        _adapter_checkpoint=lambda path: (path / "adapter_model.safetensors").is_file(),
    )

    class ReadyForFreshTraining(Exception):
        pass

    def base_env():
        raise ReadyForFreshTraining

    cfg = SimpleNamespace(init_mode="cold", base_env=base_env)
    trainer = driver.make_local_oracle_trainer(psro, cfg)
    monkeypatch.delenv("INIT_KL_COEF", raising=False)
    monkeypatch.delenv("PSRO_MIDRUN_RESUME", raising=False)
    if explicit_resume:
        monkeypatch.setenv("PSRO_MIDRUN_RESUME", "1")
    with pytest.raises(RuntimeError if explicit_resume else ReadyForFreshTraining):
        trainer(run_root=str(run), target="A1", role="attacker", opponent_pool=[],
                steps=100, lora_rank=64, lora_alpha=64, learning_rate=1e-5,
                sft_stop_after_step=30, sft_batches_per_step=1, save_steps=10,
                actor_lr_scheduler="constant_with_warmup", lr_warmup_ratio=0.05,
                training_seed=8888, wandb_identity="test", cold_start=True,
                role_start_label="A0", role_start_adapter=None, role_start_sha256="base:base")
    assert not (run / "training/A1.segment_10").exists()
    if explicit_resume:
        assert weights.read_bytes() == b"existing checkpoint"
