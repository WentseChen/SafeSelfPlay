"""Local labels must stay available to both rollout and matrix workers."""

import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from roll.utils import local_topic_runtime as runtime
from roll.utils.topic_api import TopicAPIConfig

ROOT = Path(__file__).resolve().parents[1]


def entrypoint():
    spec = importlib.util.spec_from_file_location("local_label_entrypoint", ROOT / "docker/llama8b_entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_mode_resolves_before_contract_without_cloud_credentials(tmp_path, monkeypatch):
    module = entrypoint()
    monkeypatch.setattr(module, "configure_wandb", lambda env: None)
    monkeypatch.setenv("SSP_ROOT", str(tmp_path))
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    monkeypatch.setenv("TOPIC_API_MODEL", "claude-opus-5")
    monkeypatch.setenv("TOPIC_API_KEY", "must-not-reach-local-server")
    monkeypatch.setenv("TOPIC_API_KEY_FILE", "/missing/cloud-key")
    for key in ("TOPIC_LOCAL_MODEL", "TOPIC_JUDGE_UTIL", "UTIL", "OPP_UTIL", "VLLM_SLEEP"):
        monkeypatch.delenv(key, raising=False)
    env = module.environment()
    assert env["TOPIC_JUDGE_MANAGED"] == "1"
    assert env["TOPIC_API_KEY"] == "local" and "TOPIC_API_KEY_FILE" not in env
    assert env["TOPIC_API_MODEL"] == module.BASE_MODEL
    assert (env["TOPIC_JUDGE_UTIL"], env["UTIL"], env["OPP_UTIL"], env["VLLM_SLEEP"]) == ("0.16", "0.24", "0.22", "1")
    for key in ("TOPIC_API_PROVIDER", "TOPIC_API_BASE_URL", "TOPIC_API_MODEL", "TOPIC_API_KEY"):
        monkeypatch.setenv(key, env[key])
    assert TopicAPIConfig.from_environment().provider == "openai_compatible"
    env.update(SSP_WORK="/work", SSP_SFT_DIR="/sft", JUDGE_MODEL_PATH="/judge")
    args = SimpleNamespace(command="psro", run_suffix="local_test", generations=8, steps_per_role=300, matrix_episodes=4000)
    command = module.training_command(args, env)
    assert command[command.index("--cell-workers") + 1] == "2"
    # The matrix driver and every training phase inherit this same endpoint.
    assert env["TOPIC_API_BASE_URL"].startswith("http://127.0.0.1:")


@pytest.mark.parametrize("failure", [None, "startup", "during_training"])
def test_service_lifetime_spans_driver_and_cleans_both_process_groups(tmp_path, monkeypatch, failure):
    env = {"SSP_ROOT": str(tmp_path), "SSP_PYTHON": "python", "TOPIC_JUDGE_PORT": "0",
           "TOPIC_API_MODEL": "local-model", "TOPIC_API_BASE_URL": "http://127.0.0.1:0/v1"}
    service = SimpleNamespace(pid=1001, poll=lambda: 17 if failure == "during_training" else None)
    polls = iter([None, 0])
    trainer = SimpleNamespace(pid=1002, poll=lambda: next(polls), returncode=0)
    calls, stopped = [], []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return service if len(calls) == 1 else trainer
    def ready(*args):
        if failure == "startup":
            raise RuntimeError("startup failed")
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime, "wait_for_topic_service", ready)
    monkeypatch.setattr(runtime, "stop_process_group", lambda p: stopped.append(p.pid) if p else None)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    if failure:
        with pytest.raises(RuntimeError):
            runtime.run_with_local_topic(["python", "cold_psro_local.py"], env, ROOT)
    else:
        assert runtime.run_with_local_topic(["python", "cold_psro_local.py"], env, ROOT) == 0
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert calls[0][1]["env"]["TOPIC_JUDGE_NAME"] == env["TOPIC_API_MODEL"]
    assert stopped == ([1001] if failure == "startup" else [1002, 1001])
    if len(calls) > 1:
        assert calls[1][1]["env"]["TOPIC_API_BASE_URL"] == env["TOPIC_API_BASE_URL"]


def test_cleanup_reaches_children_even_after_parent_exit(monkeypatch):
    sent = []
    process = SimpleNamespace(pid=1234, wait=lambda timeout: 0)
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    runtime.stop_process_group(process)
    assert sent == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
