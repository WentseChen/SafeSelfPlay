"""Local labels must stay available to both rollout and matrix workers."""

import importlib.util
from pathlib import Path
import signal
import os
import sys
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
    # The settings of the validated 4xH200 A3 run: judge ~17 GiB on its own
    # card, engines resident. See test_docker_llama8b for the measurement.
    assert (env["TOPIC_JUDGE_UTIL"], env["UTIL"], env["OPP_UTIL"], env["VLLM_SLEEP"]) == ("0.13", "0.28", "0.26", "0")
    assert env["TOPIC_JUDGE_GPU"] != env.get("JUDGE_GPU", "0")
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


def test_a_busy_topic_judge_port_explains_itself(tmp_path, monkeypatch):
    """A crashed run leaves a judge holding the port; the container always
    reuses the same default, so this is the first thing the next run hits."""
    import socket
    from roll.utils.local_topic_runtime import run_with_local_topic

    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        env = {"TOPIC_JUDGE_PORT": str(port), "SSP_ROOT": str(tmp_path),
               "SSP_PYTHON": sys.executable, "TOPIC_API_MODEL": "local-topic-judge"}
        with pytest.raises(RuntimeError) as raised:
            run_with_local_topic(["/bin/true"], env, Path("/nonexistent"))
    finally:
        holder.close()
    message = str(raised.value)
    assert str(port) in message                 # which port
    assert "already in use" in message          # what is wrong
    assert "TOPIC_JUDGE_PORT" in message        # how to proceed
    # and it must not quietly adopt whatever is already answering there
    assert "did not start" in message


def test_the_drivers_resolve_the_local_judge_before_the_contract(monkeypatch):
    """shared_game_reward_contract() resolves the topic judge config, which
    role_phase.sh only sets later. history_selfplay died before training and
    cold_psro_local died AFTER a role had finished -- discarding a completed
    oracle -- while the container survived because its entrypoint exports the
    same values first."""
    from roll.utils.shared_game_reward import shared_game_reward_contract
    from roll.utils.topic_api import configure_local_topic_backend

    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    monkeypatch.setenv("JUDGE_PORT", "5591")
    for name in ("TOPIC_API_PROVIDER", "TOPIC_API_BASE_URL", "TOPIC_API_MODEL",
                 "TOPIC_API_KEY", "TOPIC_API_KEY_FILE", "TOPIC_JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Exception):
        shared_game_reward_contract()
    configure_local_topic_backend()
    contract = shared_game_reward_contract()["topic_judge"]
    assert contract["provider"] == "openai_compatible"
    assert contract["model"] == "local-topic-judge"
    # role_phase.sh uses JUDGE_PORT + 300; the base URL is part of the public
    # contract, so a mismatch would look like a contract change on resume.
    assert contract["base_url"].endswith(":5891/v1")
    assert "127.0.0.1" not in contract["base_url"]


def test_an_existing_configuration_is_left_alone(monkeypatch):
    from roll.utils.topic_api import configure_local_topic_backend

    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    monkeypatch.setenv("TOPIC_API_MODEL", "frozen-local-model")
    monkeypatch.setenv("TOPIC_API_KEY", "local")
    monkeypatch.setenv("TOPIC_API_BASE_URL", "http://127.0.0.1:5410/v1")
    configure_local_topic_backend()
    assert os.environ["TOPIC_API_MODEL"] == "frozen-local-model"
    assert os.environ["TOPIC_API_BASE_URL"] == "http://127.0.0.1:5410/v1"


def test_the_hosted_backend_is_not_touched(monkeypatch):
    from roll.utils.topic_api import configure_local_topic_backend

    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "anthropic")
    monkeypatch.delenv("TOPIC_API_PROVIDER", raising=False)
    configure_local_topic_backend()
    assert "TOPIC_API_PROVIDER" not in os.environ
