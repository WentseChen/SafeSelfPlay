"""CPU tests of the handoff commands and full PSRO orchestration, without GPUs."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_reward_environment(monkeypatch):
    # main() exports defaults before exec; isolate that mutation in CPU tests.
    monkeypatch.setenv("GAME_REWARD_MODE", "legacy")
    monkeypatch.delenv("ATTACKER_INVALID_MOVE", raising=False)
    monkeypatch.setenv("TOPIC_PENALTY", "2")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    monkeypatch.setenv("TOPIC_API_PROVIDER", "anthropic")
    monkeypatch.setenv("TOPIC_API_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("TOPIC_API_MODEL", "claude-opus-5")


def test_new_container_recipe_defaults_to_the_invalid_move_reward(monkeypatch):
    monkeypatch.delenv("GAME_REWARD_MODE", raising=False)
    monkeypatch.delenv("ATTACKER_INVALID_MOVE", raising=False)
    monkeypatch.delenv("TOPIC_PENALTY", raising=False)
    env = entrypoint().environment()
    assert env["GAME_REWARD_MODE"] == "legacy"
    assert env["ATTACKER_INVALID_MOVE"] == "1"
    assert env["TOPIC_PENALTY"] == "2"
    assert env["TOPIC_JUDGE_BACKEND"] == "local"   # no credential needed; what the validated run used
    # The local backend points the client at the base model behind the shim.
    assert env["TOPIC_API_MODEL"] == "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    assert env["TOPIC_API_PROVIDER"] == "openai_compatible"


def entrypoint():
    spec = importlib.util.spec_from_file_location("llama8b_entrypoint", ROOT / "docker/llama8b_entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baked_image_defaults_use_the_current_local_recipe(monkeypatch):
    # Docker ENV overrides setdefault() in the entrypoint. Exercise the actual
    # baked environment so an old image default cannot silently win.
    dockerfile = (ROOT / "docker/Dockerfile.llama8b").read_text().replace("\\\n", " ")
    with monkeypatch.context() as image_env:
        for key in list(os.environ):
            image_env.delenv(key)
        for line in dockerfile.splitlines():
            if line.startswith("ENV "):
                for assignment in shlex.split(line[4:]):
                    key, value = assignment.split("=", 1)
                    image_env.setenv(key, value)
        env = entrypoint().environment()
    assert env["TOPIC_JUDGE_BACKEND"] == "local"
    assert (env["GAME_REWARD_MODE"], env["ATTACKER_INVALID_MOVE"]) == ("legacy", "1")
    assert env["TOPIC_PENALTY"] == "2"
    assert env["TOPIC_JUDGE_GPU"] == "3"
    assert env["TOPIC_JUDGE_UTIL"] == "0.13"
    assert (env["UTIL"], env["OPP_UTIL"], env["VLLM_SLEEP"]) == ("0.28", "0.26", "0")


@pytest.mark.parametrize("method,rounds,script", [
    ("psro", 8, "cold_psro_local.py"), ("uniform", 8, "history_selfplay.py"),
    ("naive", 6, "history_selfplay.py"),
])
def test_default_commands_and_dry_run_skip_gpu(tmp_path, monkeypatch, capsys, method, rounds, script):
    module = entrypoint()
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("SSP_ROOT", str(tmp_path))
    monkeypatch.setenv("SSP_WORK", "/opt/work")
    monkeypatch.setenv("SSP_SFT_DIR", "/opt/sft")
    for key in ("GENERATIONS", "INIT_KL_COEF", "STEPS_PER_ROLE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(module, "model_paths", lambda *a, **kw: {"JUDGE_MODEL_PATH": "/models/judge"})
    def unexpected_gpu():
        pytest.fail("dry-run must not use GPUs")
    monkeypatch.setattr(module, "check_gpus", unexpected_gpu)
    assert module.main([method, "--run-suffix", "test_run"]) == 0
    output = capsys.readouterr().out
    assert script in output
    assert f"--{'generations' if method == 'psro' else 'rounds'} {rounds}" in output
    if method == "uniform":
        assert "--steps 300" in output
    if method == "naive":
        assert "--steps 100" in output
    if method == "psro":
        assert "steps-per-role=300" in output
        assert os.environ["LADDER_EARLY_STOP"] == "1"
    assert not (tmp_path / ".llama8b-training.lock").exists()


@pytest.mark.parametrize("argv", [
    ["psro", "--run-suffix", "../bad"], ["uniform", "--run-suffix", "space name"],
    ["naive", "--run-suffix", "run", "--rounds", "0"],
    ["psro", "--run-suffix", "run", "--matrix-episodes", "255"],
])
def test_invalid_arguments_fail_before_model_access(monkeypatch, argv):
    module = entrypoint()
    monkeypatch.setattr(module, "model_paths", lambda *a, **kw: pytest.fail("model access before validation"))
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2


def test_settings_reentry_and_workspace_lock(tmp_path):
    module = entrypoint()
    env = {"SSP_ROOT": str(tmp_path), "PSRO_ROOT": str(tmp_path / "psro"), "HF_TOKEN": "test-secret"}
    args = SimpleNamespace(command="psro", run_suffix="run", generations=8, steps_per_role=100,
                           matrix_episodes=4000)
    module.record_run_settings(args, env)
    module.record_run_settings(args, env)
    assert "test-secret" not in (tmp_path / "psro/run/docker_run_settings.json").read_text()
    with pytest.raises(ValueError, match="settings changed"):
        module.record_run_settings(args, {**env, "INIT_KL_COEF": "0.01"})
    with pytest.raises(ValueError, match="settings changed"):
        module.record_run_settings(args, {**env, "DEF_THRESHOLD": "0.02"})
    with module.lock_workspace(env) as lock:
        assert os.get_inheritable(lock.fileno())
        with pytest.raises(RuntimeError, match="Another training job"):
            module.lock_workspace(env)
    with module.lock_workspace(env):
        pass


def test_fixed_wandb_configuration_overrides_host_settings_without_logging(tmp_path, monkeypatch, capsys):
    module = entrypoint()
    monkeypatch.setattr(module, "REPO", tmp_path)
    (tmp_path / "docker").mkdir()
    config = {"WANDB_API_KEY": "private-test-value", "WANDB_ENTITY": "original-team",
              "WANDB_PROJECT": "self-play", "WANDB_GROUP": "upstream-selfredteam-role-lora-v2"}
    (tmp_path / "docker/wandb.fixed.json").write_text(json.dumps(config))
    env = {"WANDB_API_KEY": "unrelated", "WANDB_GROUP": "unrelated", "WANDB_MODE": "disabled"}
    module.configure_wandb(env)
    assert all(env[k] == v for k, v in config.items())
    assert env["WANDB_MODE"] == "online"
    assert "private-test-value" not in capsys.readouterr().out


@pytest.mark.parametrize("method,dry_run,gpus", [
    ("prepare", False, False), ("psro", False, True), ("uniform", False, True),
    ("naive", False, True), ("psro", True, False), ("check", True, True),
])
def test_wrapper_gpu_selection_and_credentials(tmp_path, method, dry_run, gpus):
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "docker"
    mock.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    mock.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               HF_TOKEN="test-secret", SSP_DOCKER_WORKSPACE=str(tmp_path / "workspace"),
               HF_CHECKPOINT_REPO="test-org/checkpoints", HF_CHECKPOINT_PRIVATE="1",
               GPU_DEVICES="4,5,6,7", DRY_RUN=str(int(dry_run)),
               TOPIC_JUDGE_BACKEND="anthropic", TOPIC_API_MODEL="claude-opus-5",
               TOPIC_API_KEY="private-topic-test-key")
    command = ["bash", str(ROOT / "docker/run_llama8b.sh"), method]
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[args.index("--ulimit") + 1] == "nofile=65536:65536"
    assert ("--gpus" in args) == gpus
    if gpus:
        assert args[args.index("--gpus") + 1] == '"device=4,5,6,7"'
    assert "test-secret" not in result.stdout
    assert "HF_TOKEN" in args
    assert "HF_CHECKPOINT_REPO" in args and "HF_CHECKPOINT_PRIVATE" in args
    assert "TOPIC_JUDGE_BACKEND" in args and "TOPIC_API_MODEL" in args
    assert "TOPIC_API_KEY" in args and "private-topic-test-key" not in result.stdout
    assert args[-1] == method




def test_handoff_packaging_omits_untracked_caches_and_credentials(tmp_path):
    archive = tmp_path / "test.build.tar.gz"
    subprocess.run([sys.executable, str(ROOT / "docker/package_llama8b.py"), "--output", str(archive)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    import tarfile
    with tarfile.open(archive) as bundle:
        names = {member.name.split("/", 1)[1] for member in bundle.getmembers()}
        assert {"docker/Dockerfile.llama8b", "docker/README.llama8b.md", "scripts/slurm/history_selfplay.py",
                "roll/utils/history_selfplay.py", "TOPIC_JUDGE_AND_SHARED_REWARD.md", "START_HERE.md"} <= names
        assert "uv.lock" not in names
        assert "docker/wandb.fixed.json" not in names
        assert not any(name.startswith(("output/", ".git/", ".env", "hf/")) for name in names)


def test_local_judge_gets_a_card_of_its_own(monkeypatch):
    # Both judges on one card is what made four GPUs look insufficient: vLLM
    # charges the second engine for what the first already holds there, so it
    # is left less memory than its own weights.
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    for name in ("TOPIC_JUDGE_GPU", "JUDGE_GPU", "TOPIC_JUDGE_UTIL", "TOPIC_JUDGE_MAX_LEN"):
        monkeypatch.delenv(name, raising=False)
    env = entrypoint().environment()
    assert env["GPUS"] == "4"
    assert env["TOPIC_JUDGE_GPU"] != env.get("JUDGE_GPU", "0")
    assert 0 <= int(env["TOPIC_JUDGE_GPU"]) < int(env["GPUS"])
    # Measured: ~17 GiB of a 143771 MiB H200 per judge, i.e. 0.118 of a card.
    assert 0.118 < float(env["TOPIC_JUDGE_UTIL"]) <= 0.20
    assert env["TOPIC_JUDGE_MAX_LEN"] == "4096"


def test_the_hosted_judge_reserves_no_card(monkeypatch):
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "anthropic")
    monkeypatch.setenv("TOPIC_API_KEY", "private-test-key")
    for name in ("TOPIC_JUDGE_GPU", "TOPIC_JUDGE_MANAGED", "TOPIC_JUDGE_MAX_LEN"):
        monkeypatch.delenv(name, raising=False)
    env = entrypoint().environment()
    assert "TOPIC_JUDGE_GPU" not in env


def test_the_placement_knobs_reach_the_container(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "docker"
    mock.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    mock.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               SSP_DOCKER_WORKSPACE=str(tmp_path / "workspace"), DRY_RUN="1",
               TOPIC_JUDGE_BACKEND="local", TOPIC_JUDGE_GPU="3",
               TOPIC_JUDGE_MAX_LEN="4096", JUDGE_GPU="0")
    result = subprocess.run(["bash", str(ROOT / "docker/run_llama8b.sh"), "psro"],
                            env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    for name in ("TOPIC_JUDGE_GPU", "TOPIC_JUDGE_MAX_LEN", "JUDGE_GPU"):
        assert name in args, f"{name} is not forwarded into the container"


def test_local_pipeline_matches_the_validated_a3_settings(monkeypatch):
    """The container must train the way the 100-step A3 run actually trained."""
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    monkeypatch.setenv("TOPIC_JUDGE_BACKEND", "local")
    for name in ("TOPIC_PENALTY", "TOPIC_JUDGE_GPU", "TOPIC_JUDGE_UTIL", "UTIL",
                 "OPP_UTIL", "VLLM_SLEEP", "TOPIC_JUDGE_MAX_LEN"):
        monkeypatch.delenv(name, raising=False)
    env = entrypoint().environment()
    # Measured on 4xH200, not estimated: judges ~17 GiB each on separate cards,
    # engines resident, ~50s/step, off-topic 43% -> 87% against the control.
    assert env["TOPIC_PENALTY"] == "2"
    assert (env["UTIL"], env["OPP_UTIL"]) == ("0.28", "0.26")
    assert env["VLLM_SLEEP"] == "0"      # sleeping costs ~110s/step and is not needed
    assert env["TOPIC_JUDGE_UTIL"] == "0.13"
    assert env["TOPIC_JUDGE_MAX_LEN"] == "4096"
    assert env["GPUS"] == "4"


def test_four_h200s_are_the_documented_default(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "docker"
    mock.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    mock.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               SSP_DOCKER_WORKSPACE=str(tmp_path / "workspace"))
    env.pop("GPU_DEVICES", None)
    result = subprocess.run(["bash", str(ROOT / "docker/run_llama8b.sh"), "psro"],
                            env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[args.index("--gpus") + 1] == '"device=0,1,2,3"'


def test_check_reports_which_commit_is_baked_in(monkeypatch, capsys):
    # The repo is COPYed into the image, so a container's code is frozen at
    # build time while /workspace persists across images. Without a stamp a
    # stale image is indistinguishable from a fresh one.
    module = entrypoint()
    module.report_build_provenance({"JUDGE_GPU": "0", "TOPIC_JUDGE_GPU": "3"})
    printed = capsys.readouterr().out
    assert "image build:" in printed
    assert "judge placement:" in printed and "GPU 3" in printed


def test_check_refuses_an_image_that_predates_the_placement_fix(tmp_path, monkeypatch):
    # Running that image reproduces "No available memory for the cache blocks"
    # at step 0, which is an expensive way to learn you forgot to rebuild.
    module = entrypoint()
    stale = tmp_path / "scripts/slurm"
    stale.mkdir(parents=True)
    (stale / "role_phase.sh").write_text("JUDGE_UTIL=${JUDGE_UTIL:-0.13}\n")  # no TOPIC_JUDGE_GPU
    monkeypatch.setattr(module, "REPO", tmp_path)
    with pytest.raises(RuntimeError) as raised:
        module.report_build_provenance({})
    message = str(raised.value)
    assert "predates the judge-placement fix" in message
    assert "TOPIC_JUDGE_GPU" in message
    assert "run_llama8b.sh build" in message          # tells you how to fix it


def test_build_stamps_the_commit(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    mock = binary / "docker"
    mock.write_text(f"#!{sys.executable}\nimport sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    mock.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(ROOT / "docker/run_llama8b.sh"), "build"],
                            env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    build_args = [args[i + 1] for i, a in enumerate(args) if a == "--build-arg"]
    assert any(a.startswith("SSP_GIT_SHA=") and a != "SSP_GIT_SHA=" for a in build_args)
    assert any(a.startswith("SSP_BUILD_DATE=") for a in build_args)


def test_invalid_move_reward_runs_legacy_with_the_managed_topic_judge(monkeypatch):
    # The flag is only defined on the legacy reward, and it needs the in-loop
    # topic judge; without both the container would train on shared_binary or
    # fail with "managed topic endpoint is missing".
    monkeypatch.delenv("GAME_REWARD_MODE", raising=False)
    monkeypatch.setenv("ATTACKER_INVALID_MOVE", "1")
    for name in ("TOPIC_JUDGE_GPU", "JUDGE_GPU", "TOPIC_JUDGE_MANAGED"):
        monkeypatch.delenv(name, raising=False)
    env = entrypoint().environment()
    assert env["GAME_REWARD_MODE"] == "legacy"
    assert env["TOPIC_JUDGE_MANAGED"] == "1"
    assert env["TOPIC_API_PROVIDER"] == "openai_compatible"
    assert "ATTACKER_INVALID_MOVE" in (ROOT / "docker/run_llama8b.sh").read_text()


def test_invalid_move_reward_rejects_shared_binary(monkeypatch):
    monkeypatch.setenv("ATTACKER_INVALID_MOVE", "1")
    monkeypatch.setenv("GAME_REWARD_MODE", "shared_binary")
    with pytest.raises(ValueError):
        entrypoint().environment()
