"""Cell judge teardown frees its card; the container re-enters a failed driver."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/slurm"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stubborn(marker_env=None, new_session=False):
    """A process that ignores SIGTERM, like a Ray actor mid-teardown."""
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
    env = dict(os.environ, **(marker_env or {}))
    return subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=new_session)


def test_stop_kills_the_group_even_when_the_leader_exits(monkeypatch, tmp_path):
    cpl = load("scripts/slurm/cold_psro_local.py", "cpl_under_test")
    monkeypatch.setattr(cpl, "_wait_for_free_gpu", lambda *a, **k: True)
    # leader exits on SIGTERM; its child in the same group ignores SIGTERM
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
         "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)']);"
         "time.sleep(600)"],
        start_new_session=True)
    time.sleep(1.5)
    group = os.getpgid(leader.pid)
    judge = cpl.Judge.__new__(cpl.Judge)
    judge.process, judge.gpu, judge.ray_tmpdir = leader, "0", str(tmp_path / "no-such-marker")
    judge.cfg = type("C", (), {"python": sys.executable})()
    judge.stop()
    time.sleep(0.5)
    alive = [p for p in Path("/proc").iterdir() if p.name.isdigit()
             and _pgid(int(p.name)) == group]
    assert alive == []


def test_stop_kills_ray_processes_that_left_the_group(monkeypatch, tmp_path):
    cpl = load("scripts/slurm/cold_psro_local.py", "cpl_under_test2")
    marker = str(tmp_path / "coldpsro-judge-5999")
    escaped = stubborn({"RAY_TMPDIR": marker}, new_session=True)   # own session: a group kill misses it
    bystander = stubborn()                                          # unrelated: must survive
    time.sleep(0.5)
    try:
        killed = cpl._kill_processes_marked(marker)
        assert escaped.pid in killed and bystander.pid not in killed
        assert escaped.wait(timeout=10) == -signal.SIGKILL
        assert bystander.poll() is None
    finally:
        bystander.kill(); escaped.kill()


def test_wait_for_free_gpu_polls_until_released(monkeypatch):
    cpl = load("scripts/slurm/cold_psro_local.py", "cpl_under_test3")
    readings = iter([0.40, 0.55, 0.97])                    # judge still holding, then gone
    monkeypatch.setattr(cpl, "_gpu_free_fraction", lambda *a: next(readings))
    monkeypatch.setattr(cpl.time, "sleep", lambda s: None)
    assert cpl._wait_for_free_gpu(sys.executable, "0", log=lambda m: None) is True


def test_wait_for_free_gpu_gives_up_without_raising(monkeypatch):
    cpl = load("scripts/slurm/cold_psro_local.py", "cpl_under_test4")
    monkeypatch.setenv("CELL_JUDGE_RELEASE_TIMEOUT", "0")
    monkeypatch.setattr(cpl, "_gpu_free_fraction", lambda *a: 0.40)
    assert cpl._wait_for_free_gpu(sys.executable, "0", log=lambda m: None) is False


def test_entrypoint_reenters_a_failed_driver(monkeypatch):
    ep = load("docker/llama8b_entrypoint.py", "ep_under_test")
    calls = []
    monkeypatch.setattr(ep, "run_driver_once", lambda cmd, env: calls.append(1) or (1 if len(calls) < 3 else 0))
    monkeypatch.setattr(ep.time, "sleep", lambda s: None)
    assert ep.run_with_retries(["driver"], {"DRIVER_MAX_ATTEMPTS": "5"}, True) == 0
    assert len(calls) == 3


def test_entrypoint_gives_up_after_the_attempt_budget(monkeypatch):
    ep = load("docker/llama8b_entrypoint.py", "ep_under_test2")
    calls = []
    monkeypatch.setattr(ep, "run_driver_once", lambda cmd, env: calls.append(1) or 2)
    monkeypatch.setattr(ep.time, "sleep", lambda s: None)
    assert ep.run_with_retries(["driver"], {"DRIVER_MAX_ATTEMPTS": "3"}, True) == 2
    assert len(calls) == 3


def test_role_and_dry_runs_are_not_retried(monkeypatch):
    ep = load("docker/llama8b_entrypoint.py", "ep_under_test3")
    calls = []
    monkeypatch.setattr(ep, "run_driver_once", lambda cmd, env: calls.append(1) or 1)
    assert ep.run_with_retries(["driver"], {"DRIVER_MAX_ATTEMPTS": "5"}, False) == 1
    assert len(calls) == 1


def test_a_signal_is_never_retried(monkeypatch):
    ep = load("docker/llama8b_entrypoint.py", "ep_under_test4")
    calls = []
    def stopped(cmd, env):
        calls.append(1)
        raise SystemExit(128 + signal.SIGTERM)
    monkeypatch.setattr(ep, "run_driver_once", stopped)
    with pytest.raises(SystemExit):
        ep.run_with_retries(["driver"], {"DRIVER_MAX_ATTEMPTS": "5"}, True)
    assert len(calls) == 1


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None
