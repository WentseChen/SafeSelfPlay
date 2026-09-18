"""Exercise the production early-stop rules and safe checkpoint handoff."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/slurm"))
import cold_psro_local as local
from early_stop_monitor import EarlyStop


def feed(criterion, first, last, *, success=0.5, compliance=0.95, violation=0.05):
    for step in range(first, last + 1):
        values = {criterion.metric: success}
        if compliance is not None:
            values[criterion.guard_metric] = compliance
        if violation is not None:
            values[criterion.ceiling_metric] = violation
        criterion.observe(step, values)


def test_attacker_plateau_requires_three_windows_since_significant_improvement(monkeypatch):
    monkeypatch.setenv("LADDER_EARLY_STOP", "1")
    criterion = local._build_criterion("attacker")
    feed(criterion, 1, 20, success=0.4)
    feed(criterion, 21, 40, success=0.407)
    feed(criterion, 41, 60, success=0.412)
    assert criterion.best_step == 60
    feed(criterion, 61, 100, success=0.41)
    assert not criterion.triggered
    feed(criterion, 101, 120, success=0.41)
    assert criterion.triggered
    assert "PLATEAU" in criterion.reason


@pytest.mark.parametrize("missing", ["guard", "ceiling"])
def test_missing_protective_metric_never_counts_as_defender_success(monkeypatch, missing):
    monkeypatch.setenv("LADDER_EARLY_STOP", "1")
    criterion = local._build_criterion("defender")
    feed(criterion, 1, 100, success=0.005,
         compliance=None if missing == "guard" else 0.95,
         violation=None if missing == "ceiling" else 0.05)
    assert not criterion.triggered


def test_defender_guard_resets_streak_and_collapse_is_distinct(monkeypatch):
    monkeypatch.setenv("LADDER_EARLY_STOP", "1")
    criterion = local._build_criterion("defender")
    feed(criterion, 1, 20, success=0.005)
    feed(criterion, 21, 40, success=0.005, compliance=0.90)
    feed(criterion, 41, 80, success=0.005)
    assert not criterion.triggered
    feed(criterion, 81, 100, success=0.005)
    assert criterion.triggered
    collapsed = local._build_criterion("defender")
    feed(collapsed, 1, 40, success=0.0, violation=0.9)
    assert collapsed.collapsed and not collapsed.triggered
    assert collapsed.measurements[-1].window_end == 40


def test_gap_in_logged_windows_resets_success_streak(monkeypatch):
    monkeypatch.setenv("LADDER_EARLY_STOP", "1")
    criterion = local._build_criterion("defender")
    feed(criterion, 1, 20, success=0.005)
    feed(criterion, 41, 80, success=0.005)
    assert not criterion.triggered
    feed(criterion, 81, 100, success=0.005)
    assert criterion.triggered


def checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}")
    save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, str(path / "adapter_model.safetensors"))


def test_checkpoint_readiness_rejects_directory_only_and_truncated_weights(tmp_path):
    path = tmp_path / "global_step60_hf"
    path.mkdir()
    assert not local._checkpoint_ready(path)
    checkpoint(path)
    assert local._checkpoint_ready(path)
    weights = path / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1])
    assert not local._checkpoint_ready(path)


@pytest.mark.parametrize("outcome,expected", [("converged", 0), ("collapse", 2), ("incomplete", 1)])
def test_supervisor_waits_for_valid_checkpoint_and_does_not_accept_collapse(tmp_path, monkeypatch, outcome, expected):
    log = tmp_path / "training.log"
    checkpoints = tmp_path / "ckpt"
    target = checkpoints / "global_step4_hf"
    target.mkdir(parents=True)
    if outcome != "incomplete":
        checkpoint(target)
    criterion = EarlyStop(metric="success", guard_metric="compliance", threshold=0.01,
                          window=2, patience=2, min_step=4, min_coverage=1,
                          ceiling_metric="format", ceiling_max=0.1, abort_max=0.8)
    class Process:
        pid = 12345
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            return self.returncode
    process = Process()
    def launch(*args, **kwargs):
        violation = 0.9 if outcome == "collapse" else 0.05
        with log.open("a") as f:
            for step in range(1, 5):
                f.write(f"| {step}/300 [time, success=0.0, compliance=0.95, format={violation}]\n")
        return process
    monkeypatch.setattr(local.subprocess, "Popen", launch)
    monkeypatch.setattr(local.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(local.os, "getpgid", lambda pid: pid)
    def terminate(pid, signal):
        assert pid == process.pid
        process.returncode = -signal
    monkeypatch.setattr(local.os, "killpg", terminate)
    monkeypatch.setenv("EARLY_STOP_CKPT_WAIT", "0")
    code, note = local._supervise_training({}, log, criterion, checkpoints)
    assert code == expected
    assert process.returncode is not None
    if outcome == "incomplete":
        assert "incomplete" in note
    if outcome == "collapse":
        assert "COLLAPSE" in note
