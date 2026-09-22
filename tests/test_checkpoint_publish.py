"""A finished role must never be lost to a failed upload."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from roll.utils import checkpoint_publish as publish


@pytest.fixture
def adapter(tmp_path):
    ckpt = tmp_path / "global_step100_hf"
    ckpt.mkdir()
    (ckpt / "adapter_model.bin").write_bytes(b"weights")
    return ckpt


def _api(recorder, *, fail_on: str | None = None):
    class Api:
        def __init__(self, token=None):
            recorder["token"] = token

        def create_repo(self, repo_id, **kwargs):
            recorder["repo"] = repo_id
            recorder["private"] = kwargs.get("private")

        def upload_folder(self, **kwargs):
            if fail_on == "folder":
                raise RuntimeError("hub is down")
            recorder["folder"] = kwargs

        def upload_file(self, **kwargs):
            recorder["file"] = kwargs
    return SimpleNamespace(HfApi=Api)


def test_disabled_without_a_repo(adapter, monkeypatch):
    monkeypatch.delenv("HF_CHECKPOINT_REPO", raising=False)
    assert publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=lambda _: None) is None


def test_missing_token_is_reported_not_raised(adapter, monkeypatch):
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    said = []
    assert publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=said.append) is None
    assert any("HF_TOKEN is unset" in line for line in said)


def test_uploads_under_run_suffix_and_target(adapter, monkeypatch):
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder))
    url = publish.publish_role_checkpoint(
        adapter, run_suffix="psro8_run1", target="A1", role="attacker",
        extra={"early_stop": "ran the full budget"}, log=lambda _: None)
    assert recorder["repo"] == "org/repo"
    assert recorder["folder"]["path_in_repo"] == "psro8_run1/A1"
    assert recorder["folder"]["folder_path"] == str(adapter)
    meta = json.loads(recorder["file"]["path_or_fileobj"])
    assert meta["target"] == "A1" and meta["role"] == "attacker"
    assert meta["checkpoint"] == "global_step100_hf"
    assert meta["early_stop"] == "ran the full budget"
    assert url.endswith("psro8_run1/A1")


def test_a_hub_failure_never_propagates(adapter, monkeypatch):
    """The whole point: a 300-step oracle survives a broken upload."""
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder, fail_on="folder"))
    said = []
    assert publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="D1", role="defender", log=said.append) is None
    assert any("publish FAILED for D1" in line for line in said)


def test_the_token_is_never_logged(adapter, monkeypatch):
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder, fail_on="folder"))
    said = []
    publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=said.append)
    assert "secret-token" not in "\n".join(said)


def test_defaults_to_a_private_repo(adapter, monkeypatch):
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.delenv("HF_CHECKPOINT_PRIVATE", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder))
    publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=lambda _: None)
    assert recorder["private"] is True


def test_the_trainer_publishes_the_final_checkpoint():
    """The hook sits after the adapter check, so early stops publish the
    checkpoint the monitor selected, not whatever landed last."""
    source = Path("scripts/slurm/cold_psro_local.py").read_text()
    hook = source.index("publish_role_checkpoint(")
    guard = source.index("finished without {checkpoint}")
    early = source.index("stopped early; final checkpoint is")
    assert early < guard < hook


def test_the_peft_card_is_not_uploaded(adapter, monkeypatch):
    """PEFT writes README.md with `base_model:` set to the absolute local path
    it trained from. That is not a model id, so the Hub rejects the entire
    commit with "Invalid metadata in README.md" and nothing lands."""
    (adapter / "README.md").write_text("---\nbase_model: /abs/local/path/\n---\n")
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder))
    publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=lambda _: None)
    assert "README.md" in recorder["folder"]["ignore_patterns"]


def test_metadata_records_a_usable_model_id(adapter, monkeypatch):
    recorder = {}
    monkeypatch.setenv("HF_CHECKPOINT_REPO", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv(
        "BASE_MODEL_PATH",
        "/scratch/hf/hub/models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/snapshots/368c/")
    monkeypatch.setitem(sys.modules, "huggingface_hub", _api(recorder))
    publish.publish_role_checkpoint(
        adapter, run_suffix="r", target="A1", role="attacker", log=lambda _: None)
    meta = json.loads(recorder["file"]["path_or_fileobj"])
    assert meta["base_model"] == "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
