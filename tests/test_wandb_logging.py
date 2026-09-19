"""Can a training run actually upload its logs when a W&B key is provided?

Two things have to hold, and each used to fail on its own:

  * the container has to receive the credential at all -- the repo's own configs
    set `track_with: wandb` with no `api_key`, so they depend entirely on an
    ambient `WANDB_API_KEY`, which `docker run` does not forward by default;
  * a credential that will never work has to say so immediately, instead of
    being retried five times as though it were a flaky network.
"""

import os
import subprocess
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREATE_CONTAINER = os.path.join(REPO, "docker", "create_container.sh")


def _run_create_container(tmp_path, environment):
    """Run create_container.sh with `sudo` stubbed out, and return its docker argv."""
    commands, argv = tmp_path / "bin", tmp_path / "argv"
    commands.mkdir()
    stub = commands / "sudo"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$ARGV_LOG"\n')
    stub.chmod(0o755)
    environment = {
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "ARGV_LOG": str(argv),
        **environment,
    }
    result = subprocess.run(
        ["bash", CREATE_CONTAINER, "probe"],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result, argv.read_text().splitlines()


def test_the_container_receives_a_key_the_host_has(tmp_path):
    result, argv = _run_create_container(tmp_path, {"WANDB_API_KEY": "secret-key"})
    assert "WANDB_API_KEY=secret-key" in argv
    assert argv[argv.index("WANDB_API_KEY=secret-key") - 1] == "-e"
    assert "warning" not in result.stderr


def test_a_host_without_a_key_is_warned_rather_than_passed_an_empty_one(tmp_path):
    environment = {k: "" for k in ("WANDB_API_KEY", "WANDB_ENTITY")}
    result, argv = _run_create_container(tmp_path, environment)
    # An empty -e would shadow anything the image or a later `wandb login` sets.
    assert not [a for a in argv if a.startswith("WANDB_")]
    assert "WANDB_API_KEY is not set" in result.stderr


def test_only_the_variables_the_host_sets_are_forwarded(tmp_path):
    result, argv = _run_create_container(
        tmp_path, {"WANDB_API_KEY": "k", "WANDB_ENTITY": "some-team"}
    )
    forwarded = [a for a in argv if a.startswith("WANDB_")]
    assert forwarded == ["WANDB_API_KEY=k", "WANDB_ENTITY=some-team"]


class _FakeWandb(types.ModuleType):
    """Stands in for wandb: `import wandb` inside WandbTracker.__init__ finds this."""

    def __init__(self, failures):
        super().__init__("wandb")
        self.failures = list(failures)
        self.attempts = 0
        self.errors = sys.modules["wandb.errors"]

    def login(self, key=None):
        return True

    def init(self, **kwargs):
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        run = types.SimpleNamespace(url="https://wandb.test/run")
        run.config = types.SimpleNamespace(update=lambda *a, **k: None)
        run.log = lambda *a, **k: None
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    errors = types.ModuleType("wandb.errors")

    class Error(Exception):
        pass

    errors.Error = Error
    errors.UsageError = type("UsageError", (Error,), {})
    errors.AuthenticationError = type("AuthenticationError", (Error,), {})
    errors.CommError = type("CommError", (Error,), {})
    monkeypatch.setitem(sys.modules, "wandb.errors", errors)

    def make(failures):
        module = _FakeWandb(failures)
        monkeypatch.setitem(sys.modules, "wandb", module)
        return module

    make.errors = errors
    return make


def test_a_missing_credential_fails_immediately_and_names_the_fix(fake_wandb, monkeypatch):
    from roll.utils import tracking

    monkeypatch.setattr(
        "time.sleep",
        lambda _: pytest.fail("a permanent credential error must not be retried"),
    )
    errors = fake_wandb.errors
    wandb = fake_wandb([errors.UsageError("No API key configured. Use `wandb login` to log in.")])

    with pytest.raises(errors.UsageError) as raised:
        tracking.WandbTracker({}, project="p")

    assert wandb.attempts == 1
    assert "No API key configured" in str(raised.value)
    assert "WANDB_API_KEY" in str(raised.value)
    # The one thing a containerised run needs to be told.
    assert "-e WANDB_API_KEY" in str(raised.value)


def test_a_rejected_credential_is_not_retried_either(fake_wandb, monkeypatch):
    from roll.utils import tracking

    monkeypatch.setattr("time.sleep", lambda _: None)
    errors = fake_wandb.errors
    wandb = fake_wandb([errors.AuthenticationError("api_key not valid")])

    with pytest.raises(errors.AuthenticationError):
        tracking.WandbTracker({}, project="p")
    assert wandb.attempts == 1


def test_a_flaky_network_is_still_retried(fake_wandb, monkeypatch):
    from roll.utils import tracking

    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    errors = fake_wandb.errors
    wandb = fake_wandb([errors.CommError("timed out"), errors.CommError("timed out")])

    tracker = tracking.WandbTracker({"seed": 1}, project="p", api_key="secret")
    assert wandb.attempts == 3
    assert slept == [5, 10]
    assert tracker.run.url == "https://wandb.test/run"
