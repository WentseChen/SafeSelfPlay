"""External judge protocol, bounded failure and persistent label consistency."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roll.utils.topic_api import TopicAPIClient, TopicAPIConfig, TopicAPIError, parse_topic_label


class Response:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status
    def json(self):
        return self.payload


def config(tmp_path, provider="anthropic"):
    return TopicAPIConfig(provider=provider, base_url="https://judge.example/v1", model="test-fixed-model",
                          api_key="private-test-key", cache_path=str(tmp_path / "labels.sqlite3"))


@pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
def test_provider_labels_are_strict_binary_cached_and_preserve_game_ids(tmp_path, provider):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        text = '{"off_topic":false}'
        return Response({"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}
                        if provider == "anthropic" else
                        {"choices": [{"finish_reason": "stop", "message": {"content": text}}]})
    client = TopicAPIClient(config(tmp_path, provider), post=post)
    queries = [{"game_idx": i, "seed": "original", "rewrite": "a rewrite"} for i in (4, 9)]
    first = client.label_queries(queries)
    assert [v["game_idx"] for v in first] == [4, 9]
    assert all(v["off_topic"] is False for v in first)
    assert len(calls) == 1  # identical text pairs are deduplicated
    assert client.label_queries(queries) == first and len(calls) == 1
    assert calls[0][0].endswith("/v1/messages" if provider == "anthropic" else "/v1/chat/completions")
    assert calls[0][1]["allow_redirects"] is False
    assert "private-test-key" not in repr(client.config)
    assert "private-test-key" not in json.dumps(client.config.public_contract())
    other = TopicAPIClient(replace(client.config, model="different-model"), post=post)
    other.label_queries(queries)
    assert len(calls) == 2


@pytest.mark.parametrize("text", ['{"off_topic":"false"}', '{"off_topic":0}', '{}', 'OFF_TOPIC',
                                  '{"off_topic":false,"extra":1}', 'I cannot help with that'])
def test_non_boolean_and_refused_labels_are_not_reinterpreted(text):
    with pytest.raises(TopicAPIError):
        parse_topic_label(text)


@pytest.mark.parametrize("status,attempts", [(401, 1), (429, 2), (503, 2)])
def test_http_failures_are_bounded_and_never_cached(tmp_path, monkeypatch, status, attempts):
    monkeypatch.setattr("roll.utils.topic_api.time.sleep", lambda _: None)
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return Response(status=status)
    client = TopicAPIClient(config(tmp_path), post=post)
    with pytest.raises(TopicAPIError, match=f"HTTP {status}"):
        client.classify("original", "rewrite")
    assert len(calls) == attempts
    with client._database() as db:
        assert db.execute("SELECT count(*) FROM labels").fetchone()[0] == 0


def test_missing_configuration_fails_before_network(monkeypatch):
    for name in ("TOPIC_API_KEY", "TOPIC_API_KEY_FILE", "TOPIC_API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TopicAPIError, match="Set TOPIC_API_MODEL"):
        TopicAPIConfig.from_environment()


# --- labeler selection ------------------------------------------------------
# The local judge is the default labeler; Opus 5 is the optional alternative.
# Both answer the same prompt and return the same envelope, so the only thing
# that must not drift is *which* one a given run actually used.

def _environment(monkeypatch, tmp_path, **overrides):
    for name in ("TOPIC_API_PROVIDER", "TOPIC_API_BASE_URL", "TOPIC_API_MODEL",
                 "TOPIC_API_KEY", "TOPIC_API_KEY_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOPIC_API_CACHE", str(tmp_path / "labels.sqlite3"))
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)


def test_anthropic_defaults_to_opus_5(monkeypatch, tmp_path):
    _environment(monkeypatch, tmp_path, TOPIC_API_KEY="private-test-key")
    resolved = TopicAPIConfig.from_environment()
    assert resolved.provider == "anthropic"
    assert resolved.model == "claude-opus-5"
    assert resolved.base_url == "https://api.anthropic.com"


def test_local_backend_needs_no_hosted_credential(monkeypatch, tmp_path):
    _environment(monkeypatch, tmp_path, TOPIC_API_PROVIDER="openai_compatible",
                 TOPIC_API_BASE_URL="http://127.0.0.1:5410/v1", TOPIC_API_KEY="local")
    resolved = TopicAPIConfig.from_environment()
    assert resolved.model == "local-topic-judge"


def test_an_explicit_model_overrides_the_default(monkeypatch, tmp_path):
    _environment(monkeypatch, tmp_path, TOPIC_API_KEY="private-test-key",
                 TOPIC_API_MODEL="claude-opus-4-5")
    assert TopicAPIConfig.from_environment().model == "claude-opus-4-5"


def test_the_labeler_identity_is_part_of_the_cache_key(tmp_path):
    """Switching judges must re-label, not inherit the other judge's verdicts."""
    local = TopicAPIClient(replace(config(tmp_path, "openai_compatible"), model="local-topic-judge"))
    hosted = TopicAPIClient(replace(config(tmp_path), model="claude-opus-5"))
    assert local._key("seed", "rewrite") != hosted._key("seed", "rewrite")
    # ... and the run contract records which one produced the labels
    assert hosted.config.public_contract()["model"] == "claude-opus-5"


def test_opus5_short_label_disables_default_thinking_and_records_request_policy(tmp_path):
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response({"stop_reason": "end_turn", "content": [
            {"type": "text", "text": '{"off_topic":true}'}]})
    cfg = replace(config(tmp_path), model="claude-opus-5")
    client = TopicAPIClient(cfg, post=post)
    assert client.classify("Explain rain", "Write a cake recipe") is True
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert calls[0]["output_config"] == {"effort": "low"}
    assert calls[0]["max_tokens"] == 128
    assert cfg.public_contract()["request_options"] == cfg.request_options()
    # Keep the generic compatible/local endpoint free of Anthropic parameters.
    compatible = replace(cfg, provider="openai_compatible")
    assert compatible.request_options() == {"max_tokens": 128}
    assert "request_options" not in compatible.public_contract()
