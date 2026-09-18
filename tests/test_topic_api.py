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
