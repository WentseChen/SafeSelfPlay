"""The local judge's free-text -> boolean step, which shapes the reward.

``serve_topic_judge.py`` imports vllm/fastapi inside ``main()``, so the parsing
function is importable on a CPU-only test node.
"""

import importlib.util
from pathlib import Path
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "serve_topic_judge",
    Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "serve_topic_judge.py",
)
serve_topic_judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_topic_judge)
read_verdict = serve_topic_judge.read_verdict


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"off_topic": true}', True),
        ('{"off_topic": false}', False),
        ('Here is my answer: {"off_topic": TRUE}', True),
        ("OFF_TOPIC", True),
        ("The rewrite is on-topic.", False),
        ("off topic", True),
    ],
)
def test_json_and_free_text_verdicts(text, expected):
    assert read_verdict(text) is expected


def test_the_first_verdict_wins_when_the_model_says_both():
    # "not off_topic, it is on_topic" -- a small model does this often enough
    # that falling through to None would drop usable labels.
    assert read_verdict("off_topic? no. on_topic.") is True
    assert read_verdict("on_topic, certainly not off_topic") is False


@pytest.mark.parametrize("text", ["", None, "I cannot help with that", "maybe"])
def test_an_unreadable_reply_is_none_not_on_topic(text):
    # None becomes HTTP 502 and the client raises; defaulting to False would
    # silently switch the penalty off for exactly the hardest cases.
    assert read_verdict(text) is None


def test_json_is_preferred_over_stray_prose():
    assert read_verdict('The seed is off_topic-ish but {"off_topic": false}') is False
