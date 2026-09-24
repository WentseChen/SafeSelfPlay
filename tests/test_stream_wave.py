"""The streaming cell pipeline places every result at its episode's position."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "slurm"))
import cold_psro_common  # noqa: E402  (installs the Modal stub)

cold_psro_common._stub_modal()
payoff = pytest.importorskip("modal_upstream_v2_payoff")


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 250 for c in text]

    def __call__(self, texts, add_special_tokens=False, padding=False,
                 truncation=True, max_length=None):
        return {"input_ids": [self.encode(t)[:max_length] for t in texts]}


class FakePool:
    """Finishes requests out of order, a few per poll, like a real engine."""

    size = 2
    max_num_seqs = 3

    def __init__(self):
        self.pending = []
        self.submitted = []

    @property
    def in_flight(self):
        return len(self.pending)

    def capacity(self):
        return 4

    def submit(self, key, *, role, token_ids, seed, max_new_tokens, prompt_max_tokens):
        assert len(self.pending) < self.capacity()
        self.submitted.append((key, role, seed))
        self.pending.append((key, role, seed))

    def poll(self, timeout=1.0):
        # Longest-waiting last: reverse order exercises out-of-order completion.
        done, self.pending = self.pending[-2:][::-1], self.pending[:-2]
        return [(key, [seed % 7], f"{role}:{seed}", "stop", None) for key, role, seed in done]


def run(monkeypatch, count, cached=()):
    judged_batches = []

    def fake_classify(url, queries, *, batch_size):
        judged_batches.append([q["game_idx"] for q in queries])
        return [
            {"game_idx": q["game_idx"], "prompt_harmfulness": "harmful",
             "echo": q["response"]}
            for q in queries
        ]

    monkeypatch.setattr(payoff, "_classify_wildguard", fake_classify)
    monkeypatch.setattr(payoff, "label_topic_queries",
                        lambda qs: [{"game_idx": q["game_idx"], "off_topic": False} for q in qs])
    monkeypatch.setattr(payoff, "decode_upstream_vllm_completion",
                        lambda tok, ids: "decoded")
    monkeypatch.setattr(payoff, "parse_prefilled_cot_completion",
                        lambda text: {"answer": text, "cot_format_violation": False})
    monkeypatch.setattr(payoff, "_render_defender_prompt",
                        lambda tok, attack, direct_base_defender: f"D[{attack}]")
    specs = [
        {"candidate_index": 100 + i, "candidate_seed": 1000 + i,
         "prompt_type": "generated_benign" if i % 2 else "generated_harmful",
         "seed_prompt": f"s{i}"}
        for i in range(count)
    ]
    cached_attacks = [
        (({"answer": f"cached{i}"}), {"text": f"cached{i}"}) if i in cached else None
        for i in range(count)
    ]
    fresh = []
    pool = FakePool()
    result = payoff._stream_wave(
        pool, FakeTokenizer(), specs=specs,
        attacker_prompts=[f"A{i}" for i in range(count)],
        cached_attacks=cached_attacks, direct_base_defender=False,
        max_new_tokens=8, remote_rm_url="http://x", judge_batch_size=3,
        want_topic=True, on_fresh_attack=lambda pos, parsed, out: fresh.append(pos),
    )
    return result, pool, judged_batches, fresh


def test_every_episode_lands_at_its_position(monkeypatch):
    (att_out, att, def_out, defense, labels), pool, batches, fresh = run(monkeypatch, 11)
    for i in range(11):
        seed = 1000 + i
        assert att_out[i]["vllm_raw_text"] == f"attacker:{seed}"
        assert def_out[i]["vllm_raw_text"] == f"defender:{seed + 1_000_000_000}"
        assert labels[i]["game_idx"] == 100 + i
        assert labels[i].get("off_topic") is (False if i % 2 == 0 else None)
    assert sorted(fresh) == list(range(11))
    assert all(len(b) <= 3 for b in batches)
    assert sorted(g for b in batches for g in b) == [100 + i for i in range(11)]


def test_cached_attacks_skip_attacker_generation(monkeypatch):
    (att_out, att, _, _, labels), pool, _, fresh = run(monkeypatch, 6, cached={0, 3, 4})
    roles = [(key, role) for key, role, _ in pool.submitted]
    assert {key for key, role in roles if role == "attacker"} == {"a1", "a2", "a5"}
    assert {key for key, role in roles if role == "defender"} == {f"d{i}" for i in range(6)}
    assert att[3]["answer"] == "cached3" and sorted(fresh) == [1, 2, 5]
    assert all(label is not None for label in labels)


def test_benign_drift_is_not_topic_labelled(monkeypatch):
    (_, _, _, _, labels), *_ = run(monkeypatch, 4)
    # fake WildGuard labels every prompt harmful, so benign seeds drifted
    assert "off_topic" not in labels[1] and "off_topic" not in labels[3]
    assert labels[0]["off_topic"] is False
