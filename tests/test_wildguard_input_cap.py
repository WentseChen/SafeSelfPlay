"""WildGuard judge must never see a query longer than its context, and a failed
generate() must not leak stale outputs into the next call (vLLM 0.8.2 behaviour)."""
import importlib, shutil, sys, types, pathlib, textwrap
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = pathlib.Path("/zfsauton/scratch/wentsec/coldpsro_slurm/upstream/openrlhf/cli/serve_wildguard.py")


class FakeTok:
    # 1 "token" per character: makes over-length inputs trivial to construct.
    def encode(self, text, add_special_tokens=False): return list(range(len(text)))
    def decode(self, ids, skip_special_tokens=True): return "x" * len(ids)


class FakeLLM:
    """Mimics vLLM 0.8.2: prompts are added one by one; an over-length prompt raises
    after the earlier ones were queued, and the next generate() returns them too."""
    def __init__(self, max_len): self.max_len, self.queue = max_len, []
    def get_tokenizer(self): return FakeTok()
    def _out(self, p):
        return types.SimpleNamespace(outputs=[types.SimpleNamespace(
            text="Harmful request: yes\nResponse refusal: yes\nHarmful response: no")])
    def generate(self, prompts, sampling_params=None):
        for p in prompts:
            if len(p) > self.max_len:
                raise ValueError(f"Prompt length of {len(p)} is longer than the maximum model length of {self.max_len}.")
            self.queue.append(p)
        return self._run_engine()
    def _run_engine(self, use_tqdm=False):
        out, self.queue = [self._out(p) for p in self.queue], []
        return out


def _load(tmp_path, patched):
    work = tmp_path / "work"
    (work / "openrlhf/cli").mkdir(parents=True)
    shutil.copy(UPSTREAM, work / "openrlhf/cli/serve_wildguard.py")
    if patched:
        sys.path.insert(0, str(ROOT))
        import scripts.slurm.cold_psro_common as c; c._stub_modal()
        mod = importlib.import_module("modal_upstream_selfredteam_role_lora")
        mod.UPSTREAM_WORK = work
        mod._patch_upstream_wildguard_input_cap()
    src = (work / "openrlhf/cli/serve_wildguard.py").read_text()
    # Isolate WildGuardWorker from ray/vllm/fastapi so it runs with the fake engine.
    ns = {}
    stub = textwrap.dedent('''
        import logging
        logger = logging.getLogger("wg")
        WILDGUARD_INPUT_FORMAT = "<s>{prompt}||{response}</s>"
        class _E:
            def __init__(self, v): self.value = v
        class PromptHarmfulness: HARMFUL=_E("harmful"); UNHARMFUL=_E("unharmful")
        class ResponseHarmfulness: HARMFUL=_E("harmful"); UNHARMFUL=_E("unharmful")
        class ResponseRefusal: REFUSAL=_E("refusal"); COMPLIANCE=_E("compliance")
        class SafetyClassifierOutput:
            def __init__(self, prompt_harmfulness=None, response_harmfulness=None, response_refusal=None, is_parsing_error=False):
                self.prompt_harmfulness=prompt_harmfulness; self.response_harmfulness=response_harmfulness
                self.response_refusal=response_refusal; self.is_parsing_error=is_parsing_error
        class SamplingParams:
            def __init__(self, **k): pass
    ''')
    body = src[src.index("class WildGuardWorker"):src.index("class WildGuardServerProxy")]
    body = body.replace("    def __init__(self, args, worker_id):\n        self.llm = LLM(",
                        "    def __init__(self, args, worker_id, _llm=None):\n        self.llm = _llm or LLM(")
    exec(stub + "\ndef LLM(**k): raise RuntimeError\n" + body, ns)
    return ns["WildGuardWorker"]


def _args(max_len=8192, batch_size=None):
    return types.SimpleNamespace(max_len=max_len, batch_size=batch_size, model_path="x", tensor_parallel_size=1,
                                 seed=0, max_num_seqs=8, bf16=True, enforce_eager=True, gpu_memory_utilization=0.5)


def _batch(n=32, long_at=28, long_len=14500):
    q = [{"game_idx": i, "prompt": "attack " * 10, "response": "reply " * 10} for i in range(n)]
    q[long_at]["response"] = "=" * long_len
    return q


def test_unpatched_reproduces_crash_and_stale_outputs(tmp_path):
    W = _load(tmp_path, patched=False)
    llm = FakeLLM(8192); w = W(_args(), 0, _llm=llm)
    with pytest.raises(ValueError, match="longer than the maximum model length"):
        w.classify(_batch())
    ok = [{"game_idx": i, "prompt": "a", "response": "b"} for i in range(32)]
    with pytest.raises(AssertionError, match="expected 32, got 60"):
        w.classify(ok)


@pytest.mark.parametrize("batch_size", [None, 16])
def test_patched_caps_every_query_and_returns_all_labels(tmp_path, batch_size):
    W = _load(tmp_path, patched=True)
    llm = FakeLLM(8192); w = W(_args(batch_size=batch_size), 0, _llm=llm)
    res = w.classify(_batch())
    assert sorted(res) == list(range(32))
    assert w.capped_moves == 1
    res2 = w.classify([{"game_idx": i, "prompt": "a", "response": "b"} for i in range(32)])
    assert sorted(res2) == list(range(32))


def test_patched_cap_respects_small_context(tmp_path):
    W = _load(tmp_path, patched=True)
    w = W(_args(max_len=2048), 0, _llm=FakeLLM(2048))
    assert w.move_max_tokens <= (2048 - 512) // 2
    q = [{"game_idx": 0, "prompt": "p" * 5000, "response": "r" * 5000}]
    assert sorted(w.classify(q)) == [0]


def test_patched_failure_does_not_leak_stale_outputs(tmp_path):
    W = _load(tmp_path, patched=True)
    llm = FakeLLM(8192); w = W(_args(), 0, _llm=llm)
    w.move_max_tokens = 10**9          # disable the cap to force a generate() failure
    with pytest.raises(ValueError):
        w.classify(_batch())
    assert llm.queue == []            # drained
    res = w.classify([{"game_idx": i, "prompt": "a", "response": "b"} for i in range(32)])
    assert sorted(res) == list(range(32))
