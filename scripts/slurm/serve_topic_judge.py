"""Local topic judge behind an OpenAI-compatible FastAPI endpoint.

``roll/utils/topic_api.py`` speaks to a hosted judge (Anthropic, or anything
OpenAI-compatible) and requires a reply whose content is *exactly*
``{"off_topic": true|false}`` -- ``parse_topic_label`` rejects a string "false",
an integer 0, an extra key, or a refusal, and never substitutes a label.  That
strictness is right for a paid endpoint and unreachable for a small local model
asked to emit raw JSON.

This server closes that gap.  It runs the verifier locally on one GPU, reads the
verdict out of free text, and emits the canonical JSON envelope, so the training
loop can use a local model through the same client and cache as a hosted one:

    TOPIC_API_PROVIDER=openai_compatible
    TOPIC_API_BASE_URL=http://<host>:<port>/v1
    TOPIC_API_MODEL=local-topic-judge
    TOPIC_API_KEY=local                      # unused, the client always sends one

Launch:

    TOPIC_JUDGE_MODEL=/path/to/base TOPIC_JUDGE_PORT=5410 \\
        python scripts/slurm/serve_topic_judge.py

Only ``/v1/chat/completions`` is implemented, and only for this one task: the
system prompt is supplied by the client and the user content is the JSON pair
``{"original": ..., "rewrite": ...}``.  Anything else is a 400 rather than a
guess, because a wrong label here silently corrupts the reward.
"""

# NOTE: deliberately no `from __future__ import annotations`.  FastAPI resolves
# route parameter annotations at registration time; under postponed evaluation
# the `Request` annotation is only a string, and because fastapi is imported
# inside main() it cannot be resolved -- FastAPI then treats `request` as a
# *query* parameter and rejects every POST with HTTP 422.
import json
import logging
import os
import re
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("topic-judge")

# The local model answers in words; the client needs a strict JSON boolean.
_OFF = re.compile(r"\boff[_\s-]?topic\b", re.IGNORECASE)
_ON = re.compile(r"\bon[_\s-]?topic\b", re.IGNORECASE)
_JSON_TRUE = re.compile(r'"off_topic"\s*:\s*true', re.IGNORECASE)
_JSON_FALSE = re.compile(r'"off_topic"\s*:\s*false', re.IGNORECASE)

INSTRUCTION = (
    'Answer with exactly one JSON object and nothing else: {"off_topic": true} '
    'if the rewrite changes the task, or {"off_topic": false} if the task is '
    "preserved."
)


def read_verdict(text: str) -> bool | None:
    """Extract a boolean verdict, or None when the model did not give one.

    None is propagated as an HTTP 502 rather than a default: the client's
    contract is that a missing label fails loudly, and defaulting to
    "on-topic" would quietly switch the penalty off for every hard case.
    """
    if not text:
        return None
    if _JSON_TRUE.search(text):
        return True
    if _JSON_FALSE.search(text):
        return False
    off, on = _OFF.search(text), _ON.search(text)
    if off and not on:
        return True
    if on and not off:
        return False
    if off and on:                      # both appear: trust whichever came first
        return off.start() < on.start()
    return None


def main() -> int:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = os.environ.get("TOPIC_JUDGE_MODEL")
    if not model_path:
        raise SystemExit("set TOPIC_JUDGE_MODEL")
    port = int(os.environ.get("TOPIC_JUDGE_PORT", "5410"))
    served_name = os.environ.get("TOPIC_JUDGE_NAME", "local-topic-judge")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("TOPIC_JUDGE_UTIL", "0.60")),
        max_model_len=int(os.environ.get("TOPIC_JUDGE_MAX_LEN", "8192")),
        seed=42,
    )
    # Greedy: a sampled verdict would inject noise straight into the reward.
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=24)

    app = FastAPI()
    stats = {"served": 0, "off_topic": 0, "unreadable": 0}

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True, "model": served_name, **stats})

    @app.get("/v1/models")
    async def models():
        return JSONResponse({"object": "list", "data": [{"id": served_name, "object": "model"}]})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), None
        )
        if user is None:
            raise HTTPException(400, "no user message")
        try:
            pair = json.loads(user)
            seed, rewrite = pair["original"], pair["rewrite"]
        except (ValueError, KeyError, TypeError):
            # Refuse rather than guess: this endpoint serves exactly one task.
            raise HTTPException(400, 'user content must be {"original":...,"rewrite":...}')

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"{system}\n\n{INSTRUCTION}"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"original": seed, "rewrite": rewrite}, ensure_ascii=False
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = llm.generate([prompt], params)[0].outputs[0].text
        verdict = read_verdict(raw)
        stats["served"] += 1
        if verdict is None:
            stats["unreadable"] += 1
            logger.warning("unreadable verdict: %r", raw[:120])
            raise HTTPException(502, "local judge produced no readable verdict")
        stats["off_topic"] += int(verdict)

        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", served_name),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            # exactly the one field parse_topic_label accepts
                            "content": json.dumps({"off_topic": verdict}),
                        },
                    }
                ],
            }
        )

    logger.info("topic judge on :%d serving %s from %s", port, served_name, model_path)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
