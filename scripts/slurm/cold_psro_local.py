#!/usr/bin/env python3
"""Slurm-native driver for the cold role-LoRA zero-sum PSRO continuation.

On Modal the controller re-detaches itself after every durable action
(``advance_cold_psro.spawn``).  Here the same function is called in an ordinary
loop: ``spawn`` is a no-op (see cold_psro_common), so one call advances exactly
one cell / oracle and returns, and this process decides what runs next.

Two things Modal provided as separate containers are provided here instead:

* the WildGuard judge, run on GPU 0 for the payoff-cell phases and stopped
  before a training phase, because scripts/slurm/role_phase.sh starts its own
  judge alongside the four-GPU trainer;
* the payoff cells themselves, which are prefetched three at a time on GPUs 1-3
  before re-entering the coordinator.  The coordinator then finds each raw cell
  already complete and finishes it on CPU, so the schedule it walks is
  unchanged -- only the wall-clock is.

The whole loop is idempotent: state.json is the only durable progress record,
every completed cell and oracle is skipped on re-entry, and an interrupted
oracle restarts from the base adapter exactly as it would on Modal (the run
contract pins save_steps to the final step, so there is no mid-role checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402

SLURM_DIR = Path(__file__).resolve().parent
ROLE_PHASE = SLURM_DIR / "role_phase.sh"
CELL_WORKER = SLURM_DIR / "payoff_cell_worker.py"
SELECT_CHECKPOINT = SLURM_DIR / "select_checkpoint.py"


def log(message: str) -> None:
    print(f"[driver {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# --- judge -------------------------------------------------------------------


class Judge:
    """WildGuard on GPU 0, shared by every concurrent payoff-cell worker."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.process: subprocess.Popen[bytes] | None = None
        self.url = f"http://127.0.0.1:{cfg.cell_judge_port}/classify"

    def _healthy(self) -> bool:
        payload = json.dumps(
            {
                "queries": [{"prompt": "hi", "response": "hello", "game_idx": 0}],
                "prompts": None,
            }
        ).encode()
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return b"labels" in response.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def start(self) -> str:
        if self.process is not None and self.process.poll() is None:
            return self.url
        self.cfg.judge_log.parent.mkdir(parents=True, exist_ok=True)
        environment = self.cfg.base_env()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["RAY_TMPDIR"] = f"/tmp/coldpsro-judge-{self.cfg.cell_judge_port}"
        command = [
            self.cfg.python, "-m", "openrlhf.cli.serve_wildguard",
            "--model_path", self.cfg.judge_model,
            "--port", str(self.cfg.cell_judge_port),
            "--host", "127.0.0.1",
            "--bf16", "--num_gpus", "1", "--tensor_parallel_size", "1",
            "--enforce_eager", "--seed", "42",
            # 8192 is the upstream value; a long transcript returns 500 at 4096
            # and takes the whole server down mid-cell.
            "--max_len", "8192",
            "--max_num_seqs", "64", "--batch_size", "64",
            "--gpu_memory_utilization", str(self.cfg.cell_judge_util),
        ]
        log(f"starting cell judge on GPU 0 -> {self.url}")
        handle = self.cfg.judge_log.open("ab")
        self.process = subprocess.Popen(
            command, cwd=str(self.cfg.work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        for attempt in range(180):
            if self._healthy():
                log(f"cell judge up after ~{attempt * 10}s")
                return self.url
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"cell judge exited rc={self.process.returncode}; "
                    f"see {self.cfg.judge_log}"
                )
            time.sleep(10)
        raise RuntimeError("cell judge did not become healthy within 30 minutes")

    def stop(self) -> None:
        if self.process is None:
            return
        log("stopping cell judge")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.process = None
        # vLLM releases HBM only once the process group is fully reaped.
        time.sleep(20)


# --- configuration ------------------------------------------------------------


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.workspace: Path = args.workspace.resolve()
        self.work: Path = args.work.resolve()
        self.repo: Path = SLURM_DIR.parents[1]
        self.hf_home: Path = args.hf_home.resolve()
        self.sft_dir: Path = args.sft_dir.resolve()
        self.python: str = args.python
        self.suffix: str = args.suffix
        self.psro_root: Path = args.psro_root.resolve()
        self.raw_payoff_root: Path = args.raw_payoff_root.resolve()
        self.run_root: Path = self.psro_root / self.suffix
        self.gpus: int = args.gpus
        self.cell_workers: int = args.cell_workers
        self.cell_judge_port: int = args.cell_judge_port
        self.cell_judge_util: float = args.cell_judge_util
        self.train_judge_port: int = args.train_judge_port
        self.ray_port: int = args.ray_port
        self.judge_log: Path = self.run_root / "logs" / "cell_judge.log"
        self.cell_log_dir: Path = self.run_root / "logs" / "cells"
        self.judge_model: str = str(args.judge_model)
        self.coordinator: str = args.coordinator
        self.init_mode: str = args.init_mode
        self.stop_before_evaluation: bool = not args.run_official_evaluation

    def apply_to_process(self) -> None:
        """Put the child environment on this process too.

        The coordinator evaluates a cell in-process whenever a prefetch worker
        is missing, and the Slurm job inherits the login profile's HF_HOME and
        TRANSFORMERS_CACHE, which point at a different, incomplete cache.
        """

        os.environ.pop("TRANSFORMERS_CACHE", None)
        os.environ.update(self.base_env())

    def base_env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.pop("TRANSFORMERS_CACHE", None)
        environment.update(
            {
                "HF_HOME": str(self.hf_home),
                "HF_HUB_CACHE": str(self.hf_home / "hub"),
                "HF_HUB_DISABLE_XET": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_CUMEM_ENABLE": "0",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                # Console scripts (ray, deepspeed) must come from the same
                # environment as --python; a stale ~/.local/bin copy shadows
                # them otherwise.
                "PATH": ":".join(
                    part for part in (
                        str(Path(self.python).parent),
                        os.environ.get("PATH", ""),
                    ) if part
                ),
                "PYTHONPATH": ":".join(
                    part for part in (
                        str(self.work), str(self.repo),
                        os.environ.get("PYTHONPATH", ""),
                    ) if part
                ),
            }
        )
        return environment


# --- local oracle trainer ------------------------------------------------------


def _build_criterion(role: str):
    """Early-stopping criterion for one oracle, or None to run the full budget.

    Defenders use the absolute rule: their own objective saturates, so stop once
    the attacker's success rate stays low, guarded by benign compliance and a
    CoT-format ceiling (a degenerate defender drives attack success to zero by
    emitting unparseable text, which looks identical to a win).

    Attackers use a plateau rule instead. No attacker in either arm exceeded
    ~0.57 success against a trained defender, so an absolute threshold would
    never fire and every attacker would burn the whole budget; stopping on "no
    improvement for `patience` windows" is the rule that actually binds.
    """
    if os.environ.get("LADDER_EARLY_STOP", "1") == "0":
        return None
    from early_stop_monitor import EarlyStop

    window = int(os.environ.get("EARLY_STOP_WINDOW", "20"))
    if role == "attacker":
        return EarlyStop(
            metric="attacker/info/successful_harmful_request_rate",
            guard_metric="",
            ceiling_metric="attacker/cot_format_violation",
            ceiling_max=float(os.environ.get("ATK_CEILING_MAX", "0.30")),
            abort_max=0.80, abort_patience=2,
            mode="plateau",
            min_delta=float(os.environ.get("ATK_MIN_DELTA", "0.01")),
            window=window,
            patience=int(os.environ.get("ATK_PATIENCE", "3")),
            min_step=int(os.environ.get("ATK_MIN_STEP", "60")),
        )
    return EarlyStop(
        metric="attacker/info/successful_harmful_request_rate",
        guard_metric="defender/wildguard_actual_benign_compliance_acc",
        threshold=float(os.environ.get("DEF_THRESHOLD", "0.01")),
        guard_min=float(os.environ.get("DEF_GUARD_MIN", "0.912")),
        ceiling_metric="defender/cot_format_violation",
        ceiling_max=0.10, abort_max=0.80, abort_patience=2,
        window=window,
        patience=int(os.environ.get("DEF_PATIENCE", "3")),
        min_step=int(os.environ.get("DEF_MIN_STEP", "60")),
    )


def _supervise_training(environment, training_log: Path, criterion, ckpt_dir: Path):
    """Run role_phase.sh, tail its log, and stop the run when the criterion fires.

    The trainer is started in its own process group so the whole tree can be
    signalled; on a trigger we wait for the checkpoint at the stopping step to
    appear before terminating, so the run always leaves a usable adapter.
    """
    import signal
    from early_stop_monitor import metric_re, parse_line

    names = [criterion.metric]
    if criterion.guard_metric:
        names.append(criterion.guard_metric)
    if criterion.ceiling_metric:
        names.append(criterion.ceiling_metric)
    patterns = {n: metric_re(n) for n in names}

    handle = training_log.open("ab")
    process = subprocess.Popen(
        ["bash", str(ROLE_PHASE)], cwd=str(environment.get("SSP_WORK", ".")),
        env=environment, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    offset, remainder, stop_step = 0, "", None
    try:
        # stop_step in the condition is load-bearing: without it the loop runs
        # to the trainer's natural exit and the terminate branch below never
        # gets to cut the run short.
        while process.poll() is None and stop_step is None:
            time.sleep(5)
            if not training_log.is_file():
                continue
            with training_log.open("r", errors="ignore") as reader:
                reader.seek(offset)
                chunk = reader.read()
                offset = reader.tell()
            if not chunk:
                continue
            parts = re.split(r"[\r\n]+", remainder + chunk)
            remainder = parts.pop() if parts else ""
            for line in parts:
                got = parse_line(line, patterns)
                if got is None:
                    continue
                step, values = got
                criterion.observe(step, values)
                fired = (criterion.streak >= criterion.patience
                         and step >= criterion.min_step)
                if (fired or criterion.collapsed) and stop_step is None:
                    stop_step = step
                    criterion.triggered = not criterion.collapsed
                    break
        if stop_step is None:
            return process.wait(), "ran the full budget"
        # let the checkpoint at the stopping step land before killing the tree
        deadline = time.time() + float(os.environ.get("EARLY_STOP_CKPT_WAIT", "900"))
        target = ckpt_dir / f"global_step{stop_step}_hf"
        while time.time() < deadline and not target.is_dir() and process.poll() is None:
            time.sleep(10)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=180)
        except Exception:
            try: os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception: pass
        kind = "COLLAPSE" if criterion.collapsed else "converged"
        return 0, f"{kind} at step {stop_step} ({criterion.reason or criterion.mode})"
    finally:
        handle.close()


def make_local_oracle_trainer(psro: Any, cfg: Config):
    """Replace the Modal oracle function with a role_phase.sh invocation.

    Everything the coordinator relies on is preserved: the pool is normalized
    (which re-verifies every frozen opponent's weight SHA), an already-finished
    target is returned without retraining, and the manifest carries the three
    fields the idempotence check compares.
    """

    def train_cold_psro_oracle(
        *,
        run_root: str,
        target: str,
        role: str,
        opponent_pool: list[dict[str, object]],
        steps: int,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        sft_stop_after_step: int,
        sft_batches_per_step: int,
        save_steps: int,
        actor_lr_scheduler: str,
        lr_warmup_ratio: float,
        training_seed: int,
        wandb_identity: str,
    ) -> dict[str, Any]:
        expected_prefix = "A" if role == "attacker" else "D"
        if not re.fullmatch(rf"{expected_prefix}[1-9][0-9]*", target):
            raise ValueError(f"invalid cold PSRO target: {target!r}")
        root = Path(run_root)
        if root == psro.PSRO_OUTPUT_ROOT or psro.PSRO_OUTPUT_ROOT not in root.parents:
            raise ValueError("run_root must be below the PSRO output root")
        normalized_pool = psro._normalize_fixed_opponent_pool(opponent_pool)
        run_dir = root / "training" / target
        checkpoint = run_dir / "ckpt" / f"global_step{steps}_hf"
        manifest_path = run_dir / "manifest.json"

        # Cold start: every oracle begins at the base model with a fresh
        # adapter. Hot start changes exactly this one thing -- the oracle
        # continues the previous same-role adapter, and only the first
        # generation of each role starts from base. The opponent rule, the
        # payoff matrix and the meta-game solve are untouched.
        role_start_adapter: str | None = None
        generation = int(target[1:])
        if cfg.init_mode == "hot" and generation > 1:
            parent = f"{target[0]}{generation - 1}"
            state = psro._read_json(root / "state.json")
            parent_record = state["population"].get(parent)
            if not parent_record or not parent_record.get("path"):
                raise RuntimeError(f"hot start for {target} needs {parent}")
            role_start_adapter = str(parent_record["path"])
            if not psro._adapter_checkpoint(Path(role_start_adapter)):
                raise FileNotFoundError(
                    f"missing hot-start adapter for {target}: {role_start_adapter}"
                )
        expected_role_start = role_start_adapter or psro.BASE_MODEL

        if psro._adapter_checkpoint(checkpoint):
            manifest = psro._read_json(manifest_path)
            if (
                manifest.get("role_start") != expected_role_start
                or manifest.get("fixed_opponent_pool") != normalized_pool
                or manifest.get("steps") != steps
            ):
                raise RuntimeError(f"existing {target} checkpoint contract changed")
            log(f"{target} already trained; reusing {checkpoint}")
            return psro._adapter_identity(checkpoint)

        # A preempted attempt leaves a partial ckpt tree and a manifest that
        # does not describe a finished role. Both are reproducible; drop them.
        if run_dir.exists():
            log(f"{target} restarting; clearing partial {run_dir}")
            # rmtree races with NFS silly-rename (.nfsXXXX) when a preempted
            # process still holds a log handle. Rename first -- that always
            # succeeds -- then delete best-effort.
            discarded = run_dir.with_name(f".discarded_{target}_{os.getpid()}")
            run_dir.rename(discarded)
            shutil.rmtree(discarded, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        environment = cfg.base_env()
        environment.update(
            {
                "SSP_ROOT": str(cfg.workspace),
                "SSP_WORK": str(cfg.work),
                "SSP_HF_HOME": str(cfg.hf_home),
                "SSP_SFT_DIR": str(cfg.sft_dir),
                "SSP_OUT": str(root / "training"),
                "SSP_PYTHON": cfg.python,
                "ROLE": role,
                "TAG": target,
                "GEN": target[1:],
                "STEPS": str(steps),
                "GPUS": str(cfg.gpus),
                "LORA_RANK": str(lora_rank),
                "LORA_ALPHA": str(lora_alpha),
                "LR": repr(float(learning_rate)),
                "SFT_STOP": str(sft_stop_after_step),
                "SFT_BATCHES_PER_STEP": str(sft_batches_per_step),
                "SAVE_STEPS": str(save_steps),
                # role_phase.sh keeps only ONE checkpoint by default, which
                # leaves held-out selection nothing to choose between. Retain
                # every checkpoint this oracle writes.
                "MAX_CKPT_NUM": os.environ.get(
                    "MAX_CKPT_NUM",
                    str(max(steps // max(save_steps, 1) + 2, 4)),
                ),
                "ACTOR_LR_SCHEDULER": actor_lr_scheduler,
                "LR_WARMUP_RATIO": repr(float(lr_warmup_ratio)),
                "TRAINING_SEED": str(training_seed),
                "OPPONENT_POOL_JSON": json.dumps(normalized_pool, sort_keys=True),
                # role_lora_v2.py sets both on every cold-PSRO oracle.
                "EXACT_LABEL_BALANCE": "1",
                "LABEL_DRIFT_POLICY": psro.TRAINING_LABEL_DRIFT_POLICY,
                # Score the attacker against its assigned prompt_type instead
                # of the label WildGuard observed. Empty keeps upstream v4.
                "ATTACKER_OBJECTIVE": os.environ.get(
                    "ATTACKER_OBJECTIVE", ""
                ),
                # Same in-training evaluation cadence as the archived runs, so
                # the W&B panels line up with A1/A2/D1/D2.
                "EVAL_STEPS": "10",
                "EVAL_START_STEPS": "50",
                "DIVERSITY_STEPS": "5",
                "WANDB_RUN_NAME": wandb_identity,
                "JUDGE_PORT": str(cfg.train_judge_port),
                "RAY_PORT": str(cfg.ray_port),
                # Keep scripts/slurm/README.md's 18.7s/step layout: no sleep
                # flags, 4 engines at TP=1, prefix caching, CUDA graphs. With
                # the judge colocated the sleep flags cost ~110s/step, because
                # --deepspeed_enable_sleep offloads optimizer state every step.
                "VLLM_SLEEP": os.environ.get("VLLM_SLEEP", "0"),
                "DS_SLEEP": os.environ.get("DS_SLEEP", "0"),
                "EAGER": os.environ.get("EAGER", "0"),
                # That layout leaves the ZeRO-3 actor ~22G of headroom, and a
                # PSRO oracle asked for 22.9G in one allocation. The traceback
                # puts it in make_experience -> actor.forward, i.e. the logprob
                # pass, which is batched by micro_rollout_batch_size. The
                # ladder's 32 is fine against a single frozen adapter, but a
                # trained Nash mixture answers at length and 32 packed
                # sequences reach ~48k tokens, whose fp32 logits are that 22.9G.
                # The archived Modal runs used 8; restoring it caps the pass at
                # ~6G and is pure batching, so the gradients are unchanged.
                "MICRO_ROLLOUT": os.environ.get("MICRO_ROLLOUT", "8"),
                "MICRO_TRAIN": os.environ.get("MICRO_TRAIN", "8"),
                # Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                # here. It pushes the LoRA weight sync onto the CUDA-IPC path,
                # which passes a memory fd with pidfd_getfd; this host runs
                # kernel.yama.ptrace_scope=1, so that call is denied between
                # sibling Ray actors and every engine update fails with
                # "pidfd_getfd: Operation not permitted".
            }
        )
        if role_start_adapter:
            environment["START_ADAPTER"] = role_start_adapter
        environment.pop("CUDA_VISIBLE_DEVICES", None)

        training_log = run_dir / "training.log"
        log(
            f"training {target} ({role}) [{cfg.init_mode} start from "
            f"{role_start_adapter or 'base'}] vs frozen mixture "
            + ", ".join(
                f"{entry['id']}={float(entry['probability']):.4f}"
                for entry in normalized_pool
            )
        )
        log(f"  wandb run name: {wandb_identity}")
        log(f"  log: {training_log}")
        criterion = _build_criterion(role)
        if criterion is None:
            with training_log.open("ab") as handle:
                completed = subprocess.run(
                    ["bash", str(ROLE_PHASE)],
                    cwd=str(cfg.work), env=environment,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
            returncode = completed.returncode
        else:
            returncode, note = _supervise_training(
                environment, training_log, criterion, checkpoint.parent
            )
            log(f"  early stop: {note}")
        if returncode != 0:
            raise RuntimeError(
                f"{target} training failed rc={returncode}; "
                f"see {training_log}"
            )
        # A plateau stop ends training before `steps`, so the last checkpoint
        # is the stopping step and not the budget. Resolve to the highest step
        # actually written; held-out selection then chooses among all of them.
        if not psro._adapter_checkpoint(checkpoint):
            written = sorted(
                (int(match.group(1)), path)
                for path in checkpoint.parent.glob("global_step*_hf")
                for match in [re.fullmatch(r"global_step(\d+)_hf", path.name)]
                if match and psro._adapter_checkpoint(path)
            )
            if written:
                checkpoint = written[-1][1]
                log(f"  stopped early; final checkpoint is {checkpoint.name}")
        if not psro._adapter_checkpoint(checkpoint):
            raise RuntimeError(f"{target} finished without {checkpoint}")

        import hashlib

        run_name = wandb_identity
        psro._write_json_atomic(
            manifest_path,
            {
                "method": "Self-RedTeam role-specific PEFT LoRA v2 (Slurm port)",
                "role": role,
                "ppo_reward_type": "general_sum",
                "strict_zero_sum_ppo_reward": False,
                "label_drift_training_policy": psro.TRAINING_LABEL_DRIFT_POLICY,
                "prompt_label_balance": (
                    "deterministic exact 50/50 harmful/benign finite prefix"
                ),
                "base_model": psro.BASE_MODEL,
                "init_mode": cfg.init_mode,
                "role_start": expected_role_start,
                "role_start_base": psro.BASE_MODEL,
                "fixed_opponent_base": psro.BASE_MODEL,
                "fixed_opponent_adapter": None,
                "fixed_opponent_pool": normalized_pool,
                "fixed_opponent_sampling": (
                    "deterministic request-level categorical draw; one frozen "
                    "opponent action per two-turn episode"
                ),
                "steps": steps,
                "steps_completed": int(
                    re.fullmatch(r"global_step(\d+)_hf", checkpoint.name).group(1)
                ),
                "rollout_batch_size": 128,
                "train_batch_size": 32,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "actor_lr_scheduler": actor_lr_scheduler,
                "lr_warmup_ratio": lr_warmup_ratio,
                "seed": training_seed,
                "kl_penalty": 0.0,
                "reference_kl_monitored": True,
                "sft_stop_after_step": sft_stop_after_step,
                "sft_batches_per_step": sft_batches_per_step,
                "wandb_identity": run_name,
                "wandb_run_id": hashlib.sha1(run_name.encode()).hexdigest()[:8],
                "checkpoint_retention": "all_hf_checkpoints_for_selection",
                "checkpoint": str(checkpoint),
                "executor": "slurm/role_phase.sh",
            },
        )
        selected = _select_checkpoint(cfg, role, checkpoint, normalized_pool, target)
        return psro._adapter_identity(selected)

    return train_cold_psro_oracle


def _base_model_path(cfg) -> str:
    """Resolve the base snapshot the same way role_phase.sh does."""
    override = os.environ.get("BASE_MODEL_PATH")
    if override:
        return override
    hub = cfg.hf_home / "hub" / "models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated" / "snapshots"
    snapshots = sorted(p for p in hub.glob("*") if p.is_dir())
    if not snapshots:
        raise RuntimeError(f"no base-model snapshot under {hub}")
    return str(snapshots[0])


def _select_checkpoint(cfg, role, final_checkpoint, normalized_pool, target):
    """Return the checkpoint that scores best on held-out prompts.

    Falls back to the final checkpoint on any failure: a selection problem must
    never lose an oracle that trained successfully.
    """
    if os.environ.get("CKPT_SELECTION", "1") == "0":
        return final_checkpoint
    ckpt_dir = Path(final_checkpoint).parent
    opponent = "base"
    for entry in sorted(normalized_pool, key=lambda e: -float(e["probability"])):
        if entry.get("path"):
            opponent = entry["path"]
            break
    judge = Judge(cfg)
    try:
        url = judge.start()
        command = [
            cfg.python, str(SELECT_CHECKPOINT),
            "--ckpt-dir", str(ckpt_dir), "--role", role, "--opponent", str(opponent),
            "--work", str(cfg.work), "--base-model", _base_model_path(cfg),
            "--judge-url", url,
            "--n-prompts", os.environ.get("SELECT_N_PROMPTS", "64"),
            "--offset", os.environ.get("SELECT_OFFSET", "6400"),
            "--max-new-tokens", os.environ.get("GENERATE_MAX_LEN", "4096"),
        ]
        environment = cfg.base_env()
        environment["CUDA_VISIBLE_DEVICES"] = "1"
        completed = subprocess.run(command, cwd=str(cfg.work), env=environment)
        if completed.returncode != 0:
            log(f"  {target}: checkpoint selection failed rc={completed.returncode}; "
                "keeping the final checkpoint")
            return final_checkpoint
        payload = json.loads((ckpt_dir.parent / "checkpoint_selection.json").read_text())
        best = Path(payload["best_checkpoint"])
        if not best.is_dir():
            log(f"  {target}: selected {best} is missing; keeping the final checkpoint")
            return final_checkpoint
        log(f"  {target}: selected step {payload['best_step']} "
            f"(final was {payload['final_step']})")
        return best
    except Exception as exc:                      # never lose a trained oracle
        log(f"  {target}: checkpoint selection error ({exc}); keeping the final checkpoint")
        return final_checkpoint
    finally:
        judge.stop()


# --- payoff-cell prefetch ------------------------------------------------------


def initialize_state(psro: Any, cfg: Config, advance_kwargs: dict[str, Any]) -> None:
    """Write a fresh run's contract and A0/D0 population, and nothing else.

    ``advance_cold_psro`` persists the contract and the base population before
    it dispatches an action, but it has no initialize-only mode. Stubbing both
    workers makes the first call stop right after the durable write, so the
    judge and the cell workers are still set up by the normal loop below rather
    than an action running in-process with no judge.
    """

    class _Initialized(Exception):
        pass

    def _stop(**_kwargs: Any) -> None:
        raise _Initialized

    cell, oracle = psro.evaluate_zero_sum_psro_cell, psro.train_cold_psro_oracle
    psro.evaluate_zero_sum_psro_cell = common._bind_local(_stop)
    psro.train_cold_psro_oracle = common._bind_local(_stop)
    try:
        psro.advance_cold_psro(**advance_kwargs)
    except _Initialized:
        pass
    finally:
        psro.evaluate_zero_sum_psro_cell = cell
        psro.train_cold_psro_oracle = oracle
    state_path = cfg.run_root / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"coordinator did not initialize {state_path}")
    log(f"initialized {state_path}")


def _racing_plan(
    psro: Any, cfg: Config, state: dict[str, Any], pending: list[tuple[str, str]]
) -> dict[str, int]:
    """Decide a per-cell episode budget from the pilot estimates.

    Keeps the top-``m`` attacker rows and defender columns by point estimate
    against the current meta-strategy, plus anything already carrying Nash
    weight, and gives the survivors the full budget. Everything else stops at the
    pilot. Pruning is deliberately allowed to be wrong: the offline sweep on the
    completed 4x4 matrix put this at 18.9x less wall clock than the uniform
    4,000-per-cell schedule while still recovering the defender Nash support 62%
    of the time, against 92% for the full schedule.
    """
    import numpy as np
    from roll.utils.role_lora_zero_sum_psro import solve_zero_sum_meta_game

    contract = state["contract"]
    full = int(os.environ.get("RACE_FULL_EPISODES") or contract["matrix_episodes_per_cell"])
    attackers = sorted(
        (k for k, v in state["population"].items() if k.startswith("A")),
        key=lambda k: int(k[1:]),
    )
    defenders = sorted(
        (k for k, v in state["population"].items() if k.startswith("D")),
        key=lambda k: int(k[1:]),
    )
    cells = state["cells"]
    matrix = np.array(
        [
            [cells.get(f"{a}__{d}", {}).get("value", np.nan) for d in defenders]
            for a in attackers
        ],
        dtype=float,
    )
    if np.isnan(matrix).all():
        return {f"{a}__{d}": full for a, d in pending}
    # fill gaps with row/column means so the solver sees a complete matrix
    grand = float(np.nanmean(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                row = np.nanmean(matrix[i]); col = np.nanmean(matrix[:, j])
                cand = [x for x in (row, col) if not np.isnan(x)] or [grand]
                matrix[i, j] = float(np.mean(cand))
    try:
        sol = solve_zero_sum_meta_game(matrix, max_exploitability=1.0)
        p = np.array(sol["attacker_strategy"]); q = np.array(sol["defender_strategy"])
    except Exception as exc:                       # a poor pilot solve is not fatal
        log(f"  racing: meta-solve failed ({exc}); giving every cell the full budget")
        return {f"{a}__{d}": full for a, d in pending}
    row_value = matrix @ q
    col_value = p @ matrix
    m_row = int(os.environ.get("RACE_M_ROW", "2"))
    m_col = int(os.environ.get("RACE_M_COL", "2"))
    tau = float(os.environ.get("RACE_TAU", "0.01"))
    keep_a = {attackers[i] for i in np.argsort(-row_value)[:m_row]} | {
        attackers[i] for i in range(len(attackers)) if p[i] > tau
    }
    keep_d = {defenders[j] for j in np.argsort(col_value)[:m_col]} | {
        defenders[j] for j in range(len(defenders)) if q[j] > tau
    }
    # The newest attacker and defender are the best responses this generation was
    # run to produce, and they have no measured row/column to rank on, so they are
    # never allowed to be pruned on an imputed estimate.
    if attackers:
        keep_a.add(attackers[-1])
    if defenders:
        keep_d.add(defenders[-1])
    pilot = int(os.environ.get("RACE_PILOT_EPISODES", "256"))
    plan: dict[str, int] = {}
    for a, d in pending:
        plan[f"{a}__{d}"] = full if (a in keep_a and d in keep_d) else pilot
    log(f"  racing: keeping attackers {sorted(keep_a)}, defenders {sorted(keep_d)}")
    log(f"  racing: {sum(v == full for v in plan.values())}/{len(plan)} cells at the "
        f"full {full}, the rest capped at the {pilot}-episode pilot")
    return plan


def prefetch_cells(psro: Any, cfg: Config, state: dict[str, Any], judge: Judge) -> None:
    from roll.utils.role_lora_cold_psro import missing_matrix_cells

    pending = missing_matrix_cells(state["population"], state["cells"])
    if not pending:
        return
    contract = state["contract"]
    generation = contract["matrix_generation"]
    race = os.environ.get("RACE_CELLS", "0") == "1"
    plan: dict[str, int] = {}
    if race:
        plan = _racing_plan(psro, cfg, state, list(pending))
        plan_path = cfg.run_root / "cell_episode_plan.json"
        payload = {
            "default": int(os.environ.get("RACE_FULL_EPISODES")
                           or contract["matrix_episodes_per_cell"]),
            "pilot": int(os.environ.get("RACE_PILOT_EPISODES", "256")),
            "m_row": int(os.environ.get("RACE_M_ROW", "2")),
            "m_col": int(os.environ.get("RACE_M_COL", "2")),
            "cells": plan,
            "note": (
                "Per-cell episode budgets chosen by the racing scheduler. A reduced "
                "budget is an exact prefix of the full-budget episode sequence, so a "
                "capped cell is the same estimand stopped early."
            ),
        }
        plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"  racing: wrote {plan_path}")

    def episode_budget(attacker: str, defender: str) -> int:
        return plan.get(f"{attacker}__{defender}", int(contract["matrix_episodes_per_cell"]))
    judge_url = judge.start()
    cfg.cell_log_dir.mkdir(parents=True, exist_ok=True)
    log(f"prefetching {len(pending)} payoff cell(s) on GPUs 1-{cfg.cell_workers}")

    queue = list(pending)
    running: dict[int, tuple[str, str, subprocess.Popen[bytes], Any]] = {}
    failures: list[str] = []
    free_gpus = list(range(1, cfg.cell_workers + 1))

    def launch(gpu: int, attacker: str, defender: str) -> None:
        attacker_record = state["population"][attacker]
        defender_record = state["population"][defender]
        environment = cfg.base_env()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            cfg.python, str(CELL_WORKER),
            "--coordinator", cfg.coordinator,
            "--psro-root", str(cfg.psro_root),
            "--raw-payoff-root", str(cfg.raw_payoff_root),
            "--upstream-source", str(cfg.work),
            "--judge-url", judge_url,
            "--run-suffix", cfg.suffix,
            "--attacker-label", attacker,
            "--defender-label", defender,
            "--attacker-adapter", attacker_record.get("path") or "base",
            "--defender-adapter", defender_record.get("path") or "base",
            "--attacker-sha256", attacker_record["adapter_sha256"],
            "--defender-sha256", defender_record["adapter_sha256"],
            "--episodes", str(episode_budget(attacker, defender)),
            "--seed-base", str(contract["seed_base"]),
            "--max-candidate-multiplier", str(generation["max_candidate_multiplier"]),
            "--candidate-wave-pairs", str(generation["candidate_wave_pairs"]),
            *(
                ["--adaptive-stratum-sampling"]
                if os.environ.get("ADAPTIVE_STRATUM_SAMPLING") == "1"
                else []
            ),
            "--generation-batch-size", str(generation["generation_batch_size"]),
            "--judge-batch-size", str(generation["judge_batch_size"]),
            "--max-new-tokens", str(generation["max_new_tokens"]),
        ]
        cell_log = cfg.cell_log_dir / f"{attacker}__{defender}.log"
        handle = cell_log.open("ab")
        process = subprocess.Popen(
            command, cwd=str(cfg.work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        running[gpu] = (attacker, defender, process, handle)
        log(f"  cell {attacker} vs {defender} -> GPU {gpu}, log {cell_log}")

    while queue or running:
        while queue and free_gpus:
            launch(free_gpus.pop(0), *queue.pop(0))
        time.sleep(20)
        for gpu, (attacker, defender, process, handle) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            del running[gpu]
            free_gpus.append(gpu)
            if process.returncode == 0:
                log(f"  cell {attacker} vs {defender} done")
            else:
                failures.append(f"{attacker}__{defender} rc={process.returncode}")
                log(f"  cell {attacker} vs {defender} FAILED rc={process.returncode}")
    if failures:
        # Not fatal: the coordinator re-runs any cell it still needs inline, on
        # the GPUs the workers just released. Surface it loudly either way.
        log(f"prefetch failures (coordinator will retry inline): {failures}")


# --- main loop -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--sft-dir", required=True, type=Path)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--raw-payoff-root", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cell-workers", type=int, default=3)
    parser.add_argument("--cell-judge-port", type=int, default=5310)
    parser.add_argument("--cell-judge-util", type=float, default=0.45)
    parser.add_argument("--train-judge-port", type=int, default=5311)
    parser.add_argument("--ray-port", type=int, default=6311)
    parser.add_argument("--run-official-evaluation", action="store_true")
    parser.add_argument("--coordinator", default=common.COLD_COORDINATOR)
    parser.add_argument("--init-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument(
        "--fresh-cold-run", action="store_true",
        help="initialize a new cold run instead of requiring a seeded state",
    )
    parser.add_argument("--generations", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(args)
    cfg.apply_to_process()
    state_path = cfg.run_root / "state.json"
    if not state_path.is_file():
        if cfg.init_mode != "hot" and not args.fresh_cold_run:
            raise SystemExit(
                f"no seeded run at {state_path}; run seed_cold_psro_run.py first"
            )
        # Both a hot run and an explicitly requested fresh cold run start at
        # generation one, so there is nothing to seed: the coordinator's own
        # initializer writes A0/D0 and the contract.
        log(f"no state at {state_path}; initializing a fresh {cfg.init_mode} run")

    psro, payoff = common.load_modules(cfg.coordinator)
    common.retarget(
        psro, payoff,
        psro_root=cfg.psro_root,
        raw_payoff_root=cfg.raw_payoff_root,
        upstream_source=cfg.work,
        judge_url="http://127.0.0.1:0/classify",
    )
    judge = Judge(cfg)
    psro.train_cold_psro_oracle = common._bind_local(
        make_local_oracle_trainer(psro, cfg)
    )
    psro._stable_wildguard_rm_url = lambda *a, **k: judge.url
    payoff._stable_wildguard_rm_url = lambda *a, **k: judge.url

    from roll.utils.role_lora_cold_psro import next_action

    if state_path.is_file():
        contract = psro._read_json(state_path)["contract"]
    else:
        contract = None
    advance_kwargs = (
        psro._advance_kwargs(contract, cfg.suffix)
        if contract
        else {"run_suffix": cfg.suffix}
    )
    if not contract:
        # A fresh run writes its contract on the first advance, so the two
        # behaviour changes have to be supplied here to be recorded in it.
        if args.generations:
            advance_kwargs["generations"] = args.generations
        objective = os.environ.get("ATTACKER_OBJECTIVE", "")
        if objective:
            advance_kwargs["training_attacker_objective"] = objective
        if os.environ.get("ADAPTIVE_STRATUM_SAMPLING") == "1":
            advance_kwargs["matrix_adaptive_stratum_sampling"] = True
        # Training budget and checkpoint cadence are contract fields, so they can
        # only be chosen when the contract is first written. Every attacker in
        # both arms was still improving at step 100, and 5 of 8 defenders peaked
        # before it, so the budget is raised and checkpoints are kept densely
        # enough to select on afterwards.
        for key, env in (("steps_per_role", "STEPS_PER_ROLE"),
                         ("save_steps", "SAVE_STEPS_CONTRACT"),
                         ("matrix_episodes", "MATRIX_EPISODES")):
            raw = os.environ.get(env)
            if raw:
                advance_kwargs[key] = int(raw)
    log(f"coordinator: {cfg.coordinator} (init_mode={cfg.init_mode})")
    log(f"run root: {cfg.run_root}")
    if contract:
        log(f"contract generations={contract['generations']} "
            f"steps_per_role={contract['steps_per_role']} "
            f"episodes/cell={contract['matrix_episodes_per_cell']}")

    actions = 0
    try:
        while True:
            if not state_path.is_file():
                initialize_state(psro, cfg, advance_kwargs)
                contract = psro._read_json(state_path)["contract"]
                advance_kwargs = psro._advance_kwargs(contract, cfg.suffix)
                continue
            state = psro._read_json(state_path)
            if state.get("completed") is True:
                log("run already complete")
                break
            action = next_action(
                state["population"], state["cells"], state["evaluations"],
                generations=state["contract"]["generations"],
            )
            log(f"next action: {action}")
            if action["kind"] == "evaluation" and cfg.stop_before_evaluation:
                log(
                    "stopping before the official defender evaluations: they run "
                    "through the selfredteam-official-eval workflow, which has no "
                    "local port yet. A3-A5 and D3-D5 are trained and the 6x6 "
                    "matrix is complete."
                )
                break
            if action["kind"] == "cell":
                # If a worker dies the coordinator evaluates that cell inline,
                # in this process. Keep GPU 0 out of its view so the fallback
                # never lands a 0.90-utilization engine on the judge's card.
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(index) for index in range(1, cfg.cell_workers + 1)
                )
                prefetch_cells(psro, cfg, state, judge)
            else:
                judge.stop()
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = psro.advance_cold_psro(**advance_kwargs)
            actions += 1
            if isinstance(result, dict) and result.get("completed") is True:
                log("PSRO run complete")
                break
            log(f"completed action: {result.get('completed_action', result)}")
            if args.max_actions and actions >= args.max_actions:
                log(f"stopping after {actions} action(s) as requested")
                break
    finally:
        judge.stop()

    final = psro._read_json(state_path)
    log(f"stage={final.get('stage')} completed={final.get('completed')}")
    log(f"population={sorted(final.get('population', {}))}")
    log(f"cells={len(final.get('cells', {}))}/36")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
