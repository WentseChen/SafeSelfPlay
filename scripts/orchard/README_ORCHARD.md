# Running the Self-RedTeam jailbreak baseline on ORCHARD (4×H100)

Everything here was established by getting this to actually run on ORCHARD. Numbers are
measured on the cluster, not estimated. If you only read one thing: **use 4 GPUs with the
judge-sleep patch** (§3), and expect the queue — not the GPUs — to be your bottleneck (§8).

---

## ► START HERE: `run_chunk.sh` is the main entry point

**This is the script you run.** It is the sbatch entry for the jailbreak baseline — it starts
the WildGuard judge, then `openrlhf.cli.train_ppo_ray`, on whatever GPUs Slurm gives it.

```bash
# one chunk of training on the 4-GPU baseline
sbatch --gres=gpu:4 -t 01:00:00 scripts/orchard/run_chunk.sh

# or let the supervisor keep the queue full and chain chunks together (recommended)
setsid nohup bash scripts/orchard/chain_monitor.sh >/dev/null 2>&1 &
```

Do the one-time setup in §7 first (upstream clone, `unzip data.zip`, HF token, env build,
judge-sleep patch, and repoint the paths at the top of `run_chunk.sh`).

One chunk = one Slurm job = a slice of training. Chunks **resume from each other** through
`$CHAIN_DIR/ckpt_latest`, so total progress accumulates across jobs and across preemptions;
cumulative steps live in `$CHAIN_DIR/steps_done`. Everything else in this document explains
why `run_chunk.sh` is written the way it is.

⚠ A queued job runs the script **as it exists when the job starts**, not as it was at submit
time. Editing `run_chunk.sh` therefore changes every pending chunk — handy for pushing a fix
to the whole queue, easy to trip over otherwise.

---

## 1. What this task is

The jailbreak baseline is **§1 "Reproduce Self-RedTeam"** of the SafeSelfPlay README. It is
**OpenRLHF**, not ROLL — a different code path, different pins, and a different env from the
ROLL side of the repo (`roll/`, `examples/start_agentic_pipeline.py`).

- upstream: `mickelliu/selfplay-redteaming` @ `0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123`
  (cloned by `./setup_selfredteam_upstream.sh` as a *sibling* dir)
- entry point: `python -m openrlhf.cli.train_ppo_ray`
- judge: `python -m openrlhf.cli.serve_wildguard` (`allenai/wildguard`, 7B)
- policy: **one** Llama-3.1-8B model plays *both* roles — `language_game.py:55` sets
  `player_role = "attacker" if turn % 2 == 0 else "defender"`. It is not two models.
  (Separate attacker/defender LoRAs are §2 of the README, a different experiment.)

## 2. Prerequisites that will bite you

| thing | detail |
|---|---|
| `allenai/wildguard` is **gated** | Request access on HF, then put a token in `/project/flame/$USER/.hf_token` (chmod 600). Without it the judge 403s at load and every reward call returns HTTP 500. |
| datasets ship **zipped** | `red_team/data/` contains only `data.zip`. `cd red_team/data && unzip -o data.zip` or training dies with `FileNotFoundError: vanilla_harmful_dataset.jsonl`. |
| `$HOME` has a **25 GB quota** | Export `HOME`, `HF_HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `MODELSCOPE_CACHE` to node-local `/tmp` or jobs die with `Disk quota exceeded` in odd places (modelscope, triton, deepspeed autotune). |
| `/project/flame` is **often 100% full** | It swung 93 GB → 0 → 32 GB in one day. Never assume you can write there; check before, and fail loudly rather than half-writing a checkpoint. |
| login is via **gcloud IAP tunnel** | Add `ControlMaster auto` + `ControlPersist 8h` to your `~/.ssh/config` orchard entry. Without multiplexing every `ssh` builds a new tunnel and re-checks credentials — that turns "log in once a day" into "log in hourly". |

## 3. The 4-GPU layout (recommended)

All four GPUs run actor + reference + colocated vLLM. The WildGuard judge shares GPU 3 but
**sleeps during the optimizer step**, which is what makes 4 GPUs work at all.

The judge is idle-but-resident for most of a step (measured: **5.8% mean SM, 94% idle,
holding 90% of its HBM**). Parking it frees ~20 GB exactly when ZeRO-3 needs it.

Measured, 8B model, `rollout_batch_size 128`, seq 2048/2048:

| layout | sec/step | training SM | training HBM | **GPU-s/step** |
|---|---|---|---|---|
| 4 GPU + `--adam_offload` (no judge-sleep) | 103 s | 22–28% | 45% | 412 |
| 5 GPU, dedicated judge GPU | 38 s | 67–78% | 87–96% | 190 |
| **4 GPU + judge-sleep** | **31 s** | 34–47% | 84% | **124** |

`--adam_offload` makes 4 GPUs *fit* but pushes the optimizer step onto the CPU, so the
training phase becomes CPU/PCIe-bound at ~25% GPU utilization — it is the worst option.

### The judge-sleep patch (2 files, both upstream)

`openrlhf/cli/serve_wildguard.py`
- `LLM(..., enable_sleep_mode=True)`
- `sleep(level)` / `wake()` methods on the Ray worker actor
- `POST /sleep` and `POST /wake` routes

`openrlhf/trainer/ppo_utils/experience_maker.py`
- a `judge_power(url, action)` helper (POSTs to `<base>/sleep|/wake`, try/except so reward
  servers without the routes are unaffected)
- **wake** at the top of `make_experience_list`, **sleep** just before `return experiences`
- gated by `JUDGE_SLEEP=1`

Three things this patch has to get right, all learned the hard way:

1. **Rank-0 only, with barriers.** Every DeepSpeed rank runs `make_experience_list`. Ungated,
   one rank parks the judge while the others are still scoring → `Read timed out (180s)`.
   Mirror what `batch_vllm_engine_call` does (`rank_0_only=True`).
2. **The judge is called *inline during rollout*.** Rewards are scored per turn inside the
   multi-turn game, not after generation. Sleeping it before generation starves the rollout.
   It may only sleep for the optimizer step.
3. **`JUDGE_SLEEP=1` must be exported *before* `ray start`.** Ray actors inherit env from the
   raylet. Set it in a reused Ray head's environment and the hooks silently never fire —
   you'll see `/classify` traffic in the judge log but zero `/wake`.

### Judge settings that work

```
--num_gpus 1 --tensor_parallel_size 1 --enforce_eager \
--max_len 4096 --max_num_seqs 16 --batch_size 16 \
--gpu_memory_utilization 0.28
```

- **Do not use TP>1 for the judge.** It starts fine, then dies the moment the trainer runs its
  NCCL weight sync (`--vllm_sync_backend nccl`): `Worker exit type: SYSTEM_ERROR ... code 2`,
  killed at C level with no Python traceback. Two NCCL groups over the same GPUs.
- `0.28` is the floor for TP=1: weights are ~14 GB of the budget. At `0.2` it OOMs under load;
  at `0.11` with TP=4 it fails the KV-cache check at startup.
- The length/concurrency caps matter — without them vLLM reserves for a 32k context.

## 4. Config constraints (check these before inventing numbers)

From `openrlhf/utils/deepspeed/deepspeed.py:94`:

```
accumulated_gradient = train_batch_size * ring_attn_size // micro_train_batch_size // world_size
```

With `train_batch_size=32`, 4 GPUs: `micro_train` ∈ {1,2,4,8}. **`micro_train=16` gives 0 and
divides by zero** at line 285. 8 is the ceiling (one backward, no accumulation).

From `train_ppo_ray.py:38`: `rollout_batch_size` must be divisible by the actor world size —
so 128 works on 1/2/4/8 GPUs but **not 3**. (That kills the otherwise-appealing "3 trainer GPUs
+ 1 dedicated judge GPU" layout.)

## 5. Environment build

Python **3.10** (not 3.11: `vllm==0.8.2` pins `xgrammar==0.1.16`, which has no cp311 wheel).

```bash
export HOME=/tmp/${USER}_home CONDA_PKGS_DIRS=/tmp/conda_pkgs   # conda's cache can't live on a full FS
conda create -y -p /project/flame/$USER/envs/srt python=3.10
pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir vllm==0.8.4        # NOT 0.8.2: xgrammar 0.1.16 was yanked from PyPI
pip install --no-cache-dir -r requirements.txt   # minus torch/flash-attn lines
pip install --no-cache-dir -e <upstream> --no-deps
pip install --no-cache-dir "click==8.1.8"     # click >=8.2 breaks the ray CLI ("not a valid Sentinel")
# prebuilt flash-attn wheel for torch2.6/cu12/cp310 from the Dao-AILab GitHub release
```

Verified working: torch 2.6.0+cu124, vllm 0.8.4, transformers 4.50.0, deepspeed 0.16.5,
ray 2.44.0, flash-attn 2.7.4.post1.

## 6. Things that do *not* work on this cluster

| attempt | what happens |
|---|---|
| `--adam_offload` | `CUDAMismatchException: Installed CUDA 13.0 does not match torch cu124` — DeepSpeedCPUAdam JIT-builds. `DS_SKIP_CUDA_CHECK=1` gets past it, but see §3: it halves throughput. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `AssertionError: Expandable segments are not compatible with memory pool` — vLLM's `CuMemAllocator` (i.e. anything using sleep mode). Ironic, since OOM messages suggest it. |
| ROLL + colocated vLLM (the frozen-lake demo) | `RuntimeError: pidfd_getfd: Operation not permitted`. `kernel.yama.ptrace_scope=1` on these nodes blocks CUDA-IPC weight sync between sibling Ray workers. Fix without root: put trainer / rollout / reference on **separate** GPUs so ROLL uses its NCCL broadcast path instead. |
| `--dependency=singleton` for chained jobs | Only one job is ever *eligible*; the rest show `(Dependency)` and can't backfill. Serialize the checkpoint write with `flock` instead. |

## 7. Launch

### One-time setup

```bash
git clone <SafeSelfPlay> && cd SafeSelfPlay
./setup_selfredteam_upstream.sh                 # clones upstream as a SIBLING dir
cd ../selfplay-redteaming/red_team/data && unzip -o data.zip && cd -
printf '%s\n' "$HF_TOKEN" > /project/flame/$USER/.hf_token && chmod 600 /project/flame/$USER/.hf_token
bash scripts/orchard/install_selfredteam_env.sh  # py3.10 + torch2.6 + vllm0.8.4 (~30 min)
```

Then apply the judge-sleep patch (§3) to the two upstream files, and edit the paths at the
top of `run_chunk.sh` (`CONDA_ROOT`, `ENV_PATH`, `UPSTREAM`, `CHAIN_DIR`) to your user.

### Run one chunk

```bash
sbatch --gres=gpu:4 -t 01:00:00 scripts/orchard/run_chunk.sh
tail -f /home/$USER/logs/chain_<jobid>.out
```

A chunk cold-starts from the base model if `$CHAIN_DIR/ckpt_latest` is absent, otherwise
resumes from it, and appends its step count to `$CHAIN_DIR/steps_done`.

### Run the supervised chain (recommended on `preempt`)

```bash
setsid nohup bash scripts/orchard/chain_monitor.sh >/dev/null 2>&1 &   # keeps ~8 chunks queued
tail -f /home/$USER/logs/chain_monitor.log                             # cadence: 5 min
cat /project/flame/$USER/srt_chain/steps_done                          # cumulative steps
touch /home/$USER/STOP_CHAIN                                           # stop submitting
```

Optionally add a cron watchdog so the supervisor survives a login-node reboot:

```
*/10 * * * * [ -f /home/$USER/STOP_CHAIN ] || pgrep -f "bash /home/$USER/chain_monitor.sh" >/dev/null || setsid nohup bash /home/$USER/chain_monitor.sh >/dev/null 2>&1
```

Note `chain_monitor.sh` submits a **mix** of shapes (4 GPU / 8 GPU, 45 min–2 h) because more
distinct shapes means more backfill opportunities. 4 GPUs is the baseline; an 8-GPU allocation
is simply used if one becomes free.

`run_chunk.sh` detects the GPU count, puts the judge on the last GPU, picks a legal
`micro_train`, resumes from the chain checkpoint if one exists, and mirrors the newest
checkpoint back to shared storage. Key args (upstream §1 values unless noted):

```
--colocate_all_models --vllm_gpu_memory_utilization 0.55
--zero_stage 3 --bf16 --gradient_checkpointing --packing_samples --normalize_reward
--advantage_estimator reinforce --actor_learning_rate 5e-7 --init_kl_coef 0.01
--custom_configs '{"max_turns":2,"reward_type":"general_sum","remove_ties":true}'
--vllm_sync_backend nccl --enforce_eager --vllm_enable_sleep --deepspeed_enable_sleep
--ref_reward_offload --flash_attn
--micro_train_batch_size 4 --micro_rollout_batch_size 16
--train_batch_size 32 --rollout_batch_size 128 --prompt_max_len 2048 --generate_max_len 2048
```

Steps are set by data volume: `steps = max_samples / rollout_batch_size`. There is no
step-limit flag other than `--stop_at_step_200`.

## 8. Surviving the queue (the actual hard part)

`preempt` is **`PriorityTier=1`** — every other partition outranks it regardless of age or
fairshare, and any `flame-*` QoS job can preempt you (`PreemptMode=REQUEUE`, 2 min grace).
`sprio` weights: Partition **10000**, FairShare/Age/QOS/TRES 1000 each, JobSize 500,
`PriorityFavorSmall=yes`.

What this means in practice — and it dominated ~34 hours of wall-clock for us:

- **Short walltimes backfill dramatically better.** A 3 h request sat ~24 h; the same job at
  1 h landed in ~35 min.
- **Small jobs win twice** (`PriorityFavorSmall` + easier backfill), so 4 GPUs beats 8.
- **Assume preemption.** A requeue restarts from step 0 — DeepSpeed checkpoints (~100 GB for
  ZeRO-3) only fit on node-local SSD, which doesn't survive a node change.
- Therefore: **chunk + chain**. Short jobs, `--save_steps 3`, `--max_ckpt_num 1`, mirror the
  newest HF checkpoint to shared storage every ~2 min, and have the next chunk resume from it
  with `--pretrain <chain>/ckpt_latest`. Budget ~12 min of startup per chunk, so a chunk needs
  ~15 uninterrupted minutes to bank anything.
- **Get a `flame-*` QoS if you can.** It is the only structural fix: tier 5 instead of tier 1,
  and preemption-proof rather than preemption-prey. It's a separate Google Form ("FLAME
  High-Priority Compute Request"), needs a FLAME PI sponsor, and requires preliminary results,
  model/parameter counts, data volume, and a compute justification.

## 9. What "it's learning" looks like (and doesn't)

Early reward gains are **format**, not safety. Over the first ~60 steps we saw:

- `attacker/cot_format_violation` 0.491 → **0.000**, `defender/` 0.875 → **0.032**
- rewards rise in lockstep (attacker −0.24 → 0.85, defender −0.13 → 1.6)
- but `vanilla_harmful_harmlessness_rate` stayed flat at ~0.47–0.60 and
  `vanilla_prompts_correct_refusal_acc` ~0.13–0.33, on only 5–7 harmful prompts per step

Both agents' rewards rising together is expected in a general-sum game with a shared format
term — it is *not* evidence the adversarial dynamic is working. The real number is the
held-out evaluation, which is a **separate job** (README §3: `modal_selfredteam_official_eval.py`
+ `run_selfredteam_eval.sh`), not the in-training `--eval_steps` path. Budget for porting it
off Modal the same way you port training.

## 10. Files

| file | role |
|---|---|
| `run_chunk.sh` | one chunk: judge + trainer, resume-and-mirror, GPU-count aware |
| `chain_monitor.sh` | supervisor: keeps N chunks queued across 4 shapes, tracks cumulative steps, stall detection, kill switch via `~/STOP_CHAIN` |
| `start_judge.sh` | standalone judge, for debugging inside a held allocation |
| `debug_run.sh` / `analyze_run.py` / `sample_gpu.sh` | 5-step config sweeps with per-phase SM/HBM profiling |

**Debug inside a held allocation.** Submit a holder (`--gres=gpu:4 -t 03:00:00 --wrap "sleep
infinity"`), start the judge once, and iterate with `srun --jobid=<id> --overlap`. A failed
config then costs seconds instead of another multi-hour queue wait — the single biggest
workflow lesson from this port.

⚠ `pkill -f <pattern>` inside an `srun` step matches the step's own command line and kills it.
Bit us three times. Filter by something the caller doesn't contain, or match on `/proc`.
