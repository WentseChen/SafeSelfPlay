#!/usr/bin/env bash
# Train MAGIC on 4 GPUs with the abliterated defender base. Resumes from the latest checkpoint if one exists.
# Usage: bash docker/run_magic.sh train (Docker), or MAGIC_HOME=... CUDA_VISIBLE_DEVICES=0,1,2,3 bash docker/magic/run.sh
# Layout: 2 GPUs per role (MAGIC's paper: 4), Qwen3Guard judge served on the first GPU with 20% of its memory.
set -eu -o pipefail
M=${MAGIC_HOME:?}; C=${MAGIC_CODE:-$M/code}   # inside the Docker image MAGIC_CODE=/opt/magic
export HF_HOME=${HF_HOME:-$M/hf_cache} HF_HUB_OFFLINE=1 TMPDIR=$M/tmp
# Ray puts its sockets under RAY_TMPDIR; Unix socket paths must stay under 107 bytes, so keep it short.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rw_magic}
mkdir -p $TMPDIR $RAY_TMPDIR $M/logs
[ -d $M/env/bin ] && export PATH=$M/env/bin:$PATH   # venv install; the Docker image uses its system Python
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
first_gpu=${CUDA_VISIBLE_DEVICES%%,*}
snap() { ls -d $HF_HOME/hub/models--$1/snapshots/* | head -1; }
export MODEL_DEFENDER_BASE=$M/llama31_abl_ittok
export MODEL_ATTACKER_SFT=$(snap XiaoyuWen--MAGIC-attacker-sft-Llama3.1-8B-Instruct)
# Concurrent judge calls; the judge decodes at temperature 0, so this changes speed only (public default: 1 worker).
export REWARD_SCORE_MAX_WORKERS=16
# Response cap 6144, as in MAGIC's paper and script.
export MAX_RESP=${MAX_RESP:-6144}
# Up to 600 steps (MAGIC: 300, "early stop in step200"); the abliterated base starts with no refusals, so it gets
# twice the budget and stops early once validation has converged (see early_stop below).
# Smoke tests only: fewer GPUs per role and fewer steps. Real runs keep 2 per role and 600 steps.
GPUS_PER_ROLE=${GPUS_PER_ROLE:-2}; TOTAL_STEPS=${TOTAL_STEPS:-600}
# Convergence rule: from step MIN_STEPS on, stop once val/test_score/game (every 15 steps) has not beaten its best
# by MIN_DELTA for PATIENCE validations in a row. Training stops at the next complete checkpoint.
MIN_STEPS=${MIN_STEPS:-300}; PATIENCE=${PATIENCE:-6}; MIN_DELTA=${MIN_DELTA:-0.01}
if [ -f $M/logs/CONVERGED ]; then
  [ "${FORCE:-0}" = 1 ] || { echo "=== already stopped on convergence: $(cat $M/logs/CONVERGED). Set FORCE=1 to train further."; exit 0; }
  mv $M/logs/CONVERGED $M/logs/CONVERGED.prev
fi
export WANDB_MODE=offline RAY_MASTER_PORT=${RAY_MASTER_PORT:-6391} CHECKPOINT_DIR=$M/ckpt
# MAGIC's reward code only parses Qwen3Guard output ("Safety:", "Refusal:") from a model served as "orm".
JPORT=${JPORT:-18471}
export WILDGUARD_API_ENDPOINT=http://127.0.0.1:$JPORT/v1 WILDGUARD_API_KEY=EMPTY OPENAI_API_KEY=EMPTY  # local judge, no auth
echo "=== start $(date -Is) gpus=$CUDA_VISIBLE_DEVICES max_resp=$MAX_RESP"

if ! curl -sf $WILDGUARD_API_ENDPOINT/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=$first_gpu nohup vllm serve "$(snap Qwen--Qwen3Guard-Gen-8B)" --served-model-name orm --port $JPORT \
    --gpu-memory-utilization 0.2 --max-model-len 16384 --enforce-eager > $M/logs/judge.log 2>&1 &
  for i in $(seq 1 90); do curl -sf $WILDGUARD_API_ENDPOINT/models >/dev/null && break; sleep 10; done
  curl -sf $WILDGUARD_API_ENDPOINT/models >/dev/null || { echo JUDGE_FAILED; tail -20 $M/logs/judge.log; exit 1; }
fi
echo "=== judge up $(date -Is)"

# MAGIC's public script with only these changes: 4 GPUs (2 per role), resumable, our run name, checkpoint every 25 steps,
# console logging, no per-step dump of training generations (logging only, ~95 s/step), and the response cap.
sed -e "s/--num-gpus 8/--num-gpus $((2*GPUS_PER_ROLE))/" \
    -e "s/trainer.n_gpus_per_node=4/trainer.n_gpus_per_node=$GPUS_PER_ROLE/" \
    -e "s/trainer.total_training_steps=300/trainer.total_training_steps=$TOTAL_STEPS/" \
    -e 's/actor_rollout_ref.actor.optim.lr=1e-6/actor_rollout_ref.actor.optim.lr=1e-6 \\\n  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM:-0.5}/' \
    -e 's/trainer.resume_mode=disable/trainer.resume_mode=auto/' \
    -e 's/experiment_name="D-q257bi-A-q257bi-${timestamp}"/experiment_name="D-llama31abl-A-magicsft"/' \
    -e 's/trainer.save_freq=15/trainer.save_freq=25/' \
    -e 's/trainer.logger=\[console,wandb\]/trainer.logger=[console] "$@"/' \
    -e 's/+trainer.save_train_generations=True/+trainer.save_train_generations=False/' \
    -e 's/data.max_response_length=6144/data.max_response_length=${MAX_RESP}/' \
    $C/scripts/rl/separated/grpo_public.sh > $C/scripts/rl/separated/grpo_abl4.sh
diff $C/scripts/rl/separated/grpo_public.sh $C/scripts/rl/separated/grpo_abl4.sh || true
# Each checkpoint holds both models with optimizer state (~126 GB). Keep the newest two whole; every 100th step
# keeps only the defender weights (~32 GB); the rest are deleted.
( while sleep 300; do for r in $CHECKPOINT_DIR/Game-separated/*/; do
    for d in $(ls -d ${r}global_step_* 2>/dev/null | sort -t_ -k3 -n | head -n -2); do
      if [ $(( ${d##*_} % 100 )) -eq 0 ]; then rm -rf $d/attacker $d/defender/actor/optim_* $d/defender/actor/extra_state_*
      else rm -rf $d; fi
    done
  done; done ) &

ckpt_done() { # newest checkpoint whose both models have every shard written and that has not changed for 3 minutes
  local d; d=$(ls -d $CHECKPOINT_DIR/Game-separated/*/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
  [ -n "$d" ] || return 1
  [ "$(ls $d/defender/actor/ $d/attacker/actor/ 2>/dev/null | grep -c '_world_size_.*\.pt$')" -eq $((6*GPUS_PER_ROLE)) ] || return 1
  [ -z "$(find $d -mmin -3 -print -quit)" ] || return 1
  echo ${d##*_}
}
early_stop() {
  while sleep 120; do
    if [ ! -f $M/logs/CONVERGED ]; then
      python3 - $M/logs/train.log $MIN_STEPS $PATIENCE $MIN_DELTA > $M/logs/early_stop.txt <<'PY' || continue
import re, sys
log, min_steps, patience, delta = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
vals = {}
for line in open(log, errors="ignore"):
    s, v = re.search(r"step:(\d+) - ", line), re.search(r"val/test_score/game:(-?[\d.]+)", line)
    if s and v:
        vals[int(s.group(1))] = float(v.group(1))
best, since = None, 0
for step in sorted(vals):
    if best is None or vals[step] > best + delta:
        best, best_step, since = vals[step], step, 0
    else:
        since += 1
last = max(vals) if vals else 0
print(f"validations={len(vals)} last_step={last} best={best} at step {best_step if vals else '-'} no_gain_for={since}")
if last >= min_steps and since >= patience:
    print(f"CONVERGED at step {last}: best {best:.3f} at step {best_step}, no gain >{delta} for {since} validations")
    sys.exit(0)
sys.exit(1)
PY
      tail -1 $M/logs/early_stop.txt > $M/logs/CONVERGED; conv=$(grep -o 'at step [0-9]*' $M/logs/CONVERGED | head -1 | awk '{print $3}')
      echo "=== $(cat $M/logs/CONVERGED); stopping at the next complete checkpoint" | tee -a $M/logs/train.log
    fi
    k=$(ckpt_done) || continue
    if [ "$k" -ge "${conv:-$(grep -o 'at step [0-9]*' $M/logs/CONVERGED | head -1 | awk '{print $3}')}" ]; then
      echo "=== checkpoint global_step_$k complete; stopping training $(date -Is)" | tee -a $M/logs/train.log
      pkill -f grpo_abl4.sh || true; ray stop --force >/dev/null 2>&1 || true; return
    fi
  done
}
early_stop &
cd $C && bash scripts/rl/separated/grpo_abl4.sh "$@" 2>&1 | tee -a $M/logs/train.log
echo "=== train exit ${PIPESTATUS[0]} $(date -Is)"
