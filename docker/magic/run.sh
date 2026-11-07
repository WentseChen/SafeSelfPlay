#!/usr/bin/env bash
# Train MAGIC on 4 GPUs with the abliterated defender base. Resumes from the latest checkpoint if one exists.
# Usage: MAGIC_HOME=... CUDA_VISIBLE_DEVICES=0,1,2,3 bash docker/magic/run.sh
# Layout: 2 GPUs per role (MAGIC's paper: 4), Qwen3Guard judge served on the first GPU with 20% of its memory.
set -eu -o pipefail
M=${MAGIC_HOME:?}; C=$M/code
export HF_HOME=${HF_HOME:-$M/hf_cache} HF_HUB_OFFLINE=1 TMPDIR=$M/tmp
# Ray puts its sockets under RAY_TMPDIR; Unix socket paths must stay under 107 bytes, so keep it short.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rw_magic}
mkdir -p $TMPDIR $RAY_TMPDIR $M/logs
export PATH=$M/env/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
first_gpu=${CUDA_VISIBLE_DEVICES%%,*}
snap() { ls -d $HF_HOME/hub/models--$1/snapshots/* | head -1; }
export MODEL_DEFENDER_BASE=$M/llama31_abl_ittok
export MODEL_ATTACKER_SFT=$(snap XiaoyuWen--MAGIC-attacker-sft-Llama3.1-8B-Instruct)
# Concurrent judge calls; the judge decodes at temperature 0, so this changes speed only (public default: 1 worker).
export REWARD_SCORE_MAX_WORKERS=16
# Response cap 2048 instead of MAGIC's 6144 (agreed for compute; about 0.6% of defender replies exceed it, and
# MAGIC's own mask_unfinished_reward gives truncated replies zero reward).
export MAX_RESP=${MAX_RESP:-2048}
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
sed -e 's/--num-gpus 8/--num-gpus 4/' \
    -e 's/trainer.n_gpus_per_node=4/trainer.n_gpus_per_node=2/' \
    -e 's/trainer.resume_mode=disable/trainer.resume_mode=auto/' \
    -e 's/experiment_name="D-q257bi-A-q257bi-${timestamp}"/experiment_name="D-llama31abl-A-magicsft"/' \
    -e 's/trainer.save_freq=15/trainer.save_freq=25/' \
    -e 's/trainer.logger=\[console,wandb\]/trainer.logger=[console] "$@"/' \
    -e 's/+trainer.save_train_generations=True/+trainer.save_train_generations=False/' \
    -e 's/data.max_response_length=6144/data.max_response_length=${MAX_RESP}/' \
    $C/scripts/rl/separated/grpo_public.sh > $C/scripts/rl/separated/grpo_abl4.sh
diff $C/scripts/rl/separated/grpo_public.sh $C/scripts/rl/separated/grpo_abl4.sh || true
# Each checkpoint holds both models with optimizer state; keep the newest two plus steps 200 and 300.
( while sleep 300; do for r in $CHECKPOINT_DIR/Game-separated/*/; do
    ls -d ${r}global_step_* 2>/dev/null | sort -t_ -k3 -n | head -n -2 | grep -vE 'global_step_(200|300)$' | xargs -r rm -rf
  done; done ) &
cd $C && bash scripts/rl/separated/grpo_abl4.sh "$@" 2>&1 | tee -a $M/logs/train.log
echo "=== train exit ${PIPESTATUS[0]} $(date -Is)"
