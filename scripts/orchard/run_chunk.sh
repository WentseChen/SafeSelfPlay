#!/bin/bash
#SBATCH --job-name=srtchain
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=1200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wentsec/logs/chain_%j.out
#SBATCH --error=/home/wentsec/logs/chain_%j.err
#
# One CHUNK of Self-RedTeam training. Short walltime backfills far better on the
# PriorityTier=1 preempt partition, and chunks chain through the HF checkpoint in
# CHAIN_DIR, so total training steps accumulate across jobs.
#
# Chunks may run concurrently (more shapes eligible = more backfill chances). Two
# chunks starting from the same checkpoint is wasteful but harmless; the mirror is
# flock-serialised so the shared checkpoint can never be half-written by two writers.
# Works at 4 or 8 GPUs -- whatever the allocation gives us; rollout_batch_size 128
# stays divisible by the world size either way.
set -eu -o pipefail

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=/project/flame/wentsec/envs/srt
UPSTREAM=/project/flame/wentsec/selfplay-redteaming
CHAIN_DIR=/project/flame/wentsec/srt_chain
BASE_MODEL=mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
OUT=/tmp/srt_chain/$SLURM_JOB_ID

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH
export HOME=/tmp/srt_home HF_HOME=/tmp/srt_hf TMPDIR=/tmp/srt_tmp RAY_TMPDIR=/tmp/srt_ray
export TOKENIZERS_PARALLELISM=false WANDB_MODE=${WANDB_MODE:-disabled}
export HF_TOKEN=$(cat /project/flame/wentsec/.hf_token)
export JUDGE_SLEEP=1
mkdir -p $HOME $HF_HOME $TMPDIR $OUT $CHAIN_DIR

NGPU=$(nvidia-smi -L | wc -l)
JUDGE_DEV=$((NGPU - 1))
STEPS_DONE=$(cat $CHAIN_DIR/steps_done 2>/dev/null || echo 0)

# ---- resume: train from the newest chained checkpoint if there is one ----
if [ -f "$CHAIN_DIR/ckpt_latest/config.json" ]; then
  PRETRAIN=$CHAIN_DIR/ckpt_latest
  echo "RESUMING from $PRETRAIN (cumulative steps so far: $STEPS_DONE)"
else
  PRETRAIN=$BASE_MODEL
  echo "COLD START from $BASE_MODEL"
fi

# micro_train must divide train_batch_size/world_size (deepspeed.py:94)
PER_GPU=$((32 / NGPU))
MICRO_TRAIN=$(( PER_GPU < 4 ? PER_GPU : 4 ))
[ $MICRO_TRAIN -lt 1 ] && MICRO_TRAIN=1
# budget the chunk: ~15 min of startup, ~55 s/step, leave 5 min to flush a checkpoint
BUDGET_MIN=${BUDGET_MIN:-40}
CHUNK_STEPS=$(( BUDGET_MIN * 60 / 55 ))
MAX_SAMPLES=$(( CHUNK_STEPS * 128 ))

IP=$(hostname -I | awk '{print $1}')
REMOTE_RM_URL=http://$IP:5000/classify
cd $UPSTREAM
echo "node=$(hostname) job=$SLURM_JOB_ID gpus=$NGPU micro_train=$MICRO_TRAIN chunk_steps=$CHUNK_STEPS"

CUDA_VISIBLE_DEVICES=$JUDGE_DEV RAY_TMPDIR=/tmp/srt_ray_judge \
  nohup python -m openrlhf.cli.serve_wildguard \
    --model_path allenai/wildguard --port 5000 --host $IP --bf16 \
    --num_gpus 1 --tensor_parallel_size 1 --enforce_eager --seed 42 \
    --max_len 4096 --max_num_seqs 16 --batch_size 16 \
    --gpu_memory_utilization 0.28 > $OUT/judge.log 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID ${SYNC_PID:-} 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 120); do
  curl -s -m 5 -X POST $REMOTE_RM_URL -H 'Content-Type: application/json' \
    -d '{"queries":[{"prompt":"hi","response":"hello","game_idx":0}],"prompts":null}' \
    2>/dev/null | grep -q labels && break
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge died"; tail -30 $OUT/judge.log; exit 1; }
  sleep 10
done
echo "judge serving after ~$((i * 10))s at $(date +%H:%M:%S)"

# ---- mirror newest checkpoint into the chain (rsync --inplace: only ~17G free) ----
mirror () {
  local newest
  newest=$(ls -d $OUT/ckpt/global_step*_hf 2>/dev/null | sort -t_ -k2 -V | tail -1)
  [ -n "$newest" ] && [ -f "$newest/config.json" ] || return 1
  local n=$(basename "$newest" | sed "s/global_step//; s/_hf//")
  mkdir -p $CHAIN_DIR/ckpt_latest
  # one writer at a time; concurrent chunks would otherwise interleave rsync --inplace
  flock -w 900 $CHAIN_DIR/.mirror.lock \
    rsync -a --inplace --delete "$newest/" "$CHAIN_DIR/ckpt_latest/" 2>/dev/null || return 1
  local total=$(( STEPS_DONE + n ))
  local prev=$(cat $CHAIN_DIR/steps_done 2>/dev/null || echo 0)
  [ "$total" -gt "$prev" ] && echo $total > $CHAIN_DIR/steps_done
  echo "[mirror] step $n -> chain total $(cat $CHAIN_DIR/steps_done) at $(date +%H:%M:%S)"
}
( while true; do sleep 120; mirror || true; done ) & SYNC_PID=$!

ray start --head --disable-usage-stats --temp-dir=$RAY_TMPDIR --port=6380 \
  --dashboard-port=8299 --num-gpus=$NGPU
export RAY_ADDRESS=$IP:6380

set +e
python -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 --actor_num_gpus_per_node $NGPU \
  --ref_num_nodes 1 --ref_num_gpus_per_node $NGPU \
  --remote_rm_url $REMOTE_RM_URL \
  --vllm_num_engines $NGPU --vllm_tensor_parallel_size 1 \
  --colocate_all_models --vllm_gpu_memory_utilization 0.55 \
  --pretrain $PRETRAIN \
  --save_path $OUT --ckpt_path $OUT/ckpt \
  --save_steps 3 --save_hf_ckpt --disable_ds_ckpt --max_ckpt_num 1 \
  --micro_train_batch_size $MICRO_TRAIN --train_batch_size 32 \
  --micro_rollout_batch_size 16 --rollout_batch_size 128 \
  --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl" \
  --prompt_data_probs "0.5, 0.5" \
  --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl" \
  --sft_data "red_team/data/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl, red_team/data/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl" \
  --sft_data_probs "0.5, 0.5" --sft_input_key vanilla --sft_output_key completion \
  --sft_steps 1 --sft_batches_per_step 1 \
  --max_samples $MAX_SAMPLES --max_epochs 1 \
  --prompt_max_len 2048 --generate_max_len 2048 \
  --zero_stage 3 --num_episodes 1 --bf16 --seed 8888 --top_p 1.0 \
  --actor_learning_rate 5e-7 --init_kl_coef 0.01 \
  --normalize_reward --packing_samples --gradient_checkpointing \
  --advantage_estimator reinforce \
  --custom_configs '{"max_turns":2,"reward_type":"general_sum","remove_ties":true}' \
  --actor_loss_coef 1.0 --postfill_cot_loss_coef 1.0 \
  --eval_steps 10 --eval_start_steps 50 --diversity_score_steps 5 \
  --vllm_sync_backend nccl --enforce_eager \
  --vllm_enable_sleep --deepspeed_enable_sleep \
  --ref_reward_offload --flash_attn
RC=$?
set -e
echo "train rc=$RC at $(date +%H:%M:%S)"
mirror || echo "[mirror] final mirror failed -- checkpoint stayed on node-local disk"
echo "===== CHUNK DONE (chain total: $(cat $CHAIN_DIR/steps_done 2>/dev/null || echo ?)) ====="
