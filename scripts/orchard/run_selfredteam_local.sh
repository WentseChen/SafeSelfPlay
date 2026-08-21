#!/bin/bash
# Self-RedTeam attacker/defender safety self-play on ORCHARD GPUs -- no Modal.
# Last visible GPU serves the WildGuard judge; the first $TRAIN_GPUS run the
# colocated actor + ref + vLLM engines under ZeRO-3.
#
#   TRAIN_GPUS=4 bash scripts/orchard/run_selfredteam_local.sh    # inside a 5-GPU allocation
#   TRAIN_GPUS=7 bash scripts/orchard/run_selfredteam_local.sh    # inside an 8-GPU allocation
set -eu -o pipefail

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=${ENV_PATH:-/project/flame/wentsec/envs/srt}
UPSTREAM=${UPSTREAM:-/project/flame/wentsec/selfplay-redteaming}
TRAIN_GPUS=${TRAIN_GPUS:-4}
VLLM_UTIL=${VLLM_UTIL:-0.7}   # upstream value; vllm sleeps during training
ADAM_OFFLOAD=${ADAM_OFFLOAD:-0}            # 1 = optimizer states on CPU (fallback if 4 GPUs OOM)
MAX_SAMPLES=${MAX_SAMPLES:-128}            # demo length (steps = max_samples / rollout_batch_size)
ROLLOUT_BS=${ROLLOUT_BS:-32}
MODEL_PATH=${MODEL_PATH:-mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated}
JUDGE_MODEL=${JUDGE_MODEL:-allenai/wildguard}
RUN_NAME=${RUN_NAME:-orchard_srt_demo_${TRAIN_GPUS}gpu}
OUT=${OUT:-/tmp/srt_out/$RUN_NAME}

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

export HOME=${SSP_HOME:-/tmp/srt_home}      # $HOME is at its 25G quota on ORCHARD
export HF_HOME=${HF_HOME:-/tmp/srt_hf}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/srt_ray}
export TMPDIR=${TMPDIR:-/tmp/srt_tmp}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p $HOME $HF_HOME $RAY_TMPDIR $TMPDIR $OUT
[ -f /project/flame/wentsec/.hf_token ] && export HF_TOKEN=$(cat /project/flame/wentsec/.hf_token)

NGPU=$(nvidia-smi -L | wc -l)
echo "visible GPUs: $NGPU ; training on $TRAIN_GPUS ; vllm util $VLLM_UTIL ; adam_offload=$ADAM_OFFLOAD"
# EXTERNAL_JUDGE=1 -> judge already served elsewhere (e.g. the other allocation),
# so every visible GPU can train
EXTERNAL_JUDGE=${EXTERNAL_JUDGE:-0}
[ "$EXTERNAL_JUDGE" = "1" ] || [ "$NGPU" -gt "$TRAIN_GPUS" ] || { echo "need one spare GPU for the judge"; exit 1; }

IP=$(hostname -I | awk '{print $1}')
JUDGE_PORT=${JUDGE_PORT:-5000}
REMOTE_RM_URL=${REMOTE_RM_URL:-http://$IP:$JUDGE_PORT/classify}
cd $UPSTREAM

if [ "$EXTERNAL_JUDGE" = "1" ]; then
  echo "using external judge at $REMOTE_RM_URL"
else
# ---- WildGuard judge on the last GPU ----
JUDGE_DEV=$((NGPU - 1))
echo "starting WildGuard judge on GPU $JUDGE_DEV -> $REMOTE_RM_URL"
# serve_wildguard spins up its OWN ray instance; keep it in a separate temp dir so
# the trainer does not accidentally join the judge's 1-GPU cluster
CUDA_VISIBLE_DEVICES=$JUDGE_DEV RAY_TMPDIR=/tmp/srt_ray_judge \
  nohup python -m openrlhf.cli.serve_wildguard \
  --model_path $JUDGE_MODEL --port $JUDGE_PORT --host $IP --bf16 \
  --num_gpus 1 --tensor_parallel_size 1 --enforce_eager --seed 42 \
  --gpu_memory_utilization 0.9 > $OUT/judge.log 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 180); do
  curl -s -m 3 -X POST $REMOTE_RM_URL -H 'Content-Type: application/json' \
    -d '{"queries":[{"prompt":"hi","response":"hello"}],"prompts":null}' >/dev/null 2>&1 && break
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge died -- see $OUT/judge.log"; tail -30 $OUT/judge.log; exit 1; }
  sleep 10
done
echo "judge up after ~$((i * 10))s"
fi

OFFLOAD_ARGS=""
[ "$ADAM_OFFLOAD" = "1" ] && OFFLOAD_ARGS="--adam_offload"

# ---- trainer ray head: only the training GPUs, explicit address ----
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TRAIN_GPUS - 1)))
TRAIN_RAY_PORT=${TRAIN_RAY_PORT:-6380}
ray start --head --disable-usage-stats --temp-dir=$RAY_TMPDIR \
  --port=$TRAIN_RAY_PORT --num-gpus=$TRAIN_GPUS
export RAY_ADDRESS=$IP:$TRAIN_RAY_PORT

python -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 --actor_num_gpus_per_node $TRAIN_GPUS \
  --ref_num_nodes 1 --ref_num_gpus_per_node $TRAIN_GPUS \
  --remote_rm_url $REMOTE_RM_URL \
  --vllm_num_engines $TRAIN_GPUS --vllm_tensor_parallel_size 1 \
  --colocate_all_models --vllm_gpu_memory_utilization $VLLM_UTIL \
  --pretrain $MODEL_PATH \
  --save_path $OUT --ckpt_path $OUT/ckpt \
  --save_steps 1000 --disable_ds_ckpt \
  --micro_train_batch_size 2 --train_batch_size 32 \
  --micro_rollout_batch_size 4 --rollout_batch_size $ROLLOUT_BS \
  --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl" \
  --prompt_data_probs "0.5, 0.5" \
  --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl" \
  --max_samples $MAX_SAMPLES --max_epochs 1 \
  --prompt_max_len 1024 --generate_max_len 1024 \
  --zero_stage 3 --num_episodes 1 --bf16 --seed 8888 --top_p 1.0 \
  --actor_learning_rate 5e-7 --init_kl_coef 0.01 \
  --normalize_reward --packing_samples --gradient_checkpointing \
  --advantage_estimator reinforce \
  --custom_configs '{"max_turns":2,"reward_type":"general_sum","remove_ties":true}' \
  --actor_loss_coef 1.0 --postfill_cot_loss_coef 1.0 \
  --eval_steps 1000 --eval_start_steps 1000 --diversity_score_steps 1000 \
  --vllm_sync_backend nccl --enforce_eager \
  --vllm_enable_sleep --deepspeed_enable_sleep \
  $OFFLOAD_ARGS \
  --flash_attn
