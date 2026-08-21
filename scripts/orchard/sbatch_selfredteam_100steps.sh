#!/bin/bash
#SBATCH --job-name=srt100x4
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --mem=1200G
#SBATCH --time=06:00:00
#SBATCH --output=/project/flame/wentsec/logs/srt100_%j.out
#SBATCH --error=/project/flame/wentsec/logs/srt100_%j.err
#
# Self-RedTeam reproduction on ORCHARD, 100 steps.
# All four GPUs run actor + ref + colocated vLLM (ZeRO-3, sleep mode); the WildGuard
# judge shares GPU 3 with a small memory budget. Measured on the 5-GPU version, a
# dedicated judge GPU sat 94% idle while holding 92% of its HBM, so it is not worth
# a whole H100. vLLM's utilization is cut to leave room for the judge's ~16G.
# Checkpoints land on the node-local SSD (/project/flame is full); the step-100 HF
# checkpoint is copied back to $KEEP_DIR at the end.
set -eu -o pipefail

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=/project/flame/wentsec/envs/srt
UPSTREAM=/project/flame/wentsec/selfplay-redteaming
RUN_NAME=srt100_${SLURM_JOB_ID}
OUT=/tmp/srt_runs/$RUN_NAME
KEEP_DIR=/project/flame/wentsec/srt_runs/$RUN_NAME

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

export HOME=/tmp/srt_home                      # $HOME is at its 25G quota
export HF_HOME=/tmp/srt_hf
export TMPDIR=/tmp/srt_tmp
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_TOKEN=$(cat /project/flame/wentsec/.hf_token)
mkdir -p $HOME $HF_HOME $TMPDIR $OUT $KEEP_DIR

IP=$(hostname -I | awk '{print $1}')
REMOTE_RM_URL=http://$IP:5000/classify
cd $UPSTREAM
echo "node=$(hostname) job=$SLURM_JOB_ID out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- WildGuard judge, tensor-parallel across all 4 GPUs ----
# A judge pinned to one GPU needs ~23G there, but vLLM applies ONE utilization
# fraction to every training engine -- so the shared GPU ran out of room for its
# own 15G of weights ("No available memory for the cache blocks"). TP=4 spreads the
# judge to ~4G per GPU instead, so the pressure is symmetric.
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_TMPDIR=/tmp/srt_ray_judge \
  nohup python -m openrlhf.cli.serve_wildguard \
    --model_path allenai/wildguard --port 5000 --host $IP --bf16 \
    --num_gpus 4 --tensor_parallel_size 4 --enforce_eager --seed 42 \
    --max_len 4096 --max_num_seqs 16 --batch_size 16 \
    --gpu_memory_utilization 0.11 > $KEEP_DIR/judge.log 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true' EXIT

for i in $(seq 1 180); do
  curl -s -m 3 -o /dev/null -X POST $REMOTE_RM_URL -H 'Content-Type: application/json' \
    -d '{"queries":[{"prompt":"hi","response":"hello"}],"prompts":null}' && break
  kill -0 $JUDGE_PID 2>/dev/null || { echo "judge died"; tail -40 $KEEP_DIR/judge.log; exit 1; }
  sleep 10
done
echo "judge ready after ~$((i * 10))s"

# preempt partition: mirror each HF checkpoint off the node-local SSD as soon as it
# appears, keeping only the newest (/project/flame has ~40G free, a ckpt is ~16G)
(
  while true; do
    for d in $OUT/ckpt/global_step*_hf; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      if [ ! -d "$KEEP_DIR/$name" ] && [ -f "$d/config.json" ]; then
        rm -rf "$KEEP_DIR"/global_step*_hf
        cp -r "$d" "$KEEP_DIR/" && echo "[sync] kept $name in $KEEP_DIR"
      fi
    done
    sleep 300
  done
) &
SYNC_PID=$!
trap 'kill $JUDGE_PID $SYNC_PID 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true' EXIT

# ---- trainer: its own ray head over GPUs 0-3 only ----
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=/tmp/srt_ray
ray start --head --disable-usage-stats --temp-dir=$RAY_TMPDIR --port=6380 --num-gpus=4
export RAY_ADDRESS=$IP:6380

# Upstream hyper-parameters (red_team_game_reinforce_8b.sh / modal_selfredteam_official_h200.py),
# with micro-batches halved: H100 has 80G vs the H200's 141G.
# 100 steps = max_samples 12800 / rollout_batch_size 128 (num_episodes=1, max_epochs=1);
# upstream has no step-limit flag other than --stop_at_step_200.
python -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 --actor_num_gpus_per_node 4 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 4 \
  --remote_rm_url $REMOTE_RM_URL \
  --vllm_num_engines 4 --vllm_tensor_parallel_size 1 \
  --colocate_all_models --vllm_gpu_memory_utilization 0.55 \
  --pretrain mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
  --save_path $OUT --ckpt_path $OUT/ckpt \
  --save_steps 50 --save_hf_ckpt --disable_ds_ckpt \
  --micro_train_batch_size 4 --train_batch_size 32 \
  --micro_rollout_batch_size 8 --rollout_batch_size 128 \
  --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl" \
  --prompt_data_probs "0.5, 0.5" \
  --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl" \
  --sft_data "red_team/data/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl, red_team/data/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl" \
  --sft_data_probs "0.5, 0.5" --sft_input_key vanilla --sft_output_key completion \
  --sft_steps 1 --sft_batches_per_step 1 \
  --max_samples 12800 --max_epochs 1 \
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
  --flash_attn

echo "=== training finished, copying HF checkpoints back ==="
df -h /project/flame | tail -1
for d in $OUT/ckpt/global_step*_hf; do
  [ -d "$d" ] || continue
  echo "copying $d -> $KEEP_DIR/"
  cp -r "$d" "$KEEP_DIR/" || echo "COPY FAILED (check /project/flame free space)"
done

echo "===== SRT100 DONE ====="
