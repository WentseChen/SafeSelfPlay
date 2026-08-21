#!/bin/bash
# Frozen-lake agentic demo (ROLL) on an ORCHARD node. 1 GPU, Qwen2.5-0.5B-Instruct.
set -eu -o pipefail

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=${ENV_PATH:-/project/flame/wentsec/envs/ssp}
REPO=/project/flame/wentsec/SafeSelfPlay

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $ENV_PATH

export PYTHONPATH=$REPO:${PYTHONPATH:-}
# $HOME is at its 25G quota on ORCHARD -> keep every SDK cache on node-local SSD
export HOME=${SSP_HOME:-/tmp/ssp_home}
export MODELSCOPE_CACHE=$HOME/modelscope
mkdir -p $HOME
# ROLL defaults to ModelScope, which is ~2 MB/s from here
export MODEL_DOWNLOAD_TYPE=${MODEL_DOWNLOAD_TYPE:-HUGGINGFACE_HUB}
export HF_HOME=${HF_HOME:-/tmp/ssp_hf}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/ssp_triton_$$}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ssp_ray_$$}
export TMPDIR=${TMPDIR:-/tmp/ssp_tmp_$$}
# ORCHARD nodes block pidfd_getfd, so vLLM's v1 EngineCore subprocess cannot
# import CUDA IPC handles -> keep the engine in-process
# ptrace_scope=1 on ORCHARD blocks pidfd_getfd, which torch needs to share
# expandable-segment storages across ray workers -> use legacy cudaIpc handles
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p $HF_HOME $TRITON_CACHE_DIR $RAY_TMPDIR $TMPDIR

# clear any ray head left over from a previous attempt on this node
ray stop --force >/dev/null 2>&1 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO
python examples/start_agentic_pipeline.py \
  --config_path agentic_demo \
  --config_name ${CONFIG_NAME:-agent_val_frozen_lake_orchard_demo}
