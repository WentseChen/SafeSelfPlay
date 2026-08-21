#!/bin/bash
# Env for the upstream Self-RedTeam (OpenRLHF) stack: torch 2.6.0 + vllm 0.8.2.
# Shared-FS install so any node in a later allocation can use it.
set -eu -o pipefail
set -x

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=${ENV_PATH:-/project/flame/wentsec/envs/srt}
UPSTREAM=/project/flame/wentsec/selfplay-redteaming

# python 3.10: vllm 0.8.2 pins xgrammar==0.1.16, which ships no cp311 wheel.
# HOME/pkg caches point at node-local SSD ($HOME is over quota, /project/flame is full).
export HOME=${SSP_HOME:-/tmp/ssp_home}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/conda_pkgs}
mkdir -p $HOME $CONDA_PKGS_DIRS
source $CONDA_ROOT/etc/profile.d/conda.sh
[ -d "$ENV_PATH" ] || conda create -y -p $ENV_PATH python=3.10
conda activate $ENV_PATH
python -V
pip install --no-cache-dir -U pip setuptools wheel

export TMPDIR=${TMPDIR:-/tmp/srt_pip_$$}
export HF_HOME=${HF_HOME:-/tmp/ssp_hf}
mkdir -p $TMPDIR $HF_HOME
df -h /project/flame | tail -1

pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir vllm==${VLLM_VERSION:-0.8.4}   # 0.8.2 unresolvable: it pins xgrammar==0.1.16, yanked from PyPI
# requirements minus torch/flash-attn (flash-attn built separately, torch already pinned)
grep -vE "^(torch==|flash-attn==)" $UPSTREAM/requirements.txt > $TMPDIR/req.txt
pip install --no-cache-dir -r $TMPDIR/req.txt
pip install --no-cache-dir -e $UPSTREAM --no-deps
# prebuilt flash-attn wheel matching torch2.6/cu12/py310 (abiFALSE = pypi torch build)
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir $FA || echo "FLASH_ATTN_WHEEL_FAILED (fallback: run without --flash_attn)"

python -c "
import torch, vllm, transformers, deepspeed, ray, openrlhf
print(\"torch\", torch.__version__, torch.version.cuda)
print(\"vllm\", vllm.__version__); print(\"transformers\", transformers.__version__)
print(\"deepspeed\", deepspeed.__version__); print(\"ray\", ray.__version__); print(\"openrlhf OK\")
try:
    import flash_attn; print(\"flash_attn\", flash_attn.__version__)
except Exception as e: print(\"flash_attn MISSING:\", e)
"
df -h /project/flame | tail -1
echo "===== SRT INSTALL DONE ====="
