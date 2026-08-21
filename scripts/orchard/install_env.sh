#!/bin/bash
# Build the SafeSelfPlay/ROLL env on an ORCHARD compute node.
# Usage (inside an allocation):  bash scripts/orchard/install_env.sh
set -eux

CONDA_ROOT=/project/flame/wentsec/anaconda3
ENV_PATH=${ENV_PATH:-/project/flame/wentsec/envs/ssp}
REPO=/project/flame/wentsec/SafeSelfPlay

source $CONDA_ROOT/etc/profile.d/conda.sh
[ -d "$ENV_PATH" ] || conda create -y -p $ENV_PATH python=3.10
conda activate $ENV_PATH

export TMPDIR=${TMPDIR:-/tmp/ssp_pip_$$}
mkdir -p $TMPDIR
df -h /project/flame /tmp | tail -3

pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

cd $REPO
pip install --no-cache-dir -r requirements_common.txt
pip install --no-cache-dir deepspeed==0.16.4
pip install --no-cache-dir vllm==0.10.2
pip install --no-cache-dir -e $REPO || true

# ORCHARD pins found by debugging the frozen-lake demo:
#   transformers 5.x breaks vLLM 0.10.2's tokenizer path (Qwen2Tokenizer.all_special_tokens_extended)
#   click >=8.2 breaks the ray CLI (Sentinel deepcopy), numpy 2 breaks trl/ROLL,
#   setuptools >=81 drops pkg_resources (gym/sokoban envs), tensorboard is imported by the tracker
pip install --no-cache-dir "transformers==4.56.2" "click==8.1.8" "numpy==1.26.4" "setuptools<81" tensorboard

python -c "
import torch, deepspeed, vllm, ray, roll
print(\"torch\", torch.__version__); print(\"deepspeed\", deepspeed.__version__)
print(\"vllm\", vllm.__version__); print(\"ray\", ray.__version__); print(\"roll OK\")
"
df -h /project/flame | tail -1
echo "===== INSTALL DONE ====="
