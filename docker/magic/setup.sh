#!/usr/bin/env bash
# One-time setup for MAGIC on the abliterated Llama-3.1-8B: code, Python env, models, and the defender base directory.
# No GPU needed. Usage: MAGIC_HOME=/path/with/200GB bash docker/magic/setup.sh
set -eu -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=${MAGIC_HOME:?set MAGIC_HOME to a directory with about 200 GB free}
mkdir -p $M && cd $M
export HF_HOME=${HF_HOME:-$M/hf_cache} HF_HUB_DISABLE_XET=1 TMPDIR=$M/tmp PIP_CACHE_DIR=$M/pipcache
mkdir -p $TMPDIR

# 1. MAGIC code: the branch its README points to, pinned to the commit we ran, plus one upstream bug fix.
if [ ! -d code ]; then
  git clone -q -b 20260922-qwen2.5-vllm0.8.5 https://github.com/BattleWen/MAGIC.git code
  git -C code checkout -q 85a96985a4e7cb01fec8084b6a6bd9df3762362d
  git -C code apply $HERE/seed_prompt.patch
fi
git -C code log -1 --format='MAGIC code %H'

# 2. Python 3.10 env with MAGIC's pinned versions.
[ -d env ] || python3.10 -m venv env
P=$M/env/bin/python
$P -m pip -q install -U pip wheel setuptools
$P -m pip -q install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
( cd code/src/360-LLaMA-Factory && $P -m pip -q install -e . )
( cd code/src/verl && $P -m pip -q install -e . )
( cd code && $P -m pip -q install -r requirements.txt )
# Building flash-attn from source fails on many machines; use the official prebuilt wheel for torch 2.6 / cu12 / py310.
$P -m pip -q install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
$P -c "import torch, vllm, transformers, flash_attn; print('ENV', torch.__version__, vllm.__version__, transformers.__version__, flash_attn.__version__)"
# Expect: ENV 2.6.0+cu124 0.8.5.post1 4.51.3 2.7.4.post1

# 3. Models (all public): defender weights, MAGIC's attacker SFT, and the Qwen3Guard judge that MAGIC's reward code parses.
$P - <<'PY'
from huggingface_hub import snapshot_download as s
for repo in ["mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
             "XiaoyuWen/MAGIC-attacker-sft-Llama3.1-8B-Instruct",
             "Qwen/Qwen3Guard-Gen-8B"]:
    print("MODEL", repo, s(repo, ignore_patterns=["optimizer*", "*.pth"]), flush=True)
PY

# 4. Defender base: abliterated weights with the official Instruct chat template and stop tokens.
$P $HERE/make_base.py --out $M/llama31_abl_ittok
echo SETUP_DONE
