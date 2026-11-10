#!/usr/bin/env bash
# Upload MAGIC-abl / PSRO s8891 / FSP s8891 results to Hugging Face (private repos) and W&B. Works on finished
# runs and on runs still training; rerun any time to send what is new. Nothing is retrained or changed.
#   export HF_TOKEN=... WANDB_API_KEY=...     # given by Wentse; never write them into a file
#   DRY=1 bash upload_results.sh               # list what would be uploaded, no network writes
#   bash upload_results.sh                     # upload
#   ONLY=magic bash upload_results.sh          # just some of magic,psro,fsp
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first}" "${WANDB_API_KEY:?export WANDB_API_KEY first}"
MAGIC_WS=${MAGIC_WS:-${MAGIC_DOCKER_WORKSPACE:-$HOME/magic-workspace}}   # MAGIC's workspace
SSP_WS=${SSP_WS:-${SSP_DOCKER_WORKSPACE:-$HOME/safeselfplay-data}}        # PSRO and FSP's workspace
export WANDB_ENTITY=${WANDB_ENTITY:-cwz19} WANDB_PROJECT=${WANDB_PROJECT:-self-play}
export REPO_MAGIC=${REPO_MAGIC:-cwz19/magic-abl-ckpt} REPO_PSRO=${REPO_PSRO:-cwz19/psro-s8891-ckpt} \
       REPO_FSP=${REPO_FSP:-cwz19/fsp-s8891-ckpt}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mounts=(-v "$HERE/ssp_upload.py:/ssp_upload.py:ro")
if [ -d "$MAGIC_WS" ]; then mounts+=(-v "$MAGIC_WS:/magic"); else echo "no MAGIC workspace at $MAGIC_WS"; fi
if [ -d "$SSP_WS" ]; then mounts+=(-v "$SSP_WS:/ssp"); else echo "no PSRO/FSP workspace at $SSP_WS"; fi
# Keys go in by NAME only, so they never appear in Docker's argv.
docker run --rm "${mounts[@]}" \
  -e HF_TOKEN -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT -e REPO_MAGIC -e REPO_PSRO -e REPO_FSP \
  -e DRY="${DRY:-0}" -e ONLY="${ONLY:-}" -e MIN_AGE="${MIN_AGE:-600}" -e HF_HUB_DISABLE_XET=1 \
  python:3.11-slim bash -c 'pip install -q --root-user-action=ignore "huggingface_hub<2" wandb && python /ssp_upload.py'
