#!/usr/bin/env bash
# Build/run the MAGIC baseline container (abliterated Llama-3.1-8B, 4xH200). Same layout as run_llama8b.sh.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${MAGIC_IMAGE:-safeselfplay:magic-abl-llama8b-20261107}
ACTION=${1:---help}
if [ "$#" -gt 0 ]; then shift; fi
case "$ACTION" in
  -h|--help|help)
    echo "Usage: $0 build | check | prepare | train"
    echo "Defaults: GPU_DEVICES=0,1,2,3; MAGIC_DOCKER_WORKSPACE=$REPO/output/docker-magic"
    exit 0 ;;
  build)
    # Stamp the commit being baked in, as run_llama8b.sh does: the scripts are COPYed into the image, never mounted.
    REVISION=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
    if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
       ! git -C "$REPO" diff --quiet HEAD; then REVISION="$REVISION-dirty"; fi
    exec docker build --platform linux/amd64 -f "$REPO/docker/Dockerfile.magic" \
      --build-arg "SSP_GIT_SHA=$REVISION" \
      --build-arg "SSP_BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      -t "$IMAGE" "$@" "$REPO" ;;
  check|prepare|train) ;;
  *) echo "Unknown command: $ACTION" >&2; exit 2 ;;
esac

WORKSPACE=${MAGIC_DOCKER_WORKSPACE:-$REPO/output/docker-magic}
mkdir -p "$WORKSPACE"
WORKSPACE=$(cd "$WORKSPACE" && pwd)
OPTIONS=(run --rm --init --stop-timeout 180 --shm-size "${MAGIC_SHM_SIZE:-32g}"
         --ulimit nofile=65536:65536
         --mount "type=bind,source=$WORKSPACE,target=/workspace")
# Downloads do not need GPUs.
if [ "$ACTION" != prepare ]; then OPTIONS+=(--gpus "\"device=${GPU_DEVICES:-0,1,2,3}\""); fi
if [ "${DETACH:-0}" = 1 ]; then OPTIONS+=(-d); fi
if [ -n "${CONTAINER_NAME:-}" ]; then OPTIONS+=(--name "$CONTAINER_NAME"); fi
# Ports for Ray and the judge, in case another run shares the host network namespace settings.
OPTIONS+=(-e "RAY_MASTER_PORT=${RAY_MASTER_PORT:-6391}" -e "JPORT=${JPORT:-18471}" -e "MAX_RESP=${MAX_RESP:-6144}"
         -e "GPUS_PER_ROLE=${GPUS_PER_ROLE:-2}" -e "TOTAL_STEPS=${TOTAL_STEPS:-600}" -e "ROLLOUT_MEM=${ROLLOUT_MEM:-0.5}"
         -e "MIN_STEPS=${MIN_STEPS:-300}" -e "PATIENCE=${PATIENCE:-6}" -e "MIN_DELTA=${MIN_DELTA:-0.01}" -e "FORCE=${FORCE:-0}")

case "$ACTION" in
  check)
    exec docker "${OPTIONS[@]}" --entrypoint bash "$IMAGE" -c '
      cat /opt/BUILD_INFO
      python -c "import torch, vllm, transformers, flash_attn; print(\"ENV\", torch.__version__, vllm.__version__, transformers.__version__, flash_attn.__version__, \"gpus\", torch.cuda.device_count())"
      grep -c "seed_prompt = problem_inputs" /opt/magic/src/verl/verl/separated_trainer/ppo/multi_agent_rollout.py' ;;
  prepare)
    # Models into /workspace/hf, then the defender base (abliterated weights + official Instruct template).
    exec docker "${OPTIONS[@]}" --entrypoint bash "$IMAGE" -c '
      python - <<PY
from huggingface_hub import snapshot_download as s
for repo in ["mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
             "XiaoyuWen/MAGIC-attacker-sft-Llama3.1-8B-Instruct",
             "Qwen/Qwen3Guard-Gen-8B"]:
    print("MODEL", repo, s(repo, ignore_patterns=["optimizer*", "*.pth"]), flush=True)
PY
      python /opt/magic-run/make_base.py --out /workspace/llama31_abl_ittok' ;;
  train)
    exec docker "${OPTIONS[@]}" --entrypoint bash "$IMAGE" /opt/magic-run/run.sh "$@" ;;
esac
