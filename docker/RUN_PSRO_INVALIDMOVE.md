# PSRO 两个 seed 运行说明（invalid-move reward，4×H200）

本次只跑 **PSRO**，两个 seed：`8888` 和 `8889`。
代码：分支 `wentse/v6d-invalid-move-reward`（commit `03fc5c5` 起）。镜像已预先构建，**不需要 build**。

> 注意：上一轮的 psro-a / psro-b 两个 run 实际都是 seed 8888（W&B config 里 `seed=8888`），
> 并不是两个 seed。本次用 `TRAINING_SEED` 显式区分。

## 0. 节点要求

- Linux x86_64，单节点 4×H200，已装 NVIDIA 驱动、Docker、NVIDIA Container Toolkit（见 `QUICKSTART.source.llama8b.md` 第 0 节）。
- 能访问 GitHub、ghcr.io、Hugging Face、W&B。
- **每个 seed 占一个 4 卡节点**。两个 seed 并行就用两个节点；只有一个节点就先跑 seed 8888，结束后再跑 8889。
  不要在同一节点同时跑两个 run（会争 GPU 和 Ray 端口）。

## 1. 代码与镜像（每个节点都做一次）

```bash
git clone -b wentse/v6d-invalid-move-reward https://github.com/xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay

docker login ghcr.io          # 镜像为私有时需要；token 需 read:packages
docker pull ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260923
docker tag  ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260923 \
            safeselfplay:llama8b-h200-invalidmove-20260923
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260923
```

## 2. 环境变量

```bash
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"   # 模型、checkpoint、日志都在这里
export GPU_DEVICES=0,1,2,3
export TOPIC_JUDGE_BACKEND=local RUN_LABELER=local      # 本地 topic judge，不需要 API key
# 不要设置 GAME_REWARD_MODE / ATTACKER_INVALID_MOVE：镜像默认就是新 reward（legacy + 1）。

read -rsp 'HF token: ' HF_TOKEN; echo; export HF_TOKEN              # 需有 allenai/wildguard 权限
read -rsp 'W&B key: '  WANDB_API_KEY; echo; export WANDB_API_KEY
export WANDB_ENTITY=2373025856w-the-university-of-hong-kong WANDB_PROJECT=self-play WANDB_MODE=online
export HF_CHECKPOINT_REPO=xudongwu/SafeSelfPlay-checkpoints        # 每个角色结束自动上传最终 checkpoint
```

## 3. 下载模型并检查（每个节点一次；工作目录在共享存储上则只需一次）

```bash
bash docker/run_llama8b.sh prepare
bash docker/run_llama8b.sh check
```

`check` 输出里 reward mode 必须是 `legacy`，镜像 commit 为 `03fc5c5…`。不是的话停下，镜像不对。

## 4. 启动（每个节点一个 seed）

```bash
# 节点 1
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-psro-s8888 \
  bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv1_s8888" --rounds 8 --steps-per-role 300

# 节点 2
TRAINING_SEED=8889 DETACH=1 CONTAINER_NAME=ssp-psro-s8889 \
  bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv1_s8889" --rounds 8 --steps-per-role 300

docker logs -f ssp-psro-s8888      # 看日志
docker stop -t 180 ssp-psro-s8888  # 需要停止时
```

同一个 `--run-suffix` 重新执行同一条命令即可续跑；不要换 seed 或改参数后复用旧 suffix（会被拒绝）。

## 5. 运行中要看的指标

| 什么时候 | 看什么 | 正常 | 异常 → 处理 |
| --- | --- | --- | --- |
| 每个 attacker 阶段 | W&B `attacker/debug/n_generated_benign` | 大致持平（不应持续下降） | 降到接近 0 → 通知我们 |
| 每个 attacker 阶段 | `topic/attacker_invalid_moves` | 逐步降到接近 0 | 持续上升 → 通知我们 |
| 阶段结束 | W&B 显示 `crashed` | 早停正常现象；`training/<A_k>/early_stop.json` 里 `returncode=0` 即正常 | `returncode≠0` → 看 `training.log` 末尾 |
| A1 结束后 | 进入矩阵计算，约 1 小时内开始 D1 | 出现新的 D1 W&B run | 卡在矩阵数小时无新 run → 看 `<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log` |

在我们的 2×H200 测试里（同一 reward，A1 vs D0），benign 数量 200 step 内保持稳定，attacker 成功率 0.33 → 0.8，
非法动作 0.38 → 0.02。上一轮 `shared_binary` 在 80–160 step 内 benign 降到 0，导致矩阵卡住。

## 6. Checkpoint 上传规则

1. 每个角色阶段（A1、D1、A2……）结束立即上传最终 checkpoint：
   `training/<A_k|D_k>/ckpt/global_step<N>_hf/`（N 为 early_stop 的停止 step，无早停取最大 N），
   加上 `early_stop.json`、`manifest.json`。设置了 `HF_CHECKPOINT_REPO` 时会自动上传，但**必须确认** repo 里出现
   `adapter_model.*`；上一轮自动上传静默失败了。失败时手动上传：
   ```bash
   huggingface-cli upload xudongwu/SafeSelfPlay-checkpoints \
     <run>/training/A1/ckpt/global_step200_hf psro8_local_inv1_s8888/A1 \
     --commit-message "psro8_local_inv1_s8888 A1 step200"
   ```
2. 训练中每 50 step 把最新 checkpoint 上传到 `<suffix>/<A_k>/latest/`（覆盖）。
3. 每算完一个收益格，上传 `<run>/state.json` 和 `<run>/matrices/`。
4. 不要上传基座权重、`wandb/`、`topic_labels.sqlite3` 或任何 key。
