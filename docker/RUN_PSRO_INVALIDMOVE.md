# 全部实验重跑说明（invalid-move reward，镜像 v4，4×H200）

本次**全部重跑**：PSRO（seed 8888、8889）+ uniform + naive，全部用镜像 **v4**、**新的 `--run-suffix`**（旧 run 不续跑）。
代码：分支 `wentse/v6d-invalid-move-reward`。镜像已预先构建，**不需要 build**。

v4 = v3 的全部修正 + 阶段切换时的 GPU 释放加固（见文末「镜像信息」）。

## 0. 节点要求

- Linux x86_64，单节点 4×H200，已装 NVIDIA 驱动、Docker、NVIDIA Container Toolkit（见 `QUICKSTART.source.llama8b.md` 第 0 节）。
- 能访问 GitHub、ghcr.io、Hugging Face、W&B。
- 工作目录（`SSP_DOCKER_WORKSPACE`）所在磁盘每个 seed 至少留 **150 GB**（模型约 35 GB；一个完整 8 代 run 约 65 GB checkpoint 与日志）。
- GitHub 账号需能访问私有仓库 `xudongwu-0/SafeSelfPlay`；ghcr 镜像为私有时需被授权。
- **每个 run 占一个 4 卡节点**（PSRO s8888、PSRO s8889、uniform、naive 共 4 个 run）。节点不够就排队依次跑，
  不要在同一节点同时跑两个 run（会争 GPU 和 Ray 端口）。

## 1. 代码与镜像（每个节点都做一次）

```bash
git clone -b wentse/v6d-invalid-move-reward https://github.com/xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay

docker login ghcr.io          # 镜像为私有时需要；token 需 read:packages
docker pull ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260924-v4
docker tag  ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260924-v4 \
            safeselfplay:llama8b-h200-invalidmove-20260924-v4
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260924-v4
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

`check` 会打印四张 GPU 和 `image build: git_sha=…`，该 commit 必须与本文件末尾「镜像信息」一致，不是的话停下，镜像不对。

**上传权限预检（必须做，10 秒）**：`HF_TOKEN` 需要能**写** `xudongwu/SafeSelfPlay-checkpoints`，否则自动上传只会在日志里记一行 `checkpoint publish FAILED`，训练继续——上一轮就是这样一个 checkpoint 都没传上去。

```bash
docker run --rm --entrypoint bash -e HF_TOKEN -e NODE="$(hostname)" "$SSP_IMAGE" -c \
  'echo "preflight $NODE" > /tmp/p.txt && huggingface-cli upload xudongwu/SafeSelfPlay-checkpoints /tmp/p.txt "_preflight/$NODE.txt"'
```

报错（401/403）就换一个对该仓库有 write 权限的 token，再开始训练。

## 4. 启动（每个节点一个 run）

先 `nvidia-smi` 确认 4 张卡上没有残留进程（旧容器先 `docker stop -t 180 <name>`）。

```bash
# PSRO seed 8888
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-psro-s8888 \
  bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv4_s8888" --rounds 8 --steps-per-role 300

# PSRO seed 8889
TRAINING_SEED=8889 DETACH=1 CONTAINER_NAME=ssp-psro-s8889 \
  bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv4_s8889" --rounds 8 --steps-per-role 300

# uniform（seed 8888）
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-uniform-s8888 \
  bash docker/run_llama8b.sh uniform --run-suffix "uniform8_${RUN_LABELER}_inv4_s8888" --rounds 8 --steps-per-role 300

# naive self-play（seed 8888）
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-naive-s8888 \
  bash docker/run_llama8b.sh naive --run-suffix "naive6_${RUN_LABELER}_inv4_s8888" --rounds 6 --steps-per-role 100

docker logs -f ssp-psro-s8888      # 第一行应为 `Starting psro: …; reward mode=legacy`
docker stop -t 180 ssp-psro-s8888  # 需要停止时
```

- racing（`RACE_CELLS=1`）是镜像默认：每格先跑 256 局 pilot，只有 Nash 相关的行/列补到 4000 局。
  新 attacker / 新 defender 那一行/列总是跑满 4000 局，一个格子在 4×H200 上约 1–2 小时。
- 中断后用**同一条命令、同一个 `--run-suffix`** 续跑；不要改 seed 或参数后复用 suffix（会被拒绝）。
- 失败会在容器内自动重试（最多 5 次，间隔 60 秒），从 `state.json` 续跑。
- 流式收益格（`CELL_STREAMING=1`，多卡并行生成，更快）默认**关闭**，等我们在 H200 上测完再通知开启；
  它不是矩阵设置，之后停容器、加 `CELL_STREAMING=1` 用同一 suffix 重启即可，已算好的格子不受影响。

## 5. 运行中要看的指标

| 什么时候 | 看什么 | 正常 | 异常 → 处理 |
| --- | --- | --- | --- |
| 每个 attacker 阶段 | W&B `attacker/debug/n_generated_benign` | 大致持平（不应持续下降） | 降到接近 0 → 通知我们 |
| 每个 attacker 阶段 | `topic/attacker_invalid_moves` | 逐步降到接近 0 | 持续上升 → 通知我们 |
| 阶段结束 | W&B 显示 `crashed` | 早停正常现象；`training/<A_k>/early_stop.json` 里 `returncode=0` 即正常 | `returncode≠0` → 看 `training.log` 末尾 |
| A1 结束后 | 进入矩阵计算，约 1 小时内开始 D1 | 出现新的 D1 W&B run | 卡在矩阵数小时无新 run → 看 `<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log` |

在我们的 2×H200 测试里（同一 reward，A1 vs D0，截至 151 step），每批 benign 数量稳定在 26–36，
attacker 成功率 0.33 → 0.96，非法动作 0.38 → 0.02；用 step 40 checkpoint 算的 512 局收益格只需 516 个候选即填满（benign 漂移 0.8%）。上一轮 `shared_binary` 在 80–160 step 内 benign 降到 0，导致矩阵卡住。

## 6. Checkpoint 上传规则

1. 每个角色阶段（A1、D1、A2……）结束立即上传最终 checkpoint：
   `training/<A_k|D_k>/ckpt/global_step<N>_hf/`（N 为 early_stop 的停止 step，无早停取最大 N），
   加上 `early_stop.json`、`manifest.json`。设置了 `HF_CHECKPOINT_REPO` 时会自动上传，但**必须确认** repo 里出现
   `adapter_model.*`；上一轮自动上传静默失败了。失败时手动上传：
   ```bash
   huggingface-cli upload xudongwu/SafeSelfPlay-checkpoints \
     <run>/training/A1/ckpt/global_step200_hf psro8_local_inv4_s8888/A1 \
     --commit-message "psro8_local_inv4_s8888 A1 step200"
   ```
2. 训练中每 50 step 把最新 checkpoint 上传到 `<suffix>/<A_k>/latest/`（覆盖）。
3. 每算完一个收益格，上传 `<run>/state.json` 和 `<run>/matrices/`。
4. 不要上传基座权重、`wandb/`、`topic_labels.sqlite3` 或任何 key。

## 镜像信息

| 项目 | 值 |
| --- | --- |
| 镜像 | `ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260924-v4` |
| digest | `DIGEST_PLACEHOLDER` |
| 构建 commit（`check` 应显示） | `COMMIT_PLACEHOLDER` |
| 默认配置 | `GAME_REWARD_MODE=legacy`、`ATTACKER_INVALID_MOVE=1`、`RACE_CELLS=1`、`CELL_STREAMING=0`、4×H200 |

**v4 相对 v3**：
- 每个训练阶段结束后（包括早停），杀掉训练用 WildGuard judge 的 Ray 残留进程，并确认 4 张卡都已释放，再启动收益格；
  每次进入训练前也确认 4 张卡都已释放（v3 只检查 GPU 0）。
- 包含流式收益格代码（默认关闭）。

**v3 相对 v2**：收益格结束切换到训练时，cell judge 可能未释放 GPU 0，导致下一个训练阶段起不来（seed 8889 曾卡在 A1 前）；
v3 强制清理并等待 GPU 0 空闲，driver 失败自动续跑。

旧标签 `…-invalidmove-20260923`、`-v2`、`-v3` **不要再用于新启动**。
