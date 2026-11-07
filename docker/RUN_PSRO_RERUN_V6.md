# PSRO 重跑运行说明（v6）

**目的：** 用主 run 的 reward 和 seed 重跑一条 PSRO，到第 5 代和 6×6 矩阵，检查结果能否重现、有没有其他问题。
**截止：** ET 11-11 18:00 前跑完。来不及就交已完成的世代。
**规模：** 一个 4×H200 141GB 节点。时长只有估计值：约 40–65 小时，具体以实测为准。
**有问题先问 Wentse，不要自行改设定。**

## 硬性规则

1. 不要设置 `GAME_REWARD_MODE` 或 `ATTACKER_INVALID_MOVE`。镜像默认就是要用的 reward。
2. 除了本文写的变量，不要改任何训练参数。
3. 一个节点只跑一个 run。FSP 和 PSRO 不能在同一节点同时跑。启动前 `nvidia-smi` 必须显示 4 张卡上没有进程。
4. 中断后用**同一条命令、同一个 suffix** 续跑，不要改 seed 或参数。
5. 不要把 token 写进文件或命令历史。

## 0. 节点要求

- Linux x86_64，4×H200 141GB，已装 NVIDIA 驱动、Docker、NVIDIA Container Toolkit。80 GB 的卡放不下默认配置，先联系 Wentse。
- 能访问 GitHub、ghcr.io、Hugging Face、W&B。
- 工作目录所在磁盘至少留 **150 GB**。
- ghcr 镜像是私有的，要先向 Wentse 拿读取权限。

## 1. 取得启动脚本

代码已经打包在镜像里，这里只用仓库里的启动脚本。

```bash
git clone -b wentse/rebuttal-runbooks https://github.com/WentseChen/SafeSelfPlay.git
cd SafeSelfPlay
```

## 2. 拉取镜像并检查

```bash
docker login ghcr.io
docker pull ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6
docker tag  ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6 \
            safeselfplay:llama8b-h200-invalidmove-20260925-v6
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260925-v6

docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6
docker run --rm --entrypoint bash "$SSP_IMAGE" -c \
  "grep -c _cap_move /opt/selfplay-work/openrlhf/cli/serve_wildguard.py"
```

**两项都要符合，否则停下并告诉 Wentse：**
- digest 是 `sha256:4df8a3c5bc9ee375fdf10e63c462fbf69515fdf48f7d6c4d66ebb330cc9a470b`
- 第二条命令输出 `3`

## 3. 环境变量（每个新 shell 都要执行）

```bash
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260925-v6
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"
export GPU_DEVICES=0,1,2,3
export TOPIC_JUDGE_BACKEND=local RUN_LABELER=local
export TRAINING_SEED=8889
export CELL_STREAMING=1 CELL_ENFORCE_EAGER=0 CELL_SHARE_JUDGE_GPUS=0

read -rsp 'HF token: ' HF_TOKEN; echo; export HF_TOKEN
export WANDB_MODE=disabled
unset WANDB_API_KEY
```

`HF_TOKEN` 需要有 `allenai/wildguard` 的读取权限。

**要用 W&B 和 HF 上传时，**由 Wentse 提供 entity 和 repo，再把上面的 `WANDB_MODE=disabled` 换成 `online` 并设置 `WANDB_API_KEY`、`WANDB_ENTITY`、`HF_CHECKPOINT_REPO`。

## 4. 下载模型并检查

```bash
bash docker/run_llama8b.sh prepare
bash docker/run_llama8b.sh check
```

`check` 要看到四张 GPU。请把它打印的 `image build: git_sha=…` 这一行发给 Wentse。

## 5. 先跑 smoke test

这一步会跑 A1、D1 各 2 步和 4 格收益矩阵（smoke 每格只跑 256 局），确认流程能走通，约 15 分钟。它用单独的 suffix，不影响正式 run。

```bash
SAVE_STEPS_CONTRACT=1 bash docker/run_llama8b.sh psro \
  --run-suffix "psro_smoke_s8889" --rounds 1 --steps-per-role 2 --matrix-episodes 256
```

失败就把最后 50 行日志发给 Wentse，不要继续。

## 6. 正式启动

```bash
DETACH=1 CONTAINER_NAME=ssp-psro-s8889 bash docker/run_llama8b.sh psro \
  --run-suffix "psro5_rerun_v6_s8889" --rounds 5 --steps-per-role 300

docker logs -f ssp-psro-s8889
```

**日志里必须出现 `Starting psro: …; reward mode=legacy`。** 如果显示 `shared_binary`，立刻停掉并告诉 Wentse。

要停止时：

```bash
docker stop -t 180 ssp-psro-s8889
```

中断后，用同样的环境变量和同一条命令续跑。容器内失败也会自动重试。**角色训练中途停掉，该角色会从第 0 步重训**，尽量等角色结束再停。

## 7. 运行中要看的

每天看一次就够了。

| 看什么 | 正常 | 异常时 |
|---|---|---|
| `training/<角色>/early_stop.json` | `returncode=0`。有的角色不到 300 步就早停，这是正常的 | 不是 0 就看 `training.log` 末尾 100 行，发给 Wentse |
| `state.json` 里的 `stage` | 按 A1、D1、收益矩阵、A2、D2……前进 | 超过 2 小时没变化就检查容器是否还在（`docker ps -a`） |
| `matrices/` | 每一代后出现新矩阵 | 没有出现就告诉 Wentse |

运行目录是 `$SSP_DOCKER_WORKSPACE/psro/role_lora_zero_sum_psro/psro5_rerun_v6_s8889/`。

## 8. 交付

1. A1–A5、D1–D5 的最终 checkpoint，以及各自的 `early_stop.json`、`manifest.json`。
2. `state.json` 和 `matrices/`。
3. 每个角色完成的时间，以及你用的是几张什么卡。

评测不用跑，Wentse 那边会跑。不要上传基座权重、`wandb/`、`topic_labels.sqlite3` 或任何 key。
