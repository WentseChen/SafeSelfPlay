# 全部实验重跑说明（invalid-move reward，镜像 v4，4×H200）

PSRO（seed 8888、8889）：已有的 `inv1` run 用 v4 **原 suffix 续跑**（见第 4 节末「已有的 PSRO run」）。
uniform + naive：旧 run 用的是 `shared_binary` reward，与新 PSRO 不可比，用 v4 + **新 suffix** 从头跑。
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

# 收益格加速（必须设置；原因与实测见第 4 节「收益格加速设置」）。三项都不是矩阵/合约字段，不改变收益值。
export CELL_STREAMING=1         # 流式收益格：连续批处理，每局单独推进，不再等一批 64 条里最长的回复。镜像默认 0。
export CELL_ENFORCE_EAGER=0     # 用 CUDA graph 解码。带 LoRA 的格子被 vLLM 降级到 V0 引擎，eager 下解码慢 4 倍。
export CELL_SHARE_JUDGE_GPUS=0  # 不在 judge 所在的卡（GPU 0、GPU 3）上加开生成引擎：开着会在启动时 CUDA OOM。
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
  新 attacker / 新 defender 那一行/列总是跑满 4000 局。按第 2 节的加速设置，一个 4000 局的格子在 4×H200 上约 15 分钟
  （未加速时带 LoRA 的格子要 70–140 分钟）。
- 中断后用**同一条命令、同一个 `--run-suffix`** 续跑；不要改 seed 或参数后复用 suffix（会被拒绝）。
- 失败会在容器内自动重试（最多 5 次，间隔 60 秒），从 `state.json` 续跑。
- 已经在跑、但启动时没设第 2 节三个 `CELL_*` 变量的 run：`docker stop -t 180 <name>`，在 shell 里 export 这三个变量，
  再用**同一条命令、同一个 suffix** 重启。它们不是合约字段，已算好的格子保留，续跑不会被拒绝。

### 已有的 PSRO run（invalid-move reward，`*_inv1_*`）改用 v4 续跑

PSRO s8888 / s8889 的旧 run 与 v4 reward 完全相同，**不要从头跑**，用 v4 + **原来的 suffix** 续跑：
已完成的角色和收益格全部保留，只有中断的那个角色从 step 0 重训（`PSRO_MIDRUN_RESUME=0`）。

1. 选停机时机：**等当前角色的 final checkpoint 上传后**（进入收益格阶段）再停，否则该角色要重训。已经死掉的 run 直接重启。
2. `docker stop -t 180 <旧容器名>`；`nvidia-smi` 确认 4 张卡空闲。
3. 按第 1–2 节换成 v4 镜像并 export 三个 `CELL_*` 变量，再用**与原来完全相同的命令和 `--run-suffix`** 启动。
   启动日志应显示 resume（不是 fresh run）；若报 `Training settings changed`，说明命令参数与原来不一致，停下来找我们。
4. 重启前请把旧 run 的 `training/<中断的角色>/training.log` 最后 100 行发给我们，确认崩溃原因已在 v4 修复。

### 收益格加速设置（为什么要设第 2 节那三个 `CELL_*`）

**问题**：除 A0×D0 外，所有收益格都慢得多。同一代码、4000 局：A0×D0 21 分钟，A1×D0 73 分钟（远程节点约 2 小时 20 分钟）。

**原因**：只要格子里有训练过的 LoRA（A0×D0 以外的所有格子），收益格就用 `enable_lora=True` 建引擎，
vLLM 0.8.2 的 V1 引擎不支持 LoRA，会降级到 V0（cell 日志里有 `LORA is experimental on VLLM_USE_V1=1. Falling back to V0 Engine`）。
V0 + eager 模式解码速度只有 V1 的 1/4（H200 实测 1716 vs 6842 tok/s）。再加上旧的 lockstep 路径每批 64 条要等最长的回复，
格子所在 GPU 利用率只有约 23%。收益值本身没问题，只是慢。

**两个开关的作用**：
- `CELL_ENFORCE_EAGER=0`：用 CUDA graph 解码，V0+LoRA 恢复到 5460 tok/s（3.2 倍）。单独使用（lockstep）时 GPU 利用率到 95%。
- `CELL_STREAMING=1`：流式收益格，每张卡一个连续批处理引擎，每局单独推进（攻击 → 防御 → 判分），没有批次等待。
  流式模式下引擎默认就用 CUDA graph；仍显式设置 `CELL_ENFORCE_EAGER=0`，以防关掉流式时退回 eager。

**GPU 共享与 OOM**：流式模式默认（`CELL_SHARE_JUDGE_GPUS=1`）会在 WildGuard（GPU 0）和 topic judge（GPU 3）所在的卡上
再开生成引擎。实测 4 个格子里 3 个的这类引擎启动时 CUDA OOM（`engine pool: engines failed … Tried to allocate 3.40 GiB`）：
引擎按整卡 90% 的上限定大小，vLLM 显存分析之后还要再分配 3.4 GiB，judge 旁边放不下。
不致命，格子会只用剩下的引擎算完，但会留下 traceback、白占那张卡，还有把 judge 挤到 OOM 的风险。
设 `CELL_SHARE_JUDGE_GPUS=0` 后每个格子只用自己的卡，不再碰 judge 的卡，速度几乎不变。

**实测**（4×H200，真实 PSRO driver `cold_psro_local.py`，4 个格子 A0/A1 × D0/D1，每格 1000 局，2 个格子并行）：

| 设置 | 耗时 | 问题 |
| --- | --- | --- |
| 原设置（lockstep + eager；单格测试 A1×D0 1000 局） | 单格 19.4 分钟 | 慢 |
| `CELL_STREAMING=0 CELL_ENFORCE_EAGER=0` | 4 格共 16.0 分钟 | 无 |
| `CELL_STREAMING=1 CELL_ENFORCE_EAGER=0` | 4 格共 8.3 分钟 | judge 卡上的引擎 3/4 次 OOM（不致命） |
| **`CELL_STREAMING=1 CELL_ENFORCE_EAGER=0 CELL_SHARE_JUDGE_GPUS=0`（采用）** | **4 格共 8.7 分钟** | **无** |

三种加速设置的收益值彼此一致，也与 4000 局的值一致，差异都在 1000 局的 95% 置信区间内（A1×D0：0.112–0.119 vs 0.1205；
A1×D1：−0.748 到 −0.744 vs −0.744）。
确认生效：`<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log` 里应有 `engine pool ready on GPUs ['1']`（或 `['2']`）
和 `stream: … episodes judged` 行，且没有 `engines failed`。

## 5. 运行中要看的指标

| 什么时候 | 看什么 | 正常 | 异常 → 处理 |
| --- | --- | --- | --- |
| 每个 attacker 阶段 | W&B `attacker/debug/n_generated_benign` | 大致持平（不应持续下降） | 降到接近 0 → 通知我们 |
| 每个 attacker 阶段 | `topic/attacker_invalid_moves` | 逐步降到接近 0 | 持续上升 → 通知我们 |
| 阶段结束 | W&B 显示 `crashed` | 早停正常现象；`training/<A_k>/early_stop.json` 里 `returncode=0` 即正常 | `returncode≠0` → 看 `training.log` 末尾 |
| A1 结束后 | 进入矩阵计算，约 30 分钟内开始 D1 | 出现新的 D1 W&B run | 超过 1 小时无新 run → 看 `<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log`，确认有 `stream:` 行（没有说明第 2 节的 `CELL_*` 没生效） |

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
| digest | `sha256:dcd49b1b149e592e4907268c6d2de0af3ea1e49167cfaae6ae8c720a29cef1cb` |
| 构建 commit（`check` 应显示） | `b815827523d2cf3d68d7090623eff41ba3e9a6ac` |
| 默认配置 | `GAME_REWARD_MODE=legacy`、`ATTACKER_INVALID_MOVE=1`、`RACE_CELLS=1`、`CELL_STREAMING=0`、4×H200（`CELL_STREAMING` 由第 2 节改为 1） |

**v4 相对 v3**：
- 每个训练阶段结束后（包括早停），杀掉训练用 WildGuard judge 的 Ray 残留进程，并确认 4 张卡都已释放，再启动收益格；
  每次进入训练前也确认 4 张卡都已释放（v3 只检查 GPU 0）。
- 包含流式收益格代码（默认关闭；已在 H200 上测完，按第 2 节开启）。

**v3 相对 v2**：收益格结束切换到训练时，cell judge 可能未释放 GPU 0，导致下一个训练阶段起不来（seed 8889 曾卡在 A1 前）；
v3 强制清理并等待 GPU 0 空闲，driver 失败自动续跑。

旧标签 `…-invalidmove-20260923`、`-v2`、`-v3` **不要再用于新启动**。
