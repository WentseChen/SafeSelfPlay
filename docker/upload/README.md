# 上传 rebuttal 实验结果（给执行的人或 agent）

**目标：** 把这台机器上三个实验的结果上传到 Hugging Face（私有 repo）和 W&B。
- MAGIC-abl
- PSRO seed 8891
- FSP seed 8891

**只上传，不重跑、不改动任何训练。实验跑完或还在跑都可以传，之后可以随时再执行一次，补传新的部分。**

## 规则（必须遵守）

1. 不要停止、重启或续跑任何训练，也不要 `docker stop` 正在跑的容器。上传和训练同时进行没有问题。
2. 不要删除、移动或修改工作目录里的任何文件。上传工具会在被上传的目录里建一个 `.cache/huggingface` 续传缓存，这是唯一的写入。
3. `HF_TOKEN` 和 `WANDB_API_KEY` 由 Wentse 提供，只能用 `export` 设成环境变量。
   - 不要写进任何文件、git、日志或聊天记录，也不要打印出来。
   - 不要改 `upload_results.sh`。
4. Hugging Face repo 都是 **private**，不要改成公开，也不要把链接贴到公开的地方。
5. 不要另外上传其他东西。脚本已经排除了：
   - 基础模型权重（`hf/`、`llama31_abl_ittok/` 等）
   - optimizer 状态（`optim_*`、`extra_state_*`）
   - `wandb/` 目录
   - `topic_labels.sqlite3`

## 会上传什么

| 实验 | 本地目录（默认） | HF repo | 内容 |
|---|---|---|---|
| MAGIC | `$MAGIC_DOCKER_WORKSPACE`（默认 `~/magic-workspace`） | `cwz19/magic-abl-ckpt` | 第 100、200、… 步的 `ckpt/Game-separated/*/global_step_*/defender/`（不含 optimizer）。训练停止后（有 `logs/CONVERGED`，或跑到第 600 步），再加上最后一步。另外上传 `logs/` |
| PSRO | `$SSP_DOCKER_WORKSPACE/psro/role_lora_zero_sum_psro/psro5_rerun_v6_s8891/` | `cwz19/psro-s8891-ckpt` | `state.json`、`matrices/`；每个角色的 `ckpt/*_hf/`、`training.log`、`early_stop.json`、`manifest.json`、`resume_state.json` |
| FSP | `$SSP_DOCKER_WORKSPACE/selfplay_baselines/uniform/uniform5_local_inv4_s8891/` | `cwz19/fsp-s8891-ckpt` | 同 PSRO |

**跑到一半时：** 10 分钟内还在变动的文件先不传，等下一次执行再传，所以不会传到写了一半的 checkpoint。

**W&B（entity `cwz19`，project `self-play`）：**
- 有离线记录（`wandb/offline-run-*`）的：等它 30 分钟没有更新（代表那个 run 已结束）再 `wandb sync`。
- 已经在线记录（只有 `wandb/run-*`）的：不重复上传。
- 两者都没有的：从训练日志重建曲线。MAGIC 读 `logs/train.log`；PSRO 和 FSP 读每个角色的 `training.log`。
  - 每个 run 名称固定：MAGIC 是 `magic-abl__D-llama31abl-A-magicsft`，PSRO 和 FSP 是 `<run_suffix>__<角色>`。
  - 再次执行时，会接在同一个 run 后面补新的步数，不会重复建 run。

## 步骤

**需要：**
- Docker。脚本在临时的 `python:3.11-slim` 容器里执行，主机上不用装任何 Python 套件。
- 能连到 Docker Hub、Hugging Face 和 W&B 的网络。

1. **取得脚本。** 这一步不影响正在跑的训练：
   ```bash
   git clone -b wentse/rebuttal-runbooks https://github.com/WentseChen/SafeSelfPlay.git ssp-upload
   cd ssp-upload/docker/upload
   ```
2. **设定 key 和目录**（key 由 Wentse 提供）：
   ```bash
   export HF_TOKEN=...
   export WANDB_API_KEY=...
   export MAGIC_WS=$HOME/magic-workspace      # 跑 MAGIC 时的 MAGIC_DOCKER_WORKSPACE
   export SSP_WS=$HOME/safeselfplay-data      # 跑 PSRO/FSP 时的 SSP_DOCKER_WORKSPACE
   ```
   不确定目录在哪的话：
   - 用 `docker inspect <容器名> --format '{{json .Mounts}}'` 查训练容器挂载到 `/workspace` 的主机目录。容器名默认是 `magic-abl`、`ssp-psro-s8891`、`ssp-uniform-s8891`。
   - 或者找含 `ckpt/Game-separated` 的目录，以及含 `state.json` 的 `psro5_rerun_v6_s8891`、`uniform5_local_inv4_s8891` 目录。
3. **先试跑**（不会上传）：
   ```bash
   DRY=1 bash upload_results.sh
   ```
   - 每个实验会印出 `HUB <名称>: N files, X GB -> <repo> (M still being written, next pass)` 和 `WANDB ...`，最后一行是 `ALL_DONE`。
   - 出现 `SKIP ... not found`，表示目录设错了，回到第 2 步。
   - 还没开始跑的实验，出现 `SKIP` 是正常的。
4. **上传：**
   ```bash
   nohup bash upload_results.sh > upload.log 2>&1 &
   tail -f upload.log
   ```
   - 每个实验传完会印 `UPLOAD_OK <名称> <repo 链接>`。
   - 只传部分实验时加 `ONLY=`，值是 `magic`、`psro`、`fsp`，多个用逗号分隔，例如 `ONLY=psro,fsp`。
   - 中断或失败时，重跑同一条命令，会从断点接着传。几百 GB 可能要好几个小时。
5. **训练还在跑的话**，每天执行一次第 4 步。训练全部结束后再执行最后一次，把最后的 checkpoint 和日志补齐。
6. **核对：**
   - HF：repo 的文件数，要和第 3 步印出的 `N files` 加上之前已传的相符。
   - W&B：在 `cwz19/self-play` 能看到对应的 run，最后一步和训练进度一致。

## 回报给 Wentse

- 每个实验：训练状态（跑到哪一步或哪个角色，或已完成），这次上传了几个文件、多少 GB，以及 HF repo 和 W&B run 名称。
- 用了几张、什么型号的卡，以及每个实验的开始时间。
- 有错误的话，附上 `upload.log` 最后 50 行。先确认里面没有 key。
