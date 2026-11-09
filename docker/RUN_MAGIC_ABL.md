# MAGIC baseline：在 abliterated Llama-3.1-8B 上重跑

**目的：** 用 MAGIC 官方代码（[AI45Lab/MAGIC](https://github.com/AI45Lab/MAGIC)，原 BattleWen/MAGIC），从和我们 PSRO 相同的底座（abliterated Llama-3.1-8B）出发训练，和 PSRO 做公平比较。

**需要：** Linux x86_64，4×H200 141GB，NVIDIA 驱动、Docker、NVIDIA Container Toolkit（和跑 FSP、PSRO 的要求相同），约 **600 GB** 磁盘（模型约 50 GB；每个完整 checkpoint 含两个模型和 optimizer，约 126 GB，只保留最新两个；第 100、200、…、600 步只留 defender 权重，每个约 32 GB），能访问 GitHub、Hugging Face。代码和模型都是公开的，不需要任何 token。

**11-09 更新：** 回答上限改回论文的 6144，最多跑 600 步，收敛就提前停（见第 3 节）。之前已经 build 过镜像的话，`git pull` 后重新 `build` 一次（只有最后一层会变，很快）；已经用旧设置开始训练的话，停掉，换一个新的空 `MAGIC_DOCKER_WORKSPACE` 从头跑，不要接旧的 checkpoint。

和 FSP、PSRO 一样用 Docker；不同的是这个镜像在你的机器上从源码构建（和 9 月那次交付相同），不用从 ghcr 拉。

**已实测（11-07，Lambda 1×A100）：** `build` 约 5 分钟，镜像 21.8 GB；`check` 通过；`prepare` 约 5 分钟，下载 46 GB；镜像内 judge 按 MAGIC 的 reward 代码打分正确。训练在 rhea 4×H200 上跑过 24 步（不是用这个镜像）。

**如果 `docker` 要 sudo**（报 permission denied）：`sudo usermod -aG docker $USER`，然后重新登录；没有 sudo 权限就请管理员加。

## 1. 构建镜像（只做一次，约 5–10 分钟）

```bash
git clone -b wentse/rebuttal-runbooks https://github.com/WentseChen/SafeSelfPlay.git
cd SafeSelfPlay
bash docker/run_magic.sh build 2>&1 | tee magic_build.log
bash docker/run_magic.sh check
```

`check` 要输出这三样，不对就停下来，把 `magic_build.log` 发给 Wentse：

```
git_sha=...   magic_commit=85a96985a4e7cb01fec8084b6a6bd9df3762362d
ENV 2.6.0+cu124 0.8.5.post1 4.51.3 2.7.4.post1 gpus 4
1
```

## 2. 下载模型并建底座（不占 GPU，约 46 GB，约 5 分钟）

```bash
export MAGIC_DOCKER_WORKSPACE=$HOME/magic-workspace     # 换成有 500 GB 空间的目录
bash docker/run_magic.sh prepare 2>&1 | tee magic_prepare.log
```

最后一行要是 `BASE_OK /workspace/llama31_abl_ittok`。

## 3. 训练

```bash
export MAGIC_DOCKER_WORKSPACE=$HOME/magic-workspace GPU_DEVICES=0,1,2,3
DETACH=1 CONTAINER_NAME=magic-abl bash docker/run_magic.sh train
docker logs -f magic-abl
```

- 如果这台机器同时在跑 PSRO 或 FSP，`GPU_DEVICES` 要选另外 4 张空卡。端口冲突时可设 `RAY_MASTER_PORT=6491 JPORT=18571`。
- 启动约 4 分钟：先起 judge，再加载两个模型。
- 中断后用同一条命令重跑，会从最新的 checkpoint 接着跑（每 25 步存一次）。停止用 `docker stop -t 180 magic-abl`。
- 最多 600 步，回答上限 6144（和 MAGIC 论文相同）。我们在 4×H200 上量到每步 180–200 秒，600 步约 30–33 小时。
- **收敛就提前停：** 第 300 步以后，如果验证分数（`val/test_score/game`，每 15 步一次）连续 6 次（90 步）都没有比最好成绩高出 0.01，脚本会在下一个完整 checkpoint 存好后自动停止训练，并写入 `$MAGIC_DOCKER_WORKSPACE/logs/CONVERGED`。之后再执行训练命令会直接退出；确实要继续训练时加 `FORCE=1`。

**看进度：**

```bash
W=$MAGIC_DOCKER_WORKSPACE
grep -aoE "step:[0-9]+ - " $W/logs/train.log | tail -1
grep -aoE "timing_s/step:[0-9.]+" $W/logs/train.log | tail -3
cat $W/logs/early_stop.txt          # 验证分数的最好成绩、多久没进步
nvidia-smi
```

**正常的样子：**

- `timing_s/reward` 约 10 秒。
- 日志里会反复出现 `Error during checking: len(...) = 2 != ... = 1`，这是 MAGIC 自带的调试检查，只打印、不影响训练，可以忽略。
- 任何时刻常常只有 1–2 张卡在忙，这是 MAGIC 交替训练的设计（attacker 生成 → defender 生成 → judge 打分 → 更新其中一方）。

**要停下来找 Wentse 的情况：**

- 出现 `Traceback`、`out of memory`。
- 4 张卡连续 10 分钟都是 0%。
- 容器退出。

## 4. 跑完之后

把 `$MAGIC_DOCKER_WORKSPACE/ckpt/Game-separated/D-llama31abl-A-magicsft/` 下每个 `global_step_*/defender`（第 100、200、… 步，加上最后停下时的那一个）和 `$MAGIC_DOCKER_WORKSPACE/logs/`（含 `CONVERGED`、`early_stop.txt`）传给 Wentse，评测由我们来做。

## 不用 Docker 的做法

机器能直接建 Python 3.10 环境时，也可以不用 Docker：`MAGIC_HOME=<目录> bash docker/magic/setup.sh`，再 `MAGIC_HOME=<目录> CUDA_VISIBLE_DEVICES=0,1,2,3 bash docker/magic/run.sh`。

## 和 MAGIC 原设置的差异（论文里要写明）

| 项目 | MAGIC 原设置 | 这里 | 原因 |
|---|---|---|---|
| defender 底座 | Llama-3.1-8B-Instruct | abliterated 权重，template 和停止 token 用官方 Instruct 版 | 和 PSRO 同底座；只换权重 |
| GPU | 每个角色 4 张（共 8 张） | 每个角色 2 张（共 4 张） | 算力 |
| 训练步数 | 300（第 200 步早停） | 最多 600，第 300 步后按验证分数收敛提前停 | abliterated 底座起点没有拒答能力，给两倍步数；第 200、300 步的 checkpoint 也保留，可以和原设置对照 |
| reward 并发 | 1 | 16 | 只影响速度（judge 温度 0） |
| 代码 | 公开分支 | 补一行 `seed_prompt` 未定义的 bug（`docker/magic/seed_prompt.patch`，main 分支也有这个 bug） | 不补无法运行 |
| judge | Qwen3Guard | Qwen3Guard-Gen-8B | 论文没写大小；MAGIC 的 reward 代码只支持 Qwen3Guard 的输出格式 |
| seed | 未公开 | 框架默认 | |
