# MAGIC baseline：在 abliterated Llama-3.1-8B 上重跑

**目的：** 用 MAGIC 官方代码（[AI45Lab/MAGIC](https://github.com/AI45Lab/MAGIC)，原 BattleWen/MAGIC），从和我们 PSRO 相同的底座（abliterated Llama-3.1-8B）出发训练，和 PSRO 做公平比较。

**需要：** Linux x86_64，4×H200 141GB，NVIDIA 驱动、Docker、NVIDIA Container Toolkit（和跑 FSP、PSRO 的要求相同），约 **500 GB** 磁盘（模型约 50 GB；每个 checkpoint 含两个模型和 optimizer，约 110 GB，会保留最新两个加第 200、300 步），能访问 GitHub、Hugging Face。代码和模型都是公开的，不需要任何 token。

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
- 共 300 步。我们在 4×H200 上量到每步 180–200 秒（回答上限 6144 时）；改成 2048 后预计 100–150 秒，300 步约 9–13 小时。

**看进度：**

```bash
W=$MAGIC_DOCKER_WORKSPACE
grep -aoE "step:[0-9]+ - " $W/logs/train.log | tail -1
grep -aoE "timing_s/step:[0-9.]+" $W/logs/train.log | tail -3
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

把第 200 步和第 300 步的 checkpoint（`$MAGIC_DOCKER_WORKSPACE/ckpt/Game-separated/D-llama31abl-A-magicsft/global_step_{200,300}/defender`）传给 Wentse，评测由我们来做。

## 不用 Docker 的做法

机器能直接建 Python 3.10 环境时，也可以不用 Docker：`MAGIC_HOME=<目录> bash docker/magic/setup.sh`，再 `MAGIC_HOME=<目录> CUDA_VISIBLE_DEVICES=0,1,2,3 bash docker/magic/run.sh`。

## 和 MAGIC 原设置的差异（论文里要写明）

| 项目 | MAGIC 原设置 | 这里 | 原因 |
|---|---|---|---|
| defender 底座 | Llama-3.1-8B-Instruct | abliterated 权重，template 和停止 token 用官方 Instruct 版 | 和 PSRO 同底座；只换权重 |
| GPU | 每个角色 4 张（共 8 张） | 每个角色 2 张（共 4 张） | 算力 |
| 回答长度上限 | 6144 | 2048 | 算力；只影响约 0.6% 的 defender 回答，截断的回答 reward 为 0（MAGIC 自带机制） |
| reward 并发 | 1 | 16 | 只影响速度（judge 温度 0） |
| 代码 | 公开分支 | 补一行 `seed_prompt` 未定义的 bug（`docker/magic/seed_prompt.patch`，main 分支也有这个 bug） | 不补无法运行 |
| judge | Qwen3Guard | Qwen3Guard-Gen-8B | 论文没写大小；MAGIC 的 reward 代码只支持 Qwen3Guard 的输出格式 |
| seed | 未公开 | 框架默认 | |
