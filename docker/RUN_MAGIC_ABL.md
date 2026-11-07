# MAGIC baseline：在 abliterated Llama-3.1-8B 上重跑

**目的：** 用 MAGIC 官方代码，从和我们 PSRO 相同的底座（abliterated Llama-3.1-8B）出发训练，和 PSRO 做公平比较。

**需要：** 一台机器 4 张 H200（每张 141 GB），约 200 GB 磁盘，可以连 Hugging Face 和 GitHub。不用 Docker，也不用任何 token（模型都是公开的）。

## 1. 安装（只做一次，不需要 GPU）

```bash
git clone -b wentse/rebuttal-runbooks https://github.com/WentseChen/SafeSelfPlay.git
cd SafeSelfPlay
export MAGIC_HOME=$HOME/magic_abl      # 换成有 200 GB 空间的目录
bash docker/magic/setup.sh 2>&1 | tee magic_setup.log
```

最后要看到这三行，缺一行就停下来把 `magic_setup.log` 发给我：

```
ENV 2.6.0+cu124 0.8.5.post1 4.51.3 2.7.4.post1
BASE_OK .../llama31_abl_ittok
SETUP_DONE
```

## 2. 训练

```bash
export MAGIC_HOME=$HOME/magic_abl CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash docker/magic/run.sh > $MAGIC_HOME/logs/run_$(date +%m%d_%H%M).log 2>&1 &
```

- 启动约 4 分钟：先起 judge，再加载两个模型。
- 中断后用同一条命令重跑，会从最新的 checkpoint 接着跑（每 25 步存一次）。
- 共 300 步。我们在 4×H200 上量到每步 180–200 秒（回答上限 6144 时）；改成 2048 后预计 100–150 秒，300 步约 9–13 小时。

**看进度：**

```bash
grep -aoE "step:[0-9]+ - " $MAGIC_HOME/logs/train.log | tail -1
grep -aoE "timing_s/step:[0-9.]+" $MAGIC_HOME/logs/train.log | tail -3
nvidia-smi
```

**正常的样子：**

- `timing_s/reward` 约 10 秒。
- 日志里会反复出现 `Error during checking: len(...) = 2 != ... = 1`，这是 MAGIC 自带的调试检查，只打印、不影响训练，可以忽略。
- 任何时刻常常只有 1–2 张卡在忙，这是 MAGIC 交替训练的设计（attacker 生成 → defender 生成 → judge 打分 → 更新其中一方）。

**要停下来找我的情况：**

- 出现 `Traceback`、`out of memory`。
- 4 张卡连续 10 分钟都是 0%。
- 训练进程退出。

## 3. 跑完之后

把第 200 步和第 300 步的 checkpoint（`$MAGIC_HOME/ckpt/Game-separated/D-llama31abl-A-magicsft/global_step_{200,300}/defender`）传给我，评测由我来做。

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
