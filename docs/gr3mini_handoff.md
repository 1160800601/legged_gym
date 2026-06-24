# GR3 Mini 轨迹跟踪交接文档

## 当前状态

远端仓库已经包含：

- GR3 Mini 任务代码：`legged_gym/envs/gr3mini/`
- GR3 Mini 机器人资产：`resources/robots/fourier_gr3mini_v200/`
- 走路参考轨迹：`resources/traces/Stop_forward_Walk_001__A017_M_50hz.csv`
- 精选模型：`checkpoints/`
- 训练记录：`docs/gr3mini_walk_tracking_overfit.md`

`logs/` 被 `.gitignore` 忽略，不会上传完整训练日志和全部 checkpoint。精选 checkpoint 已复制到 `checkpoints/`，用于 fresh clone 后快速恢复播放。

## 环境要求

仓库本身不包含 Isaac Gym 和 `rsl_rl` 源码。运行前仍需要本机环境满足：

```text
conda env: legged_gym_py38
Python: 3.8
Isaac Gym installed and importable
rsl_rl installed
PyTorch + CUDA available
legged_gym installed with pip install -e .
```

当前机器上 `rsl_rl` 来自：

```text
/home/fftai/miniconda3/envs/legged_gym_py38/lib/python3.8/site-packages/rsl_rl
```

## 分支说明

| Branch | Purpose | Obs dim | Future refs | Randomization | PPO |
| --- | --- | ---: | ---: | --- | --- |
| `master` | 单轨迹 overfit baseline | 274 | 5 | off | `lr=1e-4`, `entropy=0.003` |
| `randomlize` | 轻量随机化继续 overfit | 274 | 5 | init noise, friction, base mass | `lr=1e-4`, `entropy=0.003` |
| `ablation` | PPO entropy 消融 | 274 | 5 | off | `lr=1e-4`, `entropy=0.02` |
| `future-steps-1` | 未来参考帧长度消融 | 146 | 1 | off | `lr=1e-4`, `entropy=0.003` |
| `future-steps-0` | 无未来参考帧消融 | 114 | 0 | off | `lr=1e-4`, `entropy=0.003` |

注意：`future-steps-1` 和 `future-steps-0` 改变了 observation 维度，不能和 274 维 checkpoint 混用。

## 精选 Checkpoints

| Branch | File | Original run | Iteration | Notes |
| --- | --- | --- | ---: | --- |
| `master` | `checkpoints/gr3mini_trace__master__future5__baseline__Jun23_11-43-52__model_7000.pt` | `Jun23_11-43-52_` | 7000 | 单轨迹 overfit baseline |
| `randomlize` | `checkpoints/gr3mini_trace__randomlize__future5__light_rand__Jun23_13-48-13__model_10000.pt` | `Jun23_13-48-13_` | 10000 | 轻量随机化 |
| `ablation` | `checkpoints/gr3mini_trace__ablation__future5__lr1e-4_entropy0p02__Jun23_16-26-42__model_6800.pt` | `Jun23_16-26-42_lr1e-4_entropy0p02` | 6800 | entropy 消融当前最新 |
| `future-steps-1` | `checkpoints/gr3mini_trace__future-steps-1__future1__baseline__Jun23_18-47-15__model_10000.pt` | `Jun23_18-47-15_future_steps_1` | 10000 | 未来 1 帧参考 |
| `future-steps-0` | `checkpoints/gr3mini_trace__future-steps-0__future0__baseline__Jun24_09-44-11__model_600.pt` | `Jun24_09-44-11_future_steps_0` | 600 | 无未来参考，当前只训练到 600 |

## 播放方式

`play.py` 默认从 `logs/gr3mini_trace/<load_run>/model_<checkpoint>.pt` 加载模型。fresh clone 后，如果 `logs/` 为空，需要先把 `checkpoints/` 中的模型复制回一个 logs 子目录。

### Master Baseline

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
git switch master

mkdir -p logs/gr3mini_trace/selected_master_future5
cp checkpoints/gr3mini_trace__master__future5__baseline__Jun23_11-43-52__model_7000.pt \
  logs/gr3mini_trace/selected_master_future5/model_7000.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_master_future5 \
  --checkpoint=7000 \
  --num_envs=1
```

### Light Randomization

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
git switch randomlize

mkdir -p logs/gr3mini_trace/selected_randomlize_future5
cp checkpoints/gr3mini_trace__randomlize__future5__light_rand__Jun23_13-48-13__model_10000.pt \
  logs/gr3mini_trace/selected_randomlize_future5/model_10000.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_randomlize_future5 \
  --checkpoint=10000 \
  --num_envs=1
```

### Entropy Ablation

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
git switch ablation

mkdir -p logs/gr3mini_trace/selected_ablation_entropy0p02
cp checkpoints/gr3mini_trace__ablation__future5__lr1e-4_entropy0p02__Jun23_16-26-42__model_6800.pt \
  logs/gr3mini_trace/selected_ablation_entropy0p02/model_6800.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_ablation_entropy0p02 \
  --checkpoint=6800 \
  --num_envs=1
```

### Future 1 Frame

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
git switch future-steps-1

mkdir -p logs/gr3mini_trace/selected_future_steps_1
cp checkpoints/gr3mini_trace__future-steps-1__future1__baseline__Jun23_18-47-15__model_10000.pt \
  logs/gr3mini_trace/selected_future_steps_1/model_10000.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_future_steps_1 \
  --checkpoint=10000 \
  --num_envs=1
```

### Future 0 Frames

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
git switch future-steps-0

mkdir -p logs/gr3mini_trace/selected_future_steps_0
cp checkpoints/gr3mini_trace__future-steps-0__future0__baseline__Jun24_09-44-11__model_600.pt \
  logs/gr3mini_trace/selected_future_steps_0/model_600.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_future_steps_0 \
  --checkpoint=600 \
  --num_envs=1
```

## 训练命令模板

从当前分支配置从 0 开始训练：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/train.py \
  --task=gr3mini_trace \
  --headless \
  --run_name=<experiment_name> \
  --max_iterations=10000
```

从已有 checkpoint 继续训练时，要求分支配置和 checkpoint 的 observation 维度一致：

```bash
conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/train.py \
  --task=gr3mini_trace \
  --headless \
  --resume \
  --load_run=<run_name> \
  --checkpoint=<iteration> \
  --max_iterations=<extra_iterations>
```

## 注意事项

- `logs/` 仍然是本地训练输出目录，不会上传。
- `checkpoints/` 只放精选模型，不放 TensorBoard event 文件。
- 不同 `future_steps` 对应不同网络输入维度，不能混用 checkpoint。
- `future-steps-0` 目前只有 `model_600.pt`，不是完整 10000 iteration 结果。
- 部分中途保存的 checkpoint 文件名包含迭代数，但内部 `iter` metadata 仍可能是 0；它们可以用于 play，但继续训练时优先使用完整结束时保存的 checkpoint。
- `randomlize` 是分支名原样保留，拼写不是 `randomize`。
