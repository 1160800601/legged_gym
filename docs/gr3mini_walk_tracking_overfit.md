# GR3 Mini 单轨迹走路跟踪器记录

## 结论

本次训练得到的是一个 **GR3 Mini 单轨迹过拟合走路跟踪器**。它只使用一条 50Hz 走路参考轨迹训练，目标是让机器人在 Isaac Gym 中跟踪该参考动作，不是泛化型 locomotion policy。

推荐直接调用的 task 和 model：

```text
task: gr3mini_trace
run:  Jun23_11-43-52_
model: logs/gr3mini_trace/Jun23_11-43-52_/model_7000.pt
```

## 数据

```text
trajectory: resources/traces/Stop_forward_Walk_001__A017_M_50hz.csv
frames: 5228
frequency: 50Hz
duration: 104.56s
```

当前只训练这一条轨迹：

```python
file_pattern = "Stop_forward_Walk_001__A017_M_50hz.csv"
frame_dt = 1.0 / 50.0
```

## 关键配置

环境：

```text
num_envs = 1024
num_observations = 274
num_actions = 25
episode_length_s = 6
terrain = plane
```

动作空间：

```text
action = reference_joint_pos + action_scale * policy_output
action_scale = 0.15
control_type = P
```

观测里包含当前状态、当前参考关节角，以及未来 5 帧参考：

```text
future_steps = 5
per future frame = 25 dof pos + 3 root pos delta + 4 root quat delta
```

随机化状态：

```text
root_pos_noise = [0.0, 0.0, 0.0]
dof_pos_noise = 0.0
dof_vel_noise = 0.0
randomize_friction = False
randomize_base_mass = False
push_robots = False
noise.add_noise = False
```

注意：`motion.random_start = True`，表示每个 episode 可以从同一条参考轨迹的随机帧开始，用来覆盖完整轨迹的不同片段；这不是 root 初始位置噪声，也不是摩擦力/质量等物理随机化。

## Reward

当前主要跟踪项：

```text
tracking_joint_pos = 6.0
tracking_joint_vel = 0.5
tracking_root_pos = 1.0
tracking_root_orientation = 1.0
tracking_root_lin_vel = 1.0
alive = 0.2
```

主要正则/惩罚项：

```text
termination = -10.0
collision = -0.2
action_rate = -0.02
torques = -2e-6
dof_acc = -1e-7
```

姿态跟踪使用 quaternion angle error，包含 yaw/pitch/roll：

```python
angle = 2 * acos(clamp(abs(dot(base_quat, ref_root_quat)), 0, 1))
reward = exp(-(angle * angle) / 0.2)
```

## 直接播放

只看一个机器人，并关闭 reset：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=Jun23_11-43-52_ \
  --checkpoint=7000 \
  --num_envs=1 \
  --play_disable_reset
```

如果需要看 termination 统计：

```bash
conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=Jun23_11-43-52_ \
  --checkpoint=7000 \
  --num_envs=1 \
  --play_print_termination
```

## 继续训练

从当前 `model_7000.pt` 继续训练 3000 个 iteration，到约 10000：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/train.py \
  --task=gr3mini_trace \
  --headless \
  --resume \
  --load_run=Jun23_11-43-52_ \
  --checkpoint=7000 \
  --max_iterations=3000
```

从头训练：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/train.py \
  --task=gr3mini_trace \
  --headless \
  --max_iterations=10000
```

## TensorBoard

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

conda run --no-capture-output -n legged_gym_py38 tensorboard \
  --logdir logs/gr3mini_trace \
  --host 0.0.0.0 \
  --port 6006
```

重点看：

```text
Episode/metric_joint_pos_rmse
Episode/metric_root_pos_err
Episode/metric_root_orientation_err
Episode/metric_root_lin_vel_err
Episode/term_*
Train/mean_reward
Train/mean_episode_length
```

## 限制

- 当前 policy 只针对 `Stop_forward_Walk_001__A017_M_50hz.csv` 这一条轨迹。
- 没有摩擦力、质量、push、观测噪声等 domain randomization。
- 没有初始 root 位置、关节位置、关节速度噪声。
- episode 长度是 6s，训练通过随机参考帧起点覆盖长轨迹。
- obs 维度是 274，旧的 114 维 checkpoint 不能和当前代码混用。
