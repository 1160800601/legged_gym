# GR3 mini 轨迹跟踪任务配置说明

本文档说明新增任务 `gr3mini_trace` 的配置方法。任务入口配置位于：

```text
legged_gym/envs/gr3mini/gr3mini_trace_config.py
```

环境逻辑位于：

```text
legged_gym/envs/gr3mini/gr3mini_trace.py
```

任务注册名：

```text
gr3mini_trace
```

最小验证命令：

```bash
conda activate legged_gym_py38
cd /home/fftai/mjq/legged_gym/legged_gym
python legged_gym/scripts/train.py --task=gr3mini_trace --headless --num_envs=2 --max_iterations=1
```

## 1. 资产配置

```python
class asset:
    file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/fourier_gr3mini_v200/urdf/gr3mini.urdf"
    name = "fourier_gr3mini_v200"
    foot_name = "foot_roll_link"
    penalize_contacts_on = ["thigh", "shank", "upper_arm", "lower_arm", "hand"]
    terminate_after_contacts_on = ["base_link", "torso_link"]
    self_collisions = 1
```

参数含义：

- `file`: 机器人模型路径。当前使用修正后的 URDF：`resources/robots/fourier_gr3mini_v200/urdf/gr3mini.urdf`。
- `foot_name`: 用于从 body 名称里筛选脚部 link。当前匹配 `left_foot_roll_link` 和 `right_foot_roll_link`。
- `penalize_contacts_on`: 这些 body 接触地面时扣分，但不一定终止 episode。
- `terminate_after_contacts_on`: 这些 body 接触地面时终止 episode。初期建议只放核心躯干，避免训练刚开始过早 reset。
- `self_collisions = 1`: 关闭自碰撞。这个模型有不少碰撞几何，打开自碰撞容易产生内部接触。

调参建议：

- 机器人还没站稳时，不要把 `thigh`、`shank`、`hand` 全放进终止列表，先放进扣分列表。
- 如果 episode 每 1-2 步就 reset，优先检查 `terminate_after_contacts_on` 是否过严。

## 2. 轨迹配置

```python
class motion:
    trace_dir = "{LEGGED_GYM_ROOT_DIR}/resources/traces"
    file_pattern = "*.csv"
    frame_dt = 1.0 / 60.0
    random_start = True
    root_pos_noise = [0.02, 0.02, 0.01]
    dof_pos_noise = 0.02
    dof_vel_noise = 0.05
```

参数含义：

- `trace_dir`: 轨迹文件目录。
- `file_pattern`: 读取哪些 CSV。当前会读 `resources/traces/*.csv`。
- `frame_dt`: 轨迹帧间隔。若轨迹是 60 FPS，用 `1/60`；若是 30 FPS，用 `1/30`。
- `random_start`: reset 时是否随机选轨迹中的起始帧。
- `root_pos_noise`: reset 时给 root 位置加的小扰动。
- `dof_pos_noise`: reset 时给关节角加的小扰动，单位 rad。
- `dof_vel_noise`: reset 时给关节速度加的小扰动。

CSV 当前字段格式：

```text
Frame
root_translateX/Y/Z
root_quatW/X/Y/Z
<joint_name>_dof
```

注意：

- CSV 里的 quaternion 是 `WXYZ`，环境里会转换成 Isaac Gym 使用的 `XYZW`。
- CSV 里的关节列名要和 URDF 的 joint 名对应，例如 `left_hip_pitch_joint_dof` 对应 `left_hip_pitch_joint`。

## 3. 环境维度

```python
class env:
    num_envs = 1024
    num_observations = 114
    num_actions = 25
    episode_length_s = 6
```

参数含义：

- `num_envs`: 并行环境数。调试用 2-16，正式训练逐步增大到 512/1024。
- `num_observations`: 观测维度。当前由 `GR3MiniTrace.compute_observations()` 决定。
- `num_actions`: 动作维度，等于 25 个可控关节。
- `episode_length_s`: 单个 episode 的最大长度。

当前 observation 结构：

```text
base_lin_vel                  3
base_ang_vel                  3
projected_gravity             3
phase_sin_cos                 2
root_pos_error                3
dof_pos - ref_dof_pos        25
dof_vel - ref_dof_vel        25
ref_dof_pos                  25
last_action                  25
总计                         114
```

修改 observation 时必须同步修改 `num_observations`。

## 4. 控制参数

```python
class control:
    control_type = "P"
    action_scale = 0.25
    decimation = 4
    stiffness = {...}
    damping = {...}
```

当前动作含义：

```text
PD target = reference_joint_pos + action_scale * action
```

参数含义：

- `control_type = "P"`: 使用位置 PD 控制。
- `action_scale`: policy 输出的残差幅度。太大容易乱动，太小跟踪能力不足。
- `decimation`: policy step 和 physics step 的比例。当前仿真 dt 为 0.005，`decimation=4`，所以 policy dt 为 0.02 秒。
- `stiffness`: 各关节位置刚度。
- `damping`: 各关节速度阻尼。

调参建议：

- 腿部关节刚度通常高于手臂和头部。
- 脚踝过硬容易抖，过软容易塌。
- 如果动作剧烈抖动，先降低 `action_scale` 或提高 `action_rate` 惩罚。

## 5. 奖励参数

```python
class rewards:
    only_positive_rewards = False
    class scales:
        tracking_joint_pos = 6.0
        tracking_joint_vel = 0.5
        tracking_root_pos = 1.0
        tracking_root_orientation = 1.0
        tracking_root_lin_vel = 1.0
        alive = 0.2
        action_rate = -0.01
        dof_acc = -1.0e-7
        torques = -2.0e-6
        collision = -0.2
        termination = -10.0
```

主要奖励：

- `tracking_joint_pos`: 跟踪参考关节角，通常是最重要的项。
- `tracking_joint_vel`: 跟踪参考关节速度。
- `tracking_root_pos`: 跟踪 root 位置。
- `tracking_root_orientation`: 跟踪 root 姿态，这里用重力投影差近似。
- `tracking_root_lin_vel`: 跟踪 root 水平速度。
- `alive`: 存活奖励，避免纯负奖励导致学习困难。

主要惩罚：

- `action_rate`: 惩罚动作变化，抑制抖动。
- `dof_acc`: 惩罚关节加速度。
- `torques`: 惩罚力矩消耗。
- `collision`: 惩罚非脚部 body 接触。
- `termination`: 摔倒终止惩罚。

调参顺序建议：

1. 先让 episode 不要每步 reset。
2. 先调 `tracking_joint_pos` 和 PD gains，让姿态大致跟得住。
3. 再逐步加大 `tracking_root_pos`、`tracking_root_orientation`。
4. 最后调平滑项：`action_rate`、`dof_acc`、`torques`。

## 6. PPO 参数

```python
class GR3MiniTraceCfgPPO:
    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
    class algorithm:
        learning_rate = 3.0e-4
        entropy_coef = 0.01
    class runner:
        experiment_name = "gr3mini_trace"
        max_iterations = 2000
```

参数含义：

- `actor_hidden_dims` / `critic_hidden_dims`: policy/value 网络大小。
- `learning_rate`: PPO 学习率。跟踪任务建议从 `3e-4` 或 `1e-4` 开始。
- `entropy_coef`: 探索强度。太大会导致动作发散，太小可能早早收敛到差策略。
- `max_iterations`: 最大训练迭代次数。
- `experiment_name`: 日志保存目录名。

训练命令示例：

```bash
python legged_gym/scripts/train.py --task=gr3mini_trace --headless --num_envs=512 --max_iterations=2000
```

## 7. 后续需要重点验证

- 轨迹的 `frame_dt` 是否真实为 60 FPS。
- CSV root 坐标系是否和 Isaac Gym/URDF 坐标系一致。
- root 高度是否需要整体偏移。
- 关节方向是否和 URDF 中的 joint axis 一致。
- 是否需要把脚部接触状态也加入 reward。
- 是否需要加入未来 1-3 帧 reference 作为 observation。
