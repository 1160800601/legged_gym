# legged_gym 学习指南

本文档面向第一次阅读本项目的人，目标是先理解项目如何跑起来，再理解环境、配置、奖励、地形和机器人差异，最后能做小范围实验修改。

## 1. 项目定位

本项目是基于 NVIDIA Isaac Gym 的腿式机器人强化学习环境库，主要用于训练 ANYmal、A1、Cassie 等机器人在平地或复杂地形上的行走策略。

本仓库负责：

- Isaac Gym 仿真环境封装
- 机器人资产、URDF 和地形生成
- 观测构造
- 奖励函数
- reset、termination、curriculum、domain randomization
- 任务注册和配置管理

PPO 算法本身来自外部依赖 `rsl_rl`，不是本仓库实现的重点。

## 2. 推荐先读的文件

按这个顺序读效率最高：

1. `README.md`
   - 看安装要求、运行命令、任务注册方式。
   - 注意本项目依赖旧版 Isaac Gym 栈：Python 3.6-3.8、PyTorch 1.10、Isaac Gym Preview 3、`rsl_rl v1.0.2`。

2. `legged_gym/scripts/train.py`
   - 训练入口。
   - 调用 `task_registry.make_env()` 创建环境。
   - 调用 `task_registry.make_alg_runner()` 创建 PPO runner。
   - 最后执行 `ppo_runner.learn()`。

3. `legged_gym/scripts/play.py`
   - 策略回放入口。
   - 加载 checkpoint，构造 inference policy。
   - 可导出 TorchScript policy。
   - 文件里有录帧选项，但默认关闭。除非明确需要，不建议自动渲染或录制视频。

4. `legged_gym/envs/__init__.py`
   - 所有内置任务在这里注册：
     - `anymal_c_rough`
     - `anymal_c_flat`
     - `anymal_b`
     - `a1`
     - `cassie`

5. `legged_gym/utils/task_registry.py`
   - 理解 task 名字如何映射到环境类和配置类。
   - `make_env()` 负责创建 Isaac Gym 环境。
   - `make_alg_runner()` 负责创建 `rsl_rl.runners.OnPolicyRunner`。

6. `legged_gym/envs/base/legged_robot_config.py`
   - 最核心的配置文件。
   - 包含环境数量、观测维度、地形、命令、初始姿态、控制器、资产、随机化、奖励、归一化、噪声、仿真参数和 PPO 参数。

7. `legged_gym/envs/base/legged_robot.py`
   - 最核心的环境实现。
   - 重点读：
     - `step()`
     - `post_physics_step()`
     - `check_termination()`
     - `reset_idx()`
     - `compute_reward()`
     - `compute_observations()`
     - `_compute_torques()`
     - `_prepare_reward_function()`
     - `_create_envs()`

8. `legged_gym/utils/terrain.py`
   - 地形生成。
   - 包含 curriculum terrain、random terrain、trimesh/heightfield 数据生成。

9. 各机器人配置和子类：
   - `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`
   - `legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py`
   - `legged_gym/envs/anymal_c/anymal.py`
   - `legged_gym/envs/a1/a1_config.py`
   - `legged_gym/envs/cassie/cassie_config.py`
   - `legged_gym/envs/cassie/cassie.py`

## 3. 主调用链

训练主链路：

```text
train.py
  -> get_args()
  -> task_registry.make_env(task)
       -> 找到注册的 EnvClass 和 EnvConfig
       -> 解析 sim 参数
       -> 创建 LeggedRobot/Anymal/Cassie 环境
  -> task_registry.make_alg_runner(env, task)
       -> 读取 PPO 配置
       -> 创建 rsl_rl OnPolicyRunner
  -> ppo_runner.learn()
```

环境一步的主链路：

```text
LeggedRobot.step(actions)
  -> 裁剪 action
  -> 根据 control.decimation 重复执行物理子步
  -> _compute_torques(actions)
  -> Isaac Gym simulate()
  -> post_physics_step()
       -> 刷新 root/contact tensor
       -> 计算 base 速度、重力投影
       -> 重采样 command、测量地形高度、随机 push
       -> check_termination()
       -> compute_reward()
       -> reset_idx()
       -> compute_observations()
```

## 4. 配置系统怎么理解

配置类继承自 `BaseConfig`。配置使用“类中嵌套类”的形式，例如：

```python
class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
    class terrain:
        mesh_type = "trimesh"
```

`BaseConfig` 初始化时会递归实例化这些嵌套类，所以运行时可以用：

```python
cfg.env.num_envs
cfg.terrain.mesh_type
```

命令行参数只覆盖少量配置，例如：

- `--task`
- `--num_envs`
- `--seed`
- `--max_iterations`
- `--resume`
- `--experiment_name`
- `--run_name`
- `--load_run`
- `--checkpoint`

大部分实验修改仍然应该在 config 文件里完成。

## 5. 观测结构

默认观测在 `LeggedRobot.compute_observations()` 里构造：

```text
base_lin_vel              3
base_ang_vel              3
projected_gravity         3
commands[:3]              3
dof_pos - default_dof_pos 12
dof_vel                  12
last actions             12
height measurements      optional
```

无高度测量时是 48 维。

例如 `anymal_c_flat`：

- `num_observations = 48`
- `terrain.mesh_type = "plane"`
- `terrain.measure_heights = False`

粗糙地形默认会加入高度采样点：

- `measured_points_x` 长度 17
- `measured_points_y` 长度 11
- 高度点数量 187
- 总观测维度 48 + 187 = 235

## 6. 控制器逻辑

基础控制在 `LeggedRobot._compute_torques()` 中。

支持三种控制模式：

- `P`: action 表示目标关节位置偏移，用 PD 控制转换成 torque。
- `V`: action 表示目标关节速度。
- `T`: action 直接表示 torque。

常见配置项：

- `control.stiffness`
- `control.damping`
- `control.action_scale`
- `control.decimation`

`Anymal` 子类额外支持 actuator network：

- 文件：`legged_gym/envs/anymal_c/anymal.py`
- 配置：`control.use_actuator_network = True`
- 网络：`resources/actuator_nets/anydrive_v3_lstm.pt`

## 7. 奖励函数机制

奖励配置在：

```text
LeggedRobotCfg.rewards.scales
```

核心机制在：

```text
LeggedRobot._prepare_reward_function()
```

规则是：

- scale 为 0 的 reward 会被移除。
- scale 非 0 的 reward 会查找同名函数。
- `tracking_lin_vel` 会对应 `_reward_tracking_lin_vel()`。
- 每个 reward scale 会乘以 `dt`。
- `termination` 奖励单独处理。

新增奖励时通常做两件事：

1. 在环境类里新增函数：

```python
def _reward_my_term(self):
    ...
```

2. 在配置里加非零 scale：

```python
class rewards(...):
    class scales(...):
        my_term = -0.1
```

## 8. 地形和 curriculum

地形生成在 `legged_gym/utils/terrain.py`。

主要模式：

- `plane`
- `heightfield`
- `trimesh`
- `none`

粗糙地形通常用 `trimesh`。

如果 `terrain.curriculum = True`：

- 地形被组织成 rows 和 cols。
- row 表示难度。
- col 表示地形类型。
- 机器人走得足够远会进入更难 row。
- 表现差会降到更简单 row。

地形 origin 存在 `terrain.env_origins`，环境 reset 时会把机器人放到对应地形块中心。

## 9. 机器人任务差异

### ANYmal C rough

文件：

- `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`
- `legged_gym/envs/anymal_c/anymal.py`

特点：

- `trimesh` 地形。
- 12 个 action。
- 使用 ANYmal C URDF。
- 默认启用 actuator network。
- 启用 base mass randomization。

### ANYmal C flat

文件：

- `legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py`

特点：

- 平地。
- 关闭高度测量。
- 观测维度 48。
- PPO 网络更小。
- 默认最大训练迭代数 300。

### A1

文件：

- `legged_gym/envs/a1/a1_config.py`

特点：

- 直接使用基础 `LeggedRobot`。
- PD 控制。
- A1 URDF。
- 增加 `dof_pos_limits` 惩罚。

### Cassie

文件：

- `legged_gym/envs/cassie/cassie_config.py`
- `legged_gym/envs/cassie/cassie.py`

特点：

- 12 个 action。
- 169 维观测。
- `only_positive_rewards = False`。
- 额外实现 `_reward_no_fly()`。

## 10. 推荐运行命令

最小环境 sanity check：

```bash
python legged_gym/tests/test_env.py --task=anymal_c_flat --num_envs=2 --headless
```

小规模训练：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --headless --num_envs=128 --max_iterations=10
```

完整平地训练：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --headless
```

回放已训练策略：

```bash
python legged_gym/scripts/play.py --task=anymal_c_flat
```

CPU 尝试：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --sim_device=cpu --rl_device=cpu --headless --num_envs=16 --max_iterations=1
```

## 11. 推荐练习

1. 对比 `anymal_c_flat` 和 `anymal_c_rough` 配置，解释为什么一个是 48 维观测，一个是 235 维观测。

2. 把 `anymal_c_flat` 里的 `feet_air_time` scale 改小，跑少量 iteration，观察 reward 曲线变化。

3. 新增一个 reward，例如惩罚 yaw 角速度或惩罚过大的横向速度。

4. 新增一个 task：
   - 复制一个 config。
   - 修改 `experiment_name`。
   - 修改地形或 reward。
   - 在 `legged_gym/envs/__init__.py` 中注册。

5. 尝试关闭 domain randomization，对比策略是否更容易训练但泛化更差。

## 12. 常见坑

- Isaac Gym 对 Python、CUDA、PyTorch 版本很敏感。
- README 推荐 Python 3.8 和 PyTorch 1.10 + CUDA 11.3。
- `rsl_rl` 推荐使用 `v1.0.2`。
- 现代 PyTorch 或 Python 3.10/3.11 环境很可能需要改依赖或迁移到 Isaac Lab。
- GPU trimesh 地形下 `net_contact_force_tensor` 有已知不可靠问题，README 中给了 force sensor workaround。
- 不要一开始就用 4096 env 训练调试。先用 `--num_envs=2` 或 `--num_envs=16` 做 sanity check。

## 13. unitracker conda 环境检查

当前检查结果：

```text
conda 环境: unitracker
Python: 3.11.15
torch: 2.7.0
isaacgym: 未安装
rsl_rl: 可导入，来自 rsl-rl-lib 3.1.2
isaaclab: 2.3.2
isaacsim-core: 5.1.0.0
```

结论：`unitracker` 环境不能直接运行本仓库。

直接失败点：

```text
ModuleNotFoundError: No module named 'isaacgym'
```

失败发生在：

```text
legged_gym/scripts/train.py
legged_gym/envs/base/legged_robot.py
```

原因：

- 本仓库是 Isaac Gym Preview 3 时代的项目，代码里直接 `import isaacgym`。
- `unitracker` 环境装的是 Isaac Lab / Isaac Sim 新栈，不提供本仓库需要的 `isaacgym` Python 包。
- Python 3.11 和 PyTorch 2.7 也偏离 README 推荐组合。
- `rsl_rl` 虽然能导入，但版本是 3.1.2，不是 README 推荐的 `v1.0.2`。

建议新建独立环境运行本仓库，不要直接复用 `unitracker`：

```bash
conda create -n legged_gym_py38 python=3.8
conda activate legged_gym_py38
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

然后安装 Isaac Gym Preview 3：

```bash
cd /path/to/isaacgym/python
pip install -e .
```

安装 `rsl_rl` 推荐版本：

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
pip install -e .
```

最后安装本项目：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
pip install -e .
```

如果目标是复用 `unitracker` 里的 Isaac Lab / Isaac Sim，则方向不是直接跑本仓库，而是迁移到 Isaac Lab 版本的 locomotion 环境。

## 14. legged_gym_py38 环境状态

已创建独立环境：

```bash
conda activate legged_gym_py38
```

已安装：

```text
Python: 3.8.20
torch: 2.0.1+cu118
torchvision: 0.15.2+cu118
torchaudio: 2.0.2+cu118
rsl_rl: 1.0.2
numpy: 1.23.5
scipy: 1.10.1
matplotlib: 3.7.5
tensorboard: 2.14.0
setuptools: 59.5.0
isaacgym: installed from /home/fftai/mjq/nvidia/isaacgym/python
legged_gym: editable install from this repository
```

README 推荐的 `torch 1.10.0+cu113` 能安装，但在本机 RTX 4090 上实际训练会触发：

```text
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

原因是 RTX 4090 属于 Ada 架构，旧的 CUDA 11.3/PyTorch 1.10 不识别该 GPU 架构。因此当前环境已改用 `torch 2.0.1+cu118`。

Isaac Gym 安装后曾出现：

```text
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

已通过在 Isaac Gym binding 目录中添加软链接修复：

```bash
ln -s /home/fftai/miniconda3/envs/legged_gym_py38/lib/libpython3.8.so.1.0 \
  /home/fftai/mjq/nvidia/isaacgym/python/isaacgym/_bindings/linux-x86_64/libpython3.8.so.1.0
```

当前验证通过：

```bash
cd /home/fftai/mjq/legged_gym/legged_gym
python legged_gym/scripts/train.py --task=anymal_c_flat --headless --num_envs=2 --max_iterations=1
```

该命令已成功完成 1 个 PPO iteration。

另外，GPU 可见性依赖运行权限。沙盒内曾出现 `nvidia-smi` 无法与 NVIDIA driver 通信、PyTorch 报告 CUDA 不可用的情况；使用非沙箱权限检查时，GPU 正常可见：

```text
GPU: NVIDIA GeForce RTX 4090
NVIDIA Driver: 580.159.03
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
```

因此实际 GPU 训练/验证命令需要在能访问 GPU 设备的权限环境中运行。
