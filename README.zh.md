# 面向腿式机器人的 Isaac Gym 环境 #
本仓库提供了使用 NVIDIA Isaac Gym 训练 ANYmal 以及其他机器人在崎岖地形上行走的环境。
它包含 sim-to-real 迁移所需的全部组件：执行器网络、摩擦和质量随机化、带噪声观测，以及训练过程中的随机推搡。  

**维护者**: Nikita Rudin  
**单位**: ETH Zurich, Robotic Systems Lab  
**联系方式**: rudinn@ethz.ch  

---

### :bell: 公告 (09.01.2024) ###

随着 NVIDIA 从 Isaac Gym 转向 Isaac Sim，我们已将本工作的所有环境迁移到 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)。完成迁移后，本仓库将只获得有限的更新和支持。我们鼓励所有用户在自己的应用中迁移到新的框架。

关于本工作中 locomotion 相关任务在 Isaac Lab 中的信息，可在[这里](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#locomotion)查看。

---

### 有用链接 ###

项目网站: https://leggedrobotics.github.io/legged_gym/   
论文: https://arxiv.org/abs/2109.11978

### 安装 ###
1. 创建一个新的 Python 虚拟环境，使用 Python 3.6、3.7 或 3.8（推荐 3.8）。
2. 安装支持 cuda-11.3 的 pytorch 1.10：
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. 安装 Isaac Gym
   - 从 https://developer.nvidia.com/isaac-gym 下载并安装 Isaac Gym Preview 3（Preview 2 不可用！）。
   - `cd isaacgym/python && pip install -e .`
   - 尝试运行一个示例：`cd examples && python 1080_balls_of_solitude.py`
   - 如果需要排查问题，请查看文档：`isaacgym/docs/index.html`
4. 安装 rsl_rl（PPO 实现）
   - 克隆 https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 
5. 安装 legged_gym
    - 克隆本仓库
   - `cd legged_gym && pip install -e .`

### 代码结构 ###
1. 每个环境都由一个环境文件（`legged_robot.py`）和一个配置文件（`legged_robot_config.py`）定义。配置文件包含两个类：一个包含所有环境参数（`LeggedRobotCfg`），另一个包含训练参数（`LeggedRobotCfgPPo`）。  
2. 环境类和配置类都使用继承。  
3. 在 `cfg` 中指定的每个非零 reward scale，都会把对应名称的函数加入到奖励项列表中；这些奖励项求和后得到总奖励。  
4. 任务必须使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册。本仓库在 `envs/__init__.py` 中完成注册，也可以从仓库外部注册。  

### 使用 ###
1. 训练：  
  ```python legged_gym/scripts/train.py --task=anymal_c_flat```
    - 如果要在 CPU 上运行，添加以下参数：`--sim_device=cpu`、`--rl_device=cpu`（仿真在 CPU、强化学习在 GPU 上也是可行的）。
    - 如果要无界面运行（不渲染），添加 `--headless`。
    - **重要**：为了提升性能，训练开始后按 `v` 停止渲染。之后也可以再次启用渲染来检查训练进度。
    - 训练好的策略会保存到 `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。其中 `<experiment_name>` 和 `<run_name>` 在训练配置中定义。
    - 以下命令行参数会覆盖配置文件中的值：
     - --task TASK：任务名称。
     - --resume：从 checkpoint 恢复训练。
     - --experiment_name EXPERIMENT_NAME：要运行或加载的实验名称。
     - --run_name RUN_NAME：run 名称。
     - --load_run LOAD_RUN：`resume=True` 时要加载的 run 名称。如果为 -1，则加载最后一个 run。
     - --checkpoint CHECKPOINT：已保存模型的 checkpoint 编号。如果为 -1，则加载最后一个 checkpoint。
     - --num_envs NUM_ENVS：要创建的环境数量。
     - --seed SEED：随机种子。
     - --max_iterations MAX_ITERATIONS：最大训练迭代次数。
2. 回放训练好的策略：  
```python legged_gym/scripts/play.py --task=anymal_c_flat```
    - 默认情况下，会加载实验文件夹中最后一个 run 的最后一个模型。
    - 可以通过在训练配置中设置 `load_run` 和 `checkpoint` 来选择其他 run 或模型迭代。

### 添加新环境 ###
基础环境 `legged_robot` 实现了一个崎岖地形 locomotion 任务。对应的 cfg 没有指定机器人资产（URDF/MJCF），也没有 reward scale。 

1. 在 `envs/` 中添加一个新文件夹，并加入继承自现有环境配置的 `'<your_env>_config.py`。  
2. 如果要添加新机器人：
    - 将对应资产添加到 `resources/`。
    - 在 `cfg` 中设置资产路径，定义 body 名称、默认关节位置和 PD 增益。指定所需的 `train_cfg` 和环境名称（Python 类）。
    - 在 `train_cfg` 中设置 `experiment_name` 和 `run_name`。
3. 如有需要，在 `<your_env>.py` 中实现你的环境，继承自现有环境，覆写需要的函数，或者添加新的 reward 函数。
4. 在 `isaacgym_anymal/envs/__init__.py` 中注册你的环境。
5. 根据需要修改或调节 `cfg`、`cfg_train` 中的其他参数。要移除某个奖励项，将它的 scale 设置为零即可。不要修改其他环境的参数！


### 故障排查 ###
1. 如果遇到以下错误：`ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`，执行：`sudo apt install libpython3.8`。也可能需要执行 `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`（conda 用户）。请将 `/path/to/` 替换为对应路径。

### 已知问题 ###
1. 在 GPU 上使用 triangle mesh 地形进行仿真时，`net_contact_force_tensor` 报告的接触力不可靠。一种 workaround 是使用力传感器，但力会通过连续 body 的传感器传播，导致不理想的行为。不过，对于腿式机器人，可以只在脚部/末端执行器添加传感器，并得到期望结果。使用力传感器时，请确保通过 `sensor_options.enable_forward_dynamics_forces` 从报告的力中排除重力。示例：
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # 例如重力
        sensor_options.enable_constraint_solver_forces = True # 例如接触力
        sensor_options.use_world_frame = True # 在世界坐标系中报告力（更容易获取竖直分量）
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
