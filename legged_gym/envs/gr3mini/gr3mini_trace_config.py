# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GR3MiniTraceCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 114
        num_actions = 25
        episode_length_s = 6

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 3

    class motion:
        trace_dir = "{LEGGED_GYM_ROOT_DIR}/resources/traces"
        file_pattern = "Stop_forward_Walk_001__A017_M_50hz.csv"
        frame_dt = 1.0 / 60.0
        random_start = True
        root_pos_noise = [0.0, 0.0, 0.0]
        dof_pos_noise = 0.0
        dof_vel_noise = 0.0

    class termination:
        disable = False
        contact_force = 1.0
        min_base_height = 0.35
        max_projected_gravity_xy = 0.75
        max_root_pos_error = 0.8
        max_joint_pos_rmse = 0.9
        foot_contact_force = 1.0
        max_foot_slip_speed = 2.0
        max_base_lin_vel = 5.0
        max_base_ang_vel = 10.0
        max_dof_vel = 50.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.62]
        default_joint_angles = {
            "waist_yaw_joint": -0.0745061189,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": -0.174267486,
            "left_hip_pitch_joint": -0.297578007,
            "left_hip_roll_joint": 0.0388916917,
            "left_hip_yaw_joint": 0.00252577337,
            "left_knee_pitch_joint": 0.496781409,
            "left_ankle_pitch_joint": -0.292426944,
            "left_ankle_roll_joint": -0.00992245506,
            "right_hip_pitch_joint": -0.219672173,
            "right_hip_roll_joint": -0.048276525,
            "right_hip_yaw_joint": -0.237582594,
            "right_knee_pitch_joint": 0.448512465,
            "right_ankle_pitch_joint": -0.308998317,
            "right_ankle_roll_joint": 0.0143760908,
            "left_shoulder_pitch_joint": 0.0443761833,
            "left_shoulder_roll_joint": 0.173453301,
            "left_shoulder_yaw_joint": -0.356751353,
            "left_elbow_pitch_joint": -0.579352856,
            "left_wrist_yaw_joint": -0.196766719,
            "right_shoulder_pitch_joint": 0.0684600174,
            "right_shoulder_roll_joint": -0.124730654,
            "right_shoulder_yaw_joint": 0.134434775,
            "right_elbow_pitch_joint": -0.557902098,
            "right_wrist_yaw_joint": 0.180025548,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_pitch": 120.0,
            "hip_roll": 80.0,
            "hip_yaw": 80.0,
            "knee_pitch": 120.0,
            "ankle_pitch": 35.0,
            "ankle_roll": 35.0,
            "waist_yaw": 80.0,
            "shoulder_pitch": 30.0,
            "shoulder_roll": 20.0,
            "shoulder_yaw": 20.0,
            "elbow_pitch": 20.0,
            "wrist_yaw": 8.0,
            "head": 8.0,
        }
        damping = {
            "hip_pitch": 4.0,
            "hip_roll": 3.0,
            "hip_yaw": 3.0,
            "knee_pitch": 4.0,
            "ankle_pitch": 1.2,
            "ankle_roll": 1.2,
            "waist_yaw": 2.0,
            "shoulder_pitch": 1.0,
            "shoulder_roll": 0.8,
            "shoulder_yaw": 0.8,
            "elbow_pitch": 0.8,
            "wrist_yaw": 0.3,
            "head": 0.3,
        }
        # Action is a residual added on top of the reference joint angle.
        action_scale = 0.15
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/fourier_gr3mini_v200/urdf/gr3mini.urdf"
        name = "fourier_gr3mini_v200"
        foot_name = "foot_roll_link"
        penalize_contacts_on = ["thigh", "shank", "upper_arm", "lower_arm", "hand"]
        terminate_after_contacts_on = ["base_link", "torso_link"]
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.61
        max_contact_force = 700.0
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -10.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = -2.0e-6
            dof_vel = 0.0
            dof_acc = -1.0e-7
            base_height = 0.0
            feet_air_time = 0.0
            collision = -0.2
            feet_stumble = 0.0
            action_rate = -0.02
            stand_still = 0.0
            tracking_joint_pos = 6.0
            tracking_joint_vel = 0.5
            tracking_root_pos = 1.0
            tracking_root_orientation = 1.0
            tracking_root_lin_vel = 1.0
            alive = 0.2

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class viewer(LeggedRobotCfg.viewer):
        pos = [3.0, -3.0, 2.0]
        lookat = [0.0, 0.0, 0.7]


class GR3MiniTraceCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 1.0e-4
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "gr3mini_trace"
        max_iterations = 2000
        save_interval = 100
