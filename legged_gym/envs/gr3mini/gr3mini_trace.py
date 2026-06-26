# SPDX-License-Identifier: BSD-3-Clause

import csv
import glob
import json
import os

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import normalize, quat_conjugate, quat_mul, torch_rand_float, to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot


GR3MINI_MOTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
]


class GR3MiniTrace(LeggedRobot):
    """GR3 mini reference motion tracking task.

    The policy action is interpreted as a residual on top of the current
    reference joint position from the CSV motion file.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._motion_cpu = self._load_motion_files(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _load_motion_files(self, cfg):
        trace_dir = cfg.motion.trace_dir.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        pattern = os.path.join(trace_dir, cfg.motion.file_pattern)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No motion trace files matched: {pattern}")

        motions = []
        expected_joint_names = None
        for path in files:
            if path.endswith(".json"):
                motion = self._load_json_motion_file(path, cfg)
            else:
                motion = self._load_csv_motion_file(path, cfg)

            if expected_joint_names is None:
                expected_joint_names = motion["joint_names"]
            elif motion["joint_names"] != expected_joint_names:
                raise ValueError(f"Motion joint name mismatch in {path}")

            motions.append(motion)

        return motions

    def _load_csv_motion_file(self, path, cfg):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(f"Motion trace must have at least two frames: {path}")

        joint_names = [name[:-4] for name in header if name.endswith("_dof")]
        root_pos = np.array(
            [[float(r["root_translateX"]), float(r["root_translateY"]), float(r["root_translateZ"])] for r in rows],
            dtype=np.float32,
        )
        # CSV stores WXYZ, Isaac Gym root states use XYZW.
        root_quat = np.array(
            [[float(r["root_quatX"]), float(r["root_quatY"]), float(r["root_quatZ"]), float(r["root_quatW"])] for r in rows],
            dtype=np.float32,
        )
        dof_pos = np.array([[float(r[f"{name}_dof"]) for name in joint_names] for r in rows], dtype=np.float32)
        return self._build_motion_dict(path, joint_names, root_pos, root_quat, dof_pos, cfg.motion.frame_dt)

    def _load_json_motion_file(self, path, cfg):
        with open(path) as f:
            payload = json.load(f)

        joint_names = payload.get("joint_names", GR3MINI_MOTION_JOINT_NAMES)
        root_pos = np.asarray(payload["root_pos_w"], dtype=np.float32)
        root_quat = np.asarray(payload["root_rot_w"], dtype=np.float32)
        dof_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
        if dof_pos.shape[1] != len(joint_names):
            raise ValueError(
                f"Motion joint count mismatch in {path}: got {dof_pos.shape[1]}, expected {len(joint_names)}"
            )
        if len(root_pos) < 2:
            raise ValueError(f"Motion trace must have at least two frames: {path}")

        dt = 1.0 / float(payload.get("fps", 1.0 / cfg.motion.frame_dt))
        return self._build_motion_dict(path, joint_names, root_pos, root_quat, dof_pos, dt)

    def _build_motion_dict(self, path, joint_names, root_pos, root_quat, dof_pos, dt):
        dof_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
        return {
            "path": path,
            "joint_names": list(joint_names),
            "root_pos": root_pos,
            "root_quat": root_quat,
            "root_lin_vel": root_lin_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }

    def _init_buffers(self):
        super()._init_buffers()

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        self._prepare_motion_tensors()
        self.future_offsets = self._build_future_offsets()
        self.future_steps = int(self.future_offsets.numel())
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.motion_frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.ref_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_root_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_root_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        self.ref_dof_vel = torch.zeros_like(self.dof_vel)
        self.motion_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._update_reference(torch.arange(self.num_envs, device=self.device))
        self._init_episode_metrics()
        self._init_termination_buffers()

    def _init_episode_metrics(self):
        metric_names = [
            "metric_joint_pos_rmse",
            "metric_joint_vel_rmse",
            "metric_root_pos_err",
            "metric_root_orientation_err",
            "metric_root_lin_vel_err",
            "metric_action_abs",
            "metric_torque_abs",
            "metric_base_height",
            "metric_foot_slip",
        ]
        self.episode_metric_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in metric_names
        }

    def _init_termination_buffers(self):
        self.termination_base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.termination_low_height = torch.zeros_like(self.termination_base_contact)
        self.termination_bad_orientation = torch.zeros_like(self.termination_base_contact)
        self.termination_root_pos = torch.zeros_like(self.termination_base_contact)
        self.termination_joint_pos = torch.zeros_like(self.termination_base_contact)
        self.termination_foot_slip = torch.zeros_like(self.termination_base_contact)
        self.termination_bad_velocity = torch.zeros_like(self.termination_base_contact)

    def _prepare_motion_tensors(self):
        first_joint_names = self._motion_cpu[0]["joint_names"]
        joint_to_col = {name: i for i, name in enumerate(first_joint_names)}
        missing = [name for name in self.dof_names if name not in joint_to_col]
        if missing:
            raise ValueError(
                "Motion trace does not contain all robot DOFs. Missing columns for: "
                + ", ".join(f"{name}_dof" for name in missing)
            )

        starts = []
        lengths = []
        root_pos = []
        root_quat = []
        root_lin_vel = []
        dof_pos = []
        dof_vel = []
        cursor = 0
        reorder = [joint_to_col[name] for name in self.dof_names]
        for motion in self._motion_cpu:
            starts.append(cursor)
            length = motion["dof_pos"].shape[0]
            lengths.append(length)
            cursor += length
            root_pos.append(motion["root_pos"])
            root_quat.append(motion["root_quat"])
            root_lin_vel.append(motion["root_lin_vel"])
            dof_pos.append(motion["dof_pos"][:, reorder])
            dof_vel.append(motion["dof_vel"][:, reorder])

        self.motion_starts = torch.tensor(starts, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_root_pos = to_torch(np.concatenate(root_pos, axis=0), device=self.device)
        self.motion_root_quat = to_torch(np.concatenate(root_quat, axis=0), device=self.device)
        self.motion_root_lin_vel = to_torch(np.concatenate(root_lin_vel, axis=0), device=self.device)
        self.motion_dof_pos = to_torch(np.concatenate(dof_pos, axis=0), device=self.device)
        self.motion_dof_vel = to_torch(np.concatenate(dof_vel, axis=0), device=self.device)
        self.num_motions = len(starts)

    def _build_future_offsets(self):
        explicit_offsets = getattr(self.cfg.motion, "future_offsets", None)
        if explicit_offsets is not None:
            offsets = np.asarray(explicit_offsets, dtype=np.int64)
        else:
            steps = int(self.cfg.motion.future_steps)
            if steps <= 0:
                return torch.empty(0, dtype=torch.long, device=self.device)

            start = int(getattr(self.cfg.motion, "future_offset_start", 1))
            end = int(getattr(self.cfg.motion, "future_offset_end", steps))
            spacing = getattr(self.cfg.motion, "future_offset_spacing", "linear")
            if steps == 1:
                offsets = np.asarray([start], dtype=np.int64)
            elif spacing == "exp":
                if start <= 0 or end <= 0:
                    raise ValueError("Exponential future offsets require positive start and end.")
                offsets = np.rint(np.geomspace(start, end, num=steps)).astype(np.int64)
                offsets[0] = start
                offsets[-1] = end
            elif spacing == "linear":
                offsets = np.rint(np.linspace(start, end, num=steps)).astype(np.int64)
                offsets[0] = start
                offsets[-1] = end
            else:
                raise ValueError(f"Unsupported future_offset_spacing: {spacing}")

        if np.any(offsets < 1):
            raise ValueError(f"Future offsets must be positive frame offsets, got {offsets.tolist()}")
        for i in range(1, len(offsets)):
            if offsets[i] <= offsets[i - 1]:
                offsets[i] = offsets[i - 1] + 1
        return torch.tensor(offsets, dtype=torch.long, device=self.device)

    def _sample_motion_frames(self, env_ids):
        self.motion_ids[env_ids] = torch.randint(self.num_motions, (len(env_ids),), device=self.device)
        lengths = self.motion_lengths[self.motion_ids[env_ids]]
        if self.cfg.motion.random_start:
            self.motion_frame_ids[env_ids] = torch.floor(torch.rand(len(env_ids), device=self.device) * lengths.float()).long()
        else:
            self.motion_frame_ids[env_ids] = 0

    def _motion_global_indices(self, env_ids):
        return self.motion_starts[self.motion_ids[env_ids]] + self.motion_frame_ids[env_ids]

    def _future_motion_global_indices(self, offsets, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        frame_ids = self.motion_frame_ids[env_ids].unsqueeze(1) + offsets.unsqueeze(0)
        lengths = self.motion_lengths[self.motion_ids[env_ids]].unsqueeze(1)
        frame_ids = torch.minimum(frame_ids, lengths - 1)
        return self.motion_starts[self.motion_ids[env_ids]].unsqueeze(1) + frame_ids

    def _update_reference(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        idx = self._motion_global_indices(env_ids)
        self.ref_root_pos[env_ids] = self.motion_root_pos[idx]
        self.ref_root_quat[env_ids] = self.motion_root_quat[idx]
        self.ref_root_lin_vel[env_ids] = self.motion_root_lin_vel[idx]
        self.ref_dof_pos[env_ids] = self.motion_dof_pos[idx]
        self.ref_dof_vel[env_ids] = self.motion_dof_vel[idx]
        self.motion_phase[env_ids] = self.motion_frame_ids[env_ids].float() / self.motion_lengths[self.motion_ids[env_ids]].float()

    def _advance_motion(self):
        lengths = self.motion_lengths[self.motion_ids]
        self.motion_frame_ids = torch.minimum(self.motion_frame_ids + 1, lengths - 1)
        self._update_reference()

    def _reset_dofs(self, env_ids):
        self._sample_motion_frames(env_ids)
        self._update_reference(env_ids)

        dof_noise = self.cfg.motion.dof_pos_noise
        vel_noise = self.cfg.motion.dof_vel_noise
        self.dof_pos[env_ids] = self.ref_dof_pos[env_ids] + torch_rand_float(
            -dof_noise, dof_noise, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = self.ref_dof_vel[env_ids] + torch_rand_float(
            -vel_noise, vel_noise, (len(env_ids), self.num_dof), device=self.device
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, self.gymtorch_unwrap(self.dof_state), self.gymtorch_unwrap(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] = self.ref_root_pos[env_ids] + self.env_origins[env_ids]
        noise = torch.tensor(self.cfg.motion.root_pos_noise, device=self.device)
        self.root_states[env_ids, :3] += torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device) * noise
        self.root_states[env_ids, 3:7] = self.ref_root_quat[env_ids]
        self.root_states[env_ids, 7:10] = self.ref_root_lin_vel[env_ids]
        self.root_states[env_ids, 10:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.gymtorch_unwrap(self.root_states), self.gymtorch_unwrap(env_ids_int32), len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        episode_lengths = torch.clamp(self.episode_length_buf[env_ids].float(), min=1.0)
        metric_means = {}
        for name, metric_sum in self.episode_metric_sums.items():
            metric_means[name] = torch.mean(metric_sum[env_ids] / episode_lengths)
        termination_rates = {
            "term_base_contact": torch.mean(self.termination_base_contact[env_ids].float()),
            "term_low_height": torch.mean(self.termination_low_height[env_ids].float()),
            "term_bad_orientation": torch.mean(self.termination_bad_orientation[env_ids].float()),
            "term_root_pos": torch.mean(self.termination_root_pos[env_ids].float()),
            "term_joint_pos": torch.mean(self.termination_joint_pos[env_ids].float()),
            "term_foot_slip": torch.mean(self.termination_foot_slip[env_ids].float()),
            "term_bad_velocity": torch.mean(self.termination_bad_velocity[env_ids].float()),
            "term_timeout": torch.mean(self.time_out_buf[env_ids].float()),
        }

        super().reset_idx(env_ids)

        episode_extras = self.extras.setdefault("episode", {})
        episode_extras.update(metric_means)
        episode_extras.update(termination_rates)
        for metric_sum in self.episode_metric_sums.values():
            metric_sum[env_ids] = 0.0
        self.termination_base_contact[env_ids] = False
        self.termination_low_height[env_ids] = False
        self.termination_bad_orientation[env_ids] = False
        self.termination_root_pos[env_ids] = False
        self.termination_joint_pos[env_ids] = False
        self.termination_foot_slip[env_ids] = False
        self.termination_bad_velocity[env_ids] = False

    def check_termination(self):
        cfg = self.cfg.termination
        if cfg.disable:
            self.time_out_buf[:] = False
            self.reset_buf[:] = False
            self.termination_base_contact[:] = False
            self.termination_low_height[:] = False
            self.termination_bad_orientation[:] = False
            self.termination_root_pos[:] = False
            self.termination_joint_pos[:] = False
            self.termination_foot_slip[:] = False
            self.termination_bad_velocity[:] = False
            return

        root_pos = self.root_states[:, :3] - self.env_origins
        root_pos_err = torch.norm(root_pos - self.ref_root_pos, dim=1)
        joint_pos_rmse = torch.sqrt(torch.mean(torch.square(self.dof_pos - self.ref_dof_pos), dim=1))

        self.termination_base_contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > cfg.contact_force,
            dim=1,
        )
        self.termination_low_height = root_pos[:, 2] < cfg.min_base_height
        self.termination_bad_orientation = torch.norm(self.projected_gravity[:, :2], dim=1) > cfg.max_projected_gravity_xy
        self.termination_root_pos = root_pos_err > cfg.max_root_pos_error
        self.termination_joint_pos = joint_pos_rmse > cfg.max_joint_pos_rmse

        if len(self.feet_indices) > 0:
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            feet_contact = self.contact_forces[:, self.feet_indices, 2] > cfg.foot_contact_force
            feet_xy_speed = torch.norm(self.rigid_body_state[:, self.feet_indices, 7:9], dim=-1)
            self.termination_foot_slip = torch.any(feet_contact & (feet_xy_speed > cfg.max_foot_slip_speed), dim=1)
        else:
            self.termination_foot_slip[:] = False

        base_lin_vel_bad = torch.norm(self.base_lin_vel, dim=1) > cfg.max_base_lin_vel
        base_ang_vel_bad = torch.norm(self.base_ang_vel, dim=1) > cfg.max_base_ang_vel
        dof_vel_bad = torch.any(torch.abs(self.dof_vel) > cfg.max_dof_vel, dim=1)
        self.termination_bad_velocity = base_lin_vel_bad | base_ang_vel_bad | dof_vel_bad

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = (
            self.termination_base_contact
            | self.termination_low_height
            | self.termination_bad_orientation
            | self.termination_root_pos
            | self.termination_joint_pos
            | self.termination_foot_slip
            | self.termination_bad_velocity
            | self.time_out_buf
        )

    def _post_physics_step_callback(self):
        self._advance_motion()

    def _compute_torques(self, actions):
        residual = actions * self.cfg.control.action_scale
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)
        target = self.ref_dof_pos + residual
        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _future_reference_observations(self):
        if self.future_steps == 0:
            return torch.empty(self.num_envs, 0, device=self.device)

        idx = self._future_motion_global_indices(self.future_offsets)
        future_dof_pos = self.motion_dof_pos[idx] * self.obs_scales.dof_pos
        future_root_pos_delta = self.motion_root_pos[idx] - self.ref_root_pos.unsqueeze(1)

        future_root_quat = self.motion_root_quat[idx]
        current_root_quat = self.ref_root_quat.unsqueeze(1).expand(-1, self.future_steps, -1)
        future_root_quat_delta = quat_mul(quat_conjugate(current_root_quat), future_root_quat)
        future_root_quat_delta = self._canonicalize_quat(future_root_quat_delta)

        future_ref = torch.cat((future_dof_pos, future_root_pos_delta, future_root_quat_delta), dim=-1)
        return future_ref.reshape(self.num_envs, -1)

    def compute_observations(self):
        phase = torch.stack(
            (torch.sin(2.0 * torch.pi * self.motion_phase), torch.cos(2.0 * torch.pi * self.motion_phase)), dim=1
        )
        root_pos_error = (self.root_states[:, :3] - self.env_origins) - self.ref_root_pos
        future_ref = self._future_reference_observations()
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                phase,
                root_pos_error,
                (self.dof_pos - self.ref_dof_pos) * self.obs_scales.dof_pos,
                (self.dof_vel - self.ref_dof_vel) * self.obs_scales.dof_vel,
                self.ref_dof_pos * self.obs_scales.dof_pos,
                future_ref,
                self.actions,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = False
        return torch.zeros_like(self.obs_buf[0])

    def compute_reward(self):
        super().compute_reward()
        self._accumulate_episode_metrics()

    def _accumulate_episode_metrics(self):
        root_pos = self.root_states[:, :3] - self.env_origins
        self.episode_metric_sums["metric_joint_pos_rmse"] += torch.sqrt(
            torch.mean(torch.square(self.dof_pos - self.ref_dof_pos), dim=1)
        )
        self.episode_metric_sums["metric_joint_vel_rmse"] += torch.sqrt(
            torch.mean(torch.square(self.dof_vel - self.ref_dof_vel), dim=1)
        )
        self.episode_metric_sums["metric_root_pos_err"] += torch.norm(root_pos - self.ref_root_pos, dim=1)
        self.episode_metric_sums["metric_root_orientation_err"] += self._root_orientation_error()
        self.episode_metric_sums["metric_root_lin_vel_err"] += torch.norm(
            self.base_lin_vel[:, :2] - self.ref_root_lin_vel[:, :2], dim=1
        )
        self.episode_metric_sums["metric_action_abs"] += torch.mean(torch.abs(self.actions), dim=1)
        self.episode_metric_sums["metric_torque_abs"] += torch.mean(torch.abs(self.torques), dim=1)
        self.episode_metric_sums["metric_base_height"] += self.root_states[:, 2] - self.env_origins[:, 2]
        self.episode_metric_sums["metric_foot_slip"] += self.termination_foot_slip.float()

    @staticmethod
    def gymtorch_unwrap(tensor):
        from isaacgym import gymtorch

        return gymtorch.unwrap_tensor(tensor)

    @staticmethod
    def _canonicalize_quat(quat):
        quat = normalize(quat)
        return torch.where(quat[..., 3:4] < 0.0, -quat, quat)

    def _root_orientation_error(self):
        base_quat = normalize(self.root_states[:, 3:7])
        ref_quat = normalize(self.ref_root_quat)
        quat_dot = torch.sum(base_quat * ref_quat, dim=1)
        quat_dot = torch.clamp(torch.abs(quat_dot), 0.0, 1.0)
        return 2.0 * torch.acos(quat_dot)

    # ------------ reward functions ------------
    def _reward_tracking_joint_pos(self):
        err = torch.mean(torch.square(self.dof_pos - self.ref_dof_pos), dim=1)
        return torch.exp(-err / 0.08)

    def _reward_tracking_joint_vel(self):
        err = torch.mean(torch.square(self.dof_vel - self.ref_dof_vel), dim=1)
        return torch.exp(-err / 4.0)

    def _reward_tracking_root_pos(self):
        root_pos = self.root_states[:, :3] - self.env_origins
        err = torch.sum(torch.square(root_pos - self.ref_root_pos), dim=1)
        return torch.exp(-err / 0.05)

    def _reward_tracking_root_orientation(self):
        angle = self._root_orientation_error()
        return torch.exp(-(angle * angle) / 0.2)

    def _reward_tracking_root_lin_vel(self):
        err = torch.sum(torch.square(self.base_lin_vel[:, :2] - self.ref_root_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / 0.5)

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)
