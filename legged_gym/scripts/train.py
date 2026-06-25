# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import statistics
import sys
import types
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def _install_ppo_update_metrics(ppo_runner):
    alg = ppo_runner.alg

    def update_with_metrics(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        mean_clip_fraction = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            with torch.inference_mode():
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float())

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            mean_clip_fraction += clip_fraction.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        mean_clip_fraction /= num_updates
        self.last_kl_mean = mean_kl
        self.last_clip_fraction = mean_clip_fraction
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

    alg.last_kl_mean = 0.0
    alg.last_clip_fraction = 0.0
    alg.update = types.MethodType(update_with_metrics, alg)

def _log_with_fixed_eta(ppo_runner, locs, width=80, pad=35):
    ppo_runner.tot_timesteps += ppo_runner.num_steps_per_env * ppo_runner.env.num_envs
    iteration_time = locs["collection_time"] + locs["learn_time"]
    ppo_runner.tot_time += iteration_time

    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            infotensor = torch.tensor([], device=ppo_runner.device)
            for ep_info in locs["ep_infos"]:
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(ppo_runner.device)))
            value = torch.mean(infotensor)
            ppo_runner.writer.add_scalar("Episode/" + key, value, locs["it"])
            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

    mean_std = ppo_runner.alg.actor_critic.std.mean()
    fps = int(ppo_runner.num_steps_per_env * ppo_runner.env.num_envs / iteration_time)

    ppo_runner.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
    ppo_runner.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
    ppo_runner.writer.add_scalar("Loss/kl_mean", ppo_runner.alg.last_kl_mean, locs["it"])
    ppo_runner.writer.add_scalar("Loss/clip_fraction", ppo_runner.alg.last_clip_fraction, locs["it"])
    ppo_runner.writer.add_scalar("Loss/learning_rate", ppo_runner.alg.learning_rate, locs["it"])
    ppo_runner.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
    ppo_runner.writer.add_scalar("Perf/total_fps", fps, locs["it"])
    ppo_runner.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    ppo_runner.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
    if len(locs["rewbuffer"]) > 0:
        ppo_runner.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        ppo_runner.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        ppo_runner.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), ppo_runner.tot_time)
        ppo_runner.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), ppo_runner.tot_time)

    run_start_iter = ppo_runner.current_learning_iteration
    total_iterations = run_start_iter + locs["num_learning_iterations"]
    completed_run_iterations = max(locs["it"] - run_start_iter + 1, 1)
    remaining_iterations = max(total_iterations - locs["it"] - 1, 0)
    eta = ppo_runner.tot_time / completed_run_iterations * remaining_iterations
    title = f" \033[1m Learning iteration {locs['it']}/{total_iterations} \033[0m "

    if len(locs["rewbuffer"]) > 0:
        log_string = (
            f"""{'#' * width}\n"""
            f"""{title.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'KL mean:':>{pad}} {ppo_runner.alg.last_kl_mean:.5f}\n"""
            f"""{'Clip fraction:':>{pad}} {ppo_runner.alg.last_clip_fraction:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        )
    else:
        log_string = (
            f"""{'#' * width}\n"""
            f"""{title.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'KL mean:':>{pad}} {ppo_runner.alg.last_kl_mean:.5f}\n"""
            f"""{'Clip fraction:':>{pad}} {ppo_runner.alg.last_clip_fraction:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )

    log_string += ep_string
    log_string += (
        f"""{'-' * width}\n"""
        f"""{'Total timesteps:':>{pad}} {ppo_runner.tot_timesteps}\n"""
        f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
        f"""{'Total time:':>{pad}} {ppo_runner.tot_time:.2f}s\n"""
        f"""{'ETA:':>{pad}} {eta:.1f}s\n"""
    )
    print(log_string)

def _install_terminal_progress(ppo_runner):
    def log_with_progress(locs, width=80, pad=35):
        _log_with_fixed_eta(ppo_runner, locs, width, pad)
        iteration = locs["it"] + 1
        total_iterations = ppo_runner.current_learning_iteration + locs["num_learning_iterations"]
        progress = 100.0 * iteration / max(total_iterations, 1)
        reward = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else float("nan")
        episode_len = statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else float("nan")
        print(
            f"[progress] iter {iteration}/{total_iterations} ({progress:.1f}%) | "
            f"mean_reward={reward:.3f} | mean_ep_len={episode_len:.1f} | "
            f"collection={locs['collection_time']:.2f}s | learning={locs['learn_time']:.2f}s",
            flush=True,
        )
        sys.stdout.flush()

    ppo_runner.log = log_with_progress

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    _install_ppo_update_metrics(ppo_runner)
    _install_terminal_progress(ppo_runner)
    if ppo_runner.log_dir is not None:
        print(f"TensorBoard log dir: {ppo_runner.log_dir}", flush=True)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
