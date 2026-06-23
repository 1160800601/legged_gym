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
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

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
    _install_terminal_progress(ppo_runner)
    if ppo_runner.log_dir is not None:
        print(f"TensorBoard log dir: {ppo_runner.log_dir}", flush=True)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
