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

def _install_terminal_progress(ppo_runner):
    original_log = ppo_runner.log

    def log_with_progress(locs, width=80, pad=35):
        original_log(locs, width, pad)
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
