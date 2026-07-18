# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PD tracking-floor diagnostic: command the reference dof_pos directly as PD
targets (no policy) and measure the joint tracking error as a function of kp.

The Go2 DCMotor gains are runtime tensors, so the kp sweep runs in a single sim
session. Under pure reference targets the root is open loop — episodes end by
the usual tracking terminations and RSI resamples, which is fine: the per-step
joint error over alive envs IS the floor no policy can beat at that kp.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD tracking-floor diagnostic (no policy).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments.")
parser.add_argument("--steps", type=int, default=600, help="Steps per kp setting (50 Hz).")
parser.add_argument("--kp", type=str, default="25,60,100,220", help="Comma-separated kp sweep.")
parser.add_argument("--kd", type=str, default=None, help="Comma-separated kd (default: 0.5*sqrt(kp/25)).")
parser.add_argument("--task", type=str, default="Template-A2g2-Tracking-Direct-v0")
parser.add_argument("--clip", type=str, default="D1_009_KAN01_002", help="Motion clip name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import sys
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import a2g2_tracking.tasks  # noqa: F401
from a2g2_tracking.motion.motion_loader import LEG_ORDER


def main():
    kps = [float(k) for k in args_cli.kp.split(",")]
    if args_cli.kd is not None:
        kds = [float(k) for k in args_cli.kd.split(",")]
        assert len(kds) == len(kps)
    else:
        # critical-damping scaling from the stock (25, 0.5) pair
        kds = [0.5 * math.sqrt(kp / 25.0) for kp in kps]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.pd_replay = True
    env_cfg.motion_files = [f"{args_cli.clip}.pkl"]
    env_cfg.motion_cyclic = [True]
    # no random pushes: they knock over an open-loop gait and poison the floor
    env_cfg.events.push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    actuator = raw._robot.actuators["base_legs"]
    joint_names_c = [f"{leg}_{part}" for leg in LEG_ORDER for part in ("hip", "thigh", "calf")]

    rows = []
    for kp, kd in zip(kps, kds):
        with torch.inference_mode():
            actuator.stiffness[:] = kp
            actuator.damping[:] = kd
            env.reset()
            err_sum = torch.zeros(12, device=raw.device)
            n = 0
            done_count = 0
            base_contact_count = 0
            for _ in range(args_cli.steps):
                actions = torch.zeros((raw.num_envs, 12), device=raw.device)
                _, _, terminated, truncated, _ = env.step(actions)
                done = (terminated | truncated).bool()
                done_count += int(done.sum())
                base_contact_count += int(raw._term_causes["base_contact"].sum())
                alive = ~done
                if alive.any():
                    ref = raw._ref_frame()
                    err = (raw._robot.data.joint_pos - ref["dof_pos"]).abs()[alive]
                    err_sum += err.sum(dim=0)
                    n += int(alive.sum())
        per_joint = (err_sum / max(n, 1))[raw._perm_s2c].clone()
        mean_err = per_joint.mean().item()
        ep_len = args_cli.steps * raw.num_envs / max(done_count, 1)
        worst = sorted(zip(joint_names_c, per_joint.tolist()), key=lambda x: -x[1])[:3]
        rows.append((kp, kd, mean_err, ep_len, base_contact_count, worst))

    print("\n## PD tracking floor (pd_replay, clip %s, %d envs, %d steps/kp)\n" % (args_cli.clip, args_cli.num_envs, args_cli.steps))
    print("| kp | kd | mean joint err (rad) | mean ep len (steps) | base contacts | worst joints |")
    print("|---|---|---|---|---|---|")
    for kp, kd, mean_err, ep_len, falls, worst in rows:
        worst_s = ", ".join(f"{n} {e:.3f}" for n, e in worst)
        print(f"| {kp:.0f} | {kd:.2f} | {mean_err:.4f} | {ep_len:.0f} | {falls} | {worst_s} |")
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
