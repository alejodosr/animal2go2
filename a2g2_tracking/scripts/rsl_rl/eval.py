# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic per-clip tracking evaluation (stage gates + brief §5 metrics).

Runs the inference policy (mean action, no exploration noise) with events
randomization and RSI noise disabled, collects completed episodes per clip
(discarding each env's first, phase-spread episode as warmup), and writes a
markdown metrics table next to the checkpoint.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL tracking checkpoint.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--episodes", type=int, default=512, help="Minimum completed episodes (after per-env warmup).")
parser.add_argument("--max_steps", type=int, default=4000, help="Hard cap on eval env steps.")
parser.add_argument("--out", type=str, default=None, help="Output markdown path (default: <run_dir>/eval_results.md).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_error_magnitude

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import a2g2_tracking.tasks  # noqa: F401

# animal2go2 repo root: logs/ is an SSD-backed symlink there
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TERM_CAUSES = ["root_pos", "root_ori", "joint_err", "base_contact", "time_out"]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # deterministic eval: no pushes/randomization, no RSI noise
    env_cfg.events = None
    env_cfg.rsi_joint_pos_noise = 0.0
    env_cfg.rsi_root_z_noise = 0.0

    log_root_path = os.path.abspath(os.path.join(_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    run_dir = os.path.dirname(resume_path)
    print(f"[INFO] Evaluating checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    lib = raw._motion_lib
    num_envs = raw.num_envs
    device = raw.device

    # per-env accumulators for the episode in progress
    steps = torch.zeros(num_envs, device=device)
    jerr_sum = torch.zeros(num_envs, device=device)
    jerr_max = torch.zeros(num_envs, device=device)
    ori_sum = torch.zeros(num_envs, device=device)
    xy_sum = torch.zeros(num_envs, device=device)
    height_sum = torch.zeros(num_envs, device=device)
    warmup_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    records: dict[str, list[dict]] = {name: [] for name in lib.names}
    total = 0

    obs = env.get_observations()
    for _ in range(args_cli.max_steps):
        with torch.inference_mode():
            # pre-step: tracking error of the current state vs the current reference
            ref = raw._ref_frame()
            data = raw._robot.data
            ref_pos_w = raw._ref_root_pos_w(ref)
            jerr = (data.joint_pos - ref["dof_pos"]).abs().mean(dim=-1)
            jerr_sum += jerr
            jerr_max = torch.maximum(jerr_max, jerr)
            ori_sum += quat_error_magnitude(data.root_quat_w, ref["root_rot"])
            xy_sum += (data.root_pos_w[:, :2] - ref_pos_w[:, :2]).norm(dim=-1)
            height_sum += (data.root_pos_w[:, 2] - ref_pos_w[:, 2]).abs()
            steps += 1
            clip_before = raw._clip_idx.clone()

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids):
                died_any = torch.stack(list(raw._term_causes.values())).any(dim=0)
                cause_masks = dict(raw._term_causes)
                cause_masks["time_out"] = raw.reset_time_outs & ~died_any
                for i in done_ids.tolist():
                    if warmup_done[i]:
                        n = max(int(steps[i].item()), 1)
                        cause = next((c for c in TERM_CAUSES if cause_masks[c][i]), "time_out")
                        records[lib.names[clip_before[i]]].append(
                            dict(
                                length=n,
                                jerr=jerr_sum[i].item() / n,
                                jmax=jerr_max[i].item(),
                                ori=ori_sum[i].item() / n,
                                xy=xy_sum[i].item() / n,
                                height=height_sum[i].item() / n,
                                cause=cause,
                            )
                        )
                        total += 1
                    warmup_done[i] = True
                    for buf in (steps, jerr_sum, jerr_max, ori_sum, xy_sum, height_sum):
                        buf[i] = 0.0
        if total >= args_cli.episodes:
            break

    # -- report --------------------------------------------------------------
    max_len = int(raw.max_episode_length)
    lines = [
        f"# Eval: {os.path.basename(run_dir)} / {os.path.basename(resume_path)}",
        "",
        f"- envs: {num_envs}, episodes: {total} (per-env warmup episode discarded)",
        f"- deterministic policy, events randomization OFF, RSI noise OFF, max episode length {max_len} steps",
        "",
        "| clip | eps | mean len (steps) | len %max | survival | mean joint err (rad) | max joint err (rad) "
        "| mean ori err (deg) | mean xy err (m) | mean height err (m) | terminations |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, eps in records.items():
        if not eps:
            continue
        n = len(eps)
        mean = lambda key: sum(e[key] for e in eps) / n  # noqa: E731
        survival = sum(e["cause"] == "time_out" for e in eps) / n
        hist = {c: sum(e["cause"] == c for e in eps) for c in TERM_CAUSES}
        hist_str = ", ".join(f"{c}: {v}" for c, v in hist.items() if v)
        lines.append(
            f"| {name} | {n} | {mean('length'):.1f} | {mean('length') / max_len * 100:.1f}% | {survival * 100:.1f}% "
            f"| {mean('jerr'):.4f} | {max(e['jmax'] for e in eps):.4f} | {math.degrees(mean('ori')):.2f} "
            f"| {mean('xy'):.3f} | {mean('height'):.4f} | {hist_str} |"
        )

    report = "\n".join(lines) + "\n"
    out_path = args_cli.out or os.path.join(run_dir, "eval_results.md")
    with open(out_path, "w") as f:
        f.write(report)
    print("\n" + report, flush=True)
    print(f"[INFO] Wrote {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    # flush before closing the app — Kit can swallow buffered stdout on exit
    sys.stdout.flush()
    simulation_app.close()
