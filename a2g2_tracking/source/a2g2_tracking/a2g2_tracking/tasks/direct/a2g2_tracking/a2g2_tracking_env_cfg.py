# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-tracking environment config for the Unitree Go2 (DeepMimic-style).

Load-bearing constants (see brief §3):
  - motions are 50 Hz → sim dt = 0.005 s, decimation = 4 → control at 50 Hz,
    exactly one reference frame per policy step.
  - observations/actions are SERIALIZED in canonical FR, FL, RR, RL order —
    the simulator-agnostic contract for Milestone 3.
"""

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# animal2go2 repo root (this file lives 7 levels below it); the motions dir can
# be overridden with A2G2_MOTIONS_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[7]
MOTIONS_DIR = Path(os.environ.get("A2G2_MOTIONS_DIR", _REPO_ROOT / "motions"))

# Phase 2/3 curriculum clips (gait loops → all cyclic). Names match motions/.
WALK_CLIP = "D1_007_KAN01_001"
TROT_CLIP = "D1_009_KAN01_002"
CANTER_CLIP = "D1_010_KAN01_004"


@configclass
class EventCfg:
    """Basic randomization — cheap Milestone 3 insurance, ON from the start."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class A2g2TrackingEnvCfg(DirectRLEnvCfg):
    # env — 50 Hz control over 200 Hz physics (one reference frame per step)
    decimation = 4
    episode_length_s = 10.0
    action_scale = 0.25
    # obs: gravity 3 + ang vel 3 + dof pos 12 + dof vel 12 + prev action 12
    #      + K×(ref dof 12 + ref root vel 6) + phase 2 = 44 + 18K
    num_ref_targets = 2  # K future reference frames in the actor obs
    action_space = 12
    observation_space = 80
    state_space = 88  # critic = actor obs + lin vel 3 + contacts 4 + height 1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot — keep the stock actuator config; its PD gains (Kp=25, Kd=0.5) and
    # torque limits go into the policy contract for the MuJoCo port
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # motion library
    motion_files: list = [f"{name}.pkl" for name in (WALK_CLIP, TROT_CLIP, CANTER_CLIP)]
    motion_cyclic: list = [True, True, True]

    # reference state initialization noise
    rsi_joint_pos_noise = 0.05  # rad, uniform ±
    rsi_root_z_noise = 0.02  # m, uniform [0, +z]

    # early termination bounds (loose — they truncate hopeless rollouts)
    term_root_pos_err = 0.5  # m
    term_root_ori_err = 0.785  # rad (45°)
    term_joint_err = 1.0  # rad, mean over 12 dofs

    # reward weights (DeepMimic-style exp kernels; brief §3 table)
    rew_joint_pos_w = 0.5
    rew_joint_pos_k = 5.0
    rew_joint_vel_w = 0.05
    rew_joint_vel_k = 0.1
    rew_root_ori_w = 0.15
    rew_root_ori_k = 10.0
    rew_root_vel_w = 0.15
    rew_root_vel_k = 2.0
    rew_root_height_w = 0.05
    rew_root_height_k = 100.0
    rew_contact_match_w = 0.1
    rew_action_rate_w = -0.01
    rew_torque_w = -1.0e-4

    # contact detection threshold on the feet/base sensors
    contact_force_threshold = 1.0  # N

    # modes (set by play.py, not for training)
    kinematic_replay = False  # force-set reference states every step (Phase 2 gate)
    replay_clip: str | None = None  # clip name for replay; None → env_idx % num_clips
    enable_ghost = False  # transparent reference ghost robot (replay/videos)
