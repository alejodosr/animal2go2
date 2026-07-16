"""Phase 0 smoke test for the vendored Unitree Go2 model.

Loads the Menagerie Go2 scene, prints the joint table (names, qpos addresses,
limits) as read from the MJCF, then sweeps each of the 12 joints through its
full range one at a time — purely kinematically (mj_forward, no dynamics).

Usage:
    python viz/smoke_test_go2.py                  # print table + render mp4
    python viz/smoke_test_go2.py --interactive    # sweep live in the viewer
"""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_SCENE_XML = REPO_ROOT / "assets" / "unitree_go2" / "scene.xml"

# Canonical leg order used everywhere in this project (Unitree convention).
LEG_ORDER = ["FR", "FL", "RR", "RL"]
JOINT_SUFFIXES = ["hip", "thigh", "calf"]
CANONICAL_JOINTS = [f"{leg}_{suf}_joint" for leg in LEG_ORDER for suf in JOINT_SUFFIXES]


def load_model():
    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE_XML))
    data = mujoco.MjData(model)
    return model, data


def print_joint_table(model):
    print(f"Model: {GO2_SCENE_XML.relative_to(REPO_ROOT)}")
    print(f"nq={model.nq} nv={model.nv} njnt={model.njnt}")
    print(f"{'joint':<16} {'qpos_adr':>8} {'range_lo':>9} {'range_hi':>9}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "(unnamed)"
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            print(f"{name:<16} {model.jnt_qposadr[j]:>8} {'(free joint, 7 dof)':>19}")
            continue
        lo, hi = model.jnt_range[j]
        print(f"{name:<16} {model.jnt_qposadr[j]:>8} {lo:>9.4f} {hi:>9.4f}")

    key = model.key("home")
    print(f"\nkeyframe 'home': qpos = {np.array2string(key.qpos, precision=3)}")

    missing = [n for n in CANONICAL_JOINTS
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
    assert not missing, f"expected joints missing from model: {missing}"
    print(f"\nAll 12 canonical joints present (order used: {', '.join(LEG_ORDER)} x hip/thigh/calf).")


def sweep_trajectory(model, seconds_per_joint=2.0, fps=30):
    """Yield (joint_name, qpos) frames sweeping each joint home -> lo -> hi -> home."""
    home = model.key("home").qpos.copy()
    n_frames = int(seconds_per_joint * fps)
    for name in CANONICAL_JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        rest = home[adr]
        # piecewise-linear: rest -> lo -> hi -> rest
        phase = np.concatenate([
            np.linspace(rest, lo, n_frames // 4),
            np.linspace(lo, hi, n_frames // 2),
            np.linspace(hi, rest, n_frames // 4),
        ])
        for angle in phase:
            qpos = home.copy()
            qpos[adr] = angle
            yield name, qpos


def render_video(model, data, out_path, fps=30):
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.25]
    cam.distance = 1.6
    cam.elevation = -20
    cam.azimuth = 135

    frames = []
    current = None
    for name, qpos in sweep_trajectory(model, fps=fps):
        if name != current:
            current = name
            print(f"  sweeping {name}")
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)  # kinematics only, no dynamics
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
    renderer.close()

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"wrote {out_path} ({len(frames)} frames)")


def run_interactive(model, data, fps=30):
    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for name, qpos in sweep_trajectory(model, fps=fps):
            if not viewer.is_running():
                break
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true",
                        help="sweep in the live passive viewer instead of rendering an mp4")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "media" / "go2_joint_sweep.mp4")
    args = parser.parse_args()

    model, data = load_model()
    print_joint_table(model)

    if args.interactive:
        run_interactive(model, data)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        render_video(model, data, args.out)


if __name__ == "__main__":
    main()
