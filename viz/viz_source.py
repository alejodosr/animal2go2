"""Visualize raw dog skeleton motion from a BVH clip (matplotlib 3D).

Sanity check for parsing: the dog should stand on the z=0 ground plane,
head forward, and visibly walk. Up-axis / unit bugs die here.

Usage:
    python viz/viz_source.py data/D1_001_KAN01_001.bvh                # mp4 export
    python viz/viz_source.py data/D1_001_KAN01_001.bvh --show         # live window
    python viz/viz_source.py data/D1_001_KAN01_001.bvh --start 100 --end 400
"""

import argparse
import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.skeleton import (
    DOG_TOE_JOINTS,
    LEG_ORDER,
    bone_list,
    forward_kinematics,
    parse_bvh,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOE_COLORS = {"FR": "tab:red", "FL": "tab:orange", "RR": "tab:blue", "RL": "tab:cyan"}


def draw_frame(ax, positions, bones, frame, center, half_range):
    ax.clear()
    for parent, child in bones:
        p, c = positions[parent][frame], positions[child][frame]
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], "k-", lw=1.5)
    for leg, joint in zip(LEG_ORDER, DOG_TOE_JOINTS):
        t = positions[joint][frame]
        ax.scatter(*t, color=TOE_COLORS[leg], s=25, label=leg)

    # ground grid at z=0
    gx, gy = np.meshgrid(
        np.linspace(center[0] - half_range, center[0] + half_range, 9),
        np.linspace(center[1] - half_range, center[1] + half_range, 9),
    )
    ax.plot_wireframe(gx, gy, np.zeros_like(gx), color="gray", lw=0.3, alpha=0.5)

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(0, 2 * half_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2,
                        help="render every Nth frame (source is 60 fps)")
    parser.add_argument("--show", action="store_true", help="live window instead of mp4")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--follow", action="store_true", default=True,
                        help="camera follows the root (default)")
    args = parser.parse_args()

    clip = parse_bvh(args.clip)
    positions, _ = forward_kinematics(clip)
    bones = bone_list(clip)
    end = args.end if args.end is not None else clip.num_frames
    frames = range(args.start, min(end, clip.num_frames), args.stride)
    fps_out = clip.fps / args.stride

    print(f"{clip.name}: {clip.num_frames} frames @ {clip.fps:.0f} fps, "
          f"rendering frames {args.start}..{end} stride {args.stride} "
          f"-> {fps_out:.0f} fps output")

    root = positions["Hips"]
    half_range = 0.7

    if args.show:
        matplotlib.use("TkAgg")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if args.show:
        plt.ion()
        for f in frames:
            draw_frame(ax, positions, bones, f, root[f], half_range)
            ax.set_title(f"{clip.name}  frame {f}")
            plt.pause(1.0 / fps_out)
        return

    out = args.out or REPO_ROOT / "media" / f"source_{clip.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps_out)
    for f in frames:
        draw_frame(ax, positions, bones, f, root[f], half_range)
        ax.set_title(f"{clip.name}  frame {f}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
