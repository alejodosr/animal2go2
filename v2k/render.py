"""Rendered evaluation tier: mocap keypoints -> synthetic monocular video + 2D GT.

Draws the 10 canonical points as a stick figure through a known static
camera, writes the mp4 and a `<clip>_gt2d.npz` holding the projected 2D
ground truth plus the camera, so every later phase (2D detect, 3D lift,
metric root) can be scored in isolation.

Usage:
    python -m v2k.render data/processed/D1_007_KAN01_001.npz [--view side]
    # writes data/synthetic/<clip>_<view>.mp4 and <clip>_<view>_gt2d.npz
"""

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.camera import PinholeCamera
from v2k.seam import KEYPOINT_NAMES, canonical_points, load_and_validate

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "synthetic"

# Stick-figure links by canonical name (mounts hang off the trunk endpoints;
# legs are mount->toe segments — no knees at the seam, and none needed).
LINKS = [
    ("root", "chest"),
    ("chest", "FR_mount"), ("chest", "FL_mount"),
    ("root", "RR_mount"), ("root", "RL_mount"),
    ("FR_mount", "FR_toe"), ("FL_mount", "FL_toe"),
    ("RR_mount", "RR_toe"), ("RL_mount", "RL_toe"),
]
LINK_IDX = [(KEYPOINT_NAMES.index(a), KEYPOINT_NAMES.index(b)) for a, b in LINKS]

# One saturated, unique RGB per keypoint (markers drawn last, on top of the
# gray links) so tests can find marker centroids by color in the raw frames.
KEYPOINT_COLORS = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (0, 0, 128), (128, 128, 0),
], dtype=np.uint8)


def default_camera(kp, view="side", width=1280, height=720, fov_x_deg=50.0):
    """Static camera framing the whole clip trajectory.

    Views are relative to the clip's dominant horizontal travel direction
    (world axes would silently give an end-on "side" view for clips that
    happen to travel along Y).
    """
    pts = canonical_points(kp)
    lo, hi = pts.reshape(-1, 3).min(0), pts.reshape(-1, 3).max(0)
    center = 0.5 * (lo + hi)
    span = float(np.linalg.norm(hi - lo))
    dist = max(2.5, 1.2 * span / (2.0 * np.tan(np.deg2rad(fov_x_deg) / 2.0)))

    travel = kp["root_pos"][-1, :2] - kp["root_pos"][0, :2]
    if np.linalg.norm(travel) < 0.5:  # in-place clip: any heading works
        travel = np.array([1.0, 0.0])
    fwd = travel / np.linalg.norm(travel)
    left = np.array([-fwd[1], fwd[0]])
    planar = {"side": left, "front": fwd,
              "three_quarter": (fwd + left) / np.sqrt(2.0)}[view]
    lift = {"side": 0.12, "front": 0.12, "three_quarter": 0.25}[view]
    off = np.array([planar[0], planar[1], lift])
    off = off / np.linalg.norm(off)
    eye = center + off * dist
    return PinholeCamera.look_at(eye, center, fov_x_deg, width, height)


def render_frame_array(uv, depth, width, height, fig_state=None,
                       marker_px=6.0, line_px=2.5):
    """Rasterize one frame; returns (H, W, 3) uint8 and reusable fig_state."""
    if fig_state is None:
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        lines = [ax.plot([], [], color="0.45", lw=line_px, solid_capstyle="round",
                         zorder=1)[0] for _ in LINK_IDX]
        dots = ax.scatter(np.zeros(len(KEYPOINT_NAMES)),
                          np.zeros(len(KEYPOINT_NAMES)),
                          s=(2 * marker_px) ** 2, c=KEYPOINT_COLORS / 255.0,
                          zorder=2, edgecolors="none")
        fig_state = (fig, lines, dots)
    fig, lines, dots = fig_state

    vis = depth > 0.05
    for ln, (a, b) in zip(lines, LINK_IDX):
        if vis[a] and vis[b]:
            ln.set_data([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]])
        else:
            ln.set_data([], [])
    offsets = np.where(vis[:, None], uv, np.full_like(uv, -1e4))
    dots.set_offsets(offsets)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return buf, fig_state


def render_clip(kp, camera, out_mp4, out_gt, max_frames=None):
    """Render the clip through `camera`; write mp4 + 2D GT npz."""
    pts = canonical_points(kp)
    if max_frames is not None:
        pts = pts[:max_frames]
    uv, depth = camera.project(pts)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fps = float(kp["fps"])
    writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=1)
    fig_state = None
    for i in range(len(uv)):
        frame, fig_state = render_frame_array(
            uv[i], depth[i], camera.width, camera.height, fig_state)
        writer.append_data(frame)
    writer.close()
    plt.close(fig_state[0])

    np.savez_compressed(
        out_gt,
        source=str(kp["source"]), fps=fps, num_frames=len(uv),
        keypoint_names=np.array(KEYPOINT_NAMES),
        uv=uv, depth=depth, xyz_world=pts,
        **camera.params(),
    )
    return uv, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--view", default="side",
                        choices=["side", "front", "three_quarter"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    for path in args.clips:
        kp = load_and_validate(path)
        cam = default_camera(kp, args.view, args.width, args.height)
        stem = f"{Path(path).stem}_{args.view}"
        out_mp4 = OUT_DIR / f"{stem}.mp4"
        out_gt = OUT_DIR / f"{stem}_gt2d.npz"
        uv, depth = render_clip(kp, cam, out_mp4, out_gt,
                                max_frames=args.max_frames)
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] < cam.width) &
                  (uv[..., 1] >= 0) & (uv[..., 1] < cam.height)).mean()
        print(f"wrote {out_mp4.relative_to(REPO_ROOT)} "
              f"({len(uv)} frames @ {float(kp['fps']):.1f} fps, "
              f"{inside:.1%} of keypoints in frame) + "
              f"{out_gt.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
