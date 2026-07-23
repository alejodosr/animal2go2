"""The M0/M1 seam: schema of the canonical keypoint dict, and its validator.

The contract is `retarget/skeleton.py::extract_keypoints` (meters, Z-up,
leg order FR FL RR RL, quaternions xyzw). `validate_keypoints` asserts it,
including geometric sanity that catches the silent killers: FR/FL swap,
cm-vs-m, Y-up data.

Usage:
    python -m v2k.seam data/processed/*.npz
"""

import sys
from pathlib import Path

import numpy as np

LEG_ORDER = ["FR", "FL", "RR", "RL"]

# Canonical point order used everywhere in M0 (projection GT, detectors, lift).
KEYPOINT_NAMES = [
    "root", "chest",
    "FR_mount", "FL_mount", "RR_mount", "RL_mount",
    "FR_toe", "FL_toe", "RR_toe", "RL_toe",
]

REQUIRED_KEYS = {
    "fps", "num_frames", "source",
    "root_pos", "root_rot_xyzw", "chest_pos", "chest_rot_xyzw",
    "toe_pos", "leg_root_pos",
}


def canonical_points(kp):
    """(N, 10, 3) world points in KEYPOINT_NAMES order."""
    return np.concatenate(
        [kp["root_pos"][:, None], kp["chest_pos"][:, None],
         kp["leg_root_pos"], kp["toe_pos"]], axis=1)


def _trunk_axes(kp):
    """Per-frame trunk axes from positions (mirrors retarget.fit_trunk_frame)."""
    def norm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    x = norm(kp["chest_pos"] - kp["root_pos"])
    lr = kp["leg_root_pos"]  # FR FL RR RL
    y_seed = norm((lr[:, 1] - lr[:, 0]) + (lr[:, 3] - lr[:, 2]))
    z = norm(np.cross(x, y_seed))
    y = np.cross(z, x)
    return x, y, z


def validate_keypoints(d):
    """Assert the seam schema on dict `d`. Raises ValueError with the first
    violation; returns None on success."""
    def fail(msg):
        raise ValueError(f"validate_keypoints: {msg}")

    missing = REQUIRED_KEYS - set(d.keys())
    if missing:
        fail(f"missing keys {sorted(missing)}")

    n = int(d["num_frames"])
    if n < 2:
        fail(f"num_frames={n} < 2")
    fps = float(d["fps"])
    if not (0.0 < fps <= 500.0):
        fail(f"fps={fps} out of (0, 500]")

    shapes = {
        "root_pos": (n, 3), "root_rot_xyzw": (n, 4),
        "chest_pos": (n, 3), "chest_rot_xyzw": (n, 4),
        "toe_pos": (n, 4, 3), "leg_root_pos": (n, 4, 3),
    }
    for k, shape in shapes.items():
        a = np.asarray(d[k])
        if a.shape != shape:
            fail(f"{k}.shape={a.shape}, expected {shape}")
        if not np.issubdtype(a.dtype, np.floating):
            fail(f"{k}.dtype={a.dtype}, expected floating")
        if not np.isfinite(a).all():
            fail(f"{k} contains NaN/inf")

    for k in ("root_rot_xyzw", "chest_rot_xyzw"):
        nrm = np.linalg.norm(np.asarray(d[k]), axis=-1)
        if np.abs(nrm - 1.0).max() > 1e-3:
            fail(f"{k} not unit-norm (max |err|={np.abs(nrm - 1.0).max():.2e})")

    # Meters sanity: quadruped trunk (pelvis->chest) is decimeters, not cm/mm.
    trunk_len = np.linalg.norm(d["chest_pos"] - d["root_pos"], axis=-1)
    med_trunk = float(np.median(trunk_len))
    if not (0.15 < med_trunk < 1.5):
        fail(f"median trunk length {med_trunk:.3f} m outside (0.15, 1.5) — "
             "wrong units?")

    # Z-up sanity: feet live near the ground, root above the feet.
    toe_z = d["toe_pos"][..., 2]
    if not (-0.15 < float(toe_z.min()) < 0.4):
        fail(f"toe z min {toe_z.min():.3f} m — ground not at z≈0 or not Z-up")
    if float(np.median(d["root_pos"][:, 2])) <= float(np.median(toe_z)):
        fail("median root z below median toe z — not Z-up?")

    # Leg order, geometrically. In the trunk frame: left legs (FL, RL) have
    # y > 0, right legs (FR, RR) y < 0; front mounts sit ahead of rear mounts.
    x, y, z = _trunk_axes(d)
    origin = 0.5 * (d["root_pos"] + d["chest_pos"])
    mounts = d["leg_root_pos"] - origin[:, None]
    my = np.einsum("nlc,nc->nl", mounts, y)   # lateral coordinate per leg
    mx = np.einsum("nlc,nc->nl", mounts, x)   # longitudinal coordinate
    med_y = np.median(my, axis=0)             # FR FL RR RL
    if not (med_y[1] > 0 and med_y[3] > 0 and med_y[0] < 0 and med_y[2] < 0):
        fail(f"leg-mount lateral signs {med_y.round(3)} violate FR FL RR RL "
             "(left legs must have y>0 in the trunk frame) — FR/FL swap?")
    med_x = np.median(mx, axis=0)
    if not (min(med_x[0], med_x[1]) > max(med_x[2], med_x[3])):
        fail(f"leg-mount longitudinal medians {med_x.round(3)} violate "
             "front-before-rear — front/rear swap?")

    # Toes cross the midline in tight turns, so require a clear reversal
    # (~hip width would show on a real swap), not just any tiny inversion.
    toes = d["toe_pos"] - origin[:, None]
    ty = np.median(np.einsum("nlc,nc->nl", toes, y), axis=0)
    if ty[0] - ty[1] > 0.03 or ty[2] - ty[3] > 0.03:
        fail(f"toe lateral medians {ty.round(3)} clearly reversed per pair — "
             "FR/FL or RR/RL toe swap?")


def load_and_validate(path):
    d = dict(np.load(path, allow_pickle=True))
    validate_keypoints(d)
    return d


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1
    bad = 0
    for p in paths:
        try:
            load_and_validate(p)
            print(f"PASS {p}")
        except Exception as e:  # noqa: BLE001 — report and continue
            bad += 1
            print(f"FAIL {p}: {e}")
    print(f"{len(paths) - bad}/{len(paths)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
