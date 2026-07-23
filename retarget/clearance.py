"""Kinematic clearance projection: feasibility for morphology-gap poses.

Third infeasibility class in the retarget pipeline (after temporal —
timewarp.py — and environmental — reground.py): poses the source animal
can reach but the robot cannot without body-ground penetration. Dogs fold
far more compactly than the Go2's two-segment legs (digitigrade feet, a
carpus, different proportions), so deep crouches retarget into references
whose KNEES are below the floor — measured 2026-07-23: sit 39% of frames
(max 22 cm deep), canter 32% (8 cm), walk 14% (22 cm, at its spawn
frame), jump prep 9 cm at the launch crouch. Such poses are kinematically
impossible in sim: the policy can never track them, and RSI spawns into
them trigger violent PhysX depenetration.

Projection, two mechanisms in order of preference:

1. Root lift, capped at MAX_LIFT: raises the whole body by the minimal dz
   that clears every knee while the FEET stay at their original world
   positions (legs re-IK against the lifted root — foot trajectories, the
   tracked end-effector quantity, are exactly preserved). Handles the
   mild cases (canter 8 cm, jump 9 cm). The cap exists because deep
   crouches (22 cm penetration) would need lifts that put ground-level
   feet outside the 0.426 m leg reach — the IK then silently clamps and
   rips stance feet 20+ cm off the floor, destroying the pose.
2. Joint-space thigh raise, for penetration remaining at the capped lift.
   The knee's world position depends only on hip abduction (q1, kept) and
   thigh pitch (q2) — not on the calf — so the fix is direct: per
   offending leg-frame, pick the feasible q2 nearest the original (grid
   search over the joint range, knee height is what defines feasibility —
   no monotonicity assumption, so it is robust to tilted trunks where a
   naive "pull the foot inward" moves the knee DEEPER), then re-solve the
   calf angle q3 to put the foot back at its original height if the
   calf circle reaches it (stance stays stance), else as close to the
   original foot as possible. The joint deltas are Gaussian-smoothed so
   entering/leaving a corrected segment never pops.

Frames already clear come back with zero lift and zero delta, so clips
without deep poses pass through bit-identical. Because deep poses may
change foot heights slightly, callers should relabel contacts afterwards
(postprocess step 6 relabeling is reused by the pkl-level apply script).

Deliberately out of scope (v1): trunk-ground clearance (a sitting Go2's
low rear trunk is real support geometry, and lifting it away would destroy
the pose; calf-ground contact is likewise legitimate and allowed — only
knees *below* the surface are projected out).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from retarget import ik
from retarget.postprocess import _apply_per_leg, foot_base_positions

CLEARANCE_Z = 0.03   # m — calf-joint (knee) collision radius above the floor
MAX_LIFT = 0.08      # m — root-lift cap; beyond it stance feet leave reach
LIFT_SMOOTH_S = 0.1  # s — Gaussian sigma on the lift / pull trajectories
BISECT_ITERS = 20    # resolves the bisections to < 0.1 mm
# Only quasi-static, sustained penetration runs are projected — those are
# POSES the robot must hold (sit 28 s @ 0.03 m/s, trot lead-in crouch 3.6 s
# @ 0.04, walk 1.4 s @ 0.13). Fast or short dips are swing/turn retarget
# artifacts the tracker skims past harmlessly (canter pirouette 1.2 s but @
# 1.85 m/s, jump prep <= 0.2 s runs) — stage12 measured the cost of
# projecting them: root height moved 8 cm mid-gallop, canter survival
# collapsed 66->43% with 108/144 deaths at the lift ramp-down.
MIN_POSE_S = 0.5      # s — run duration floor (after gap merging)
POSE_SPEED_MAX = 0.6  # m/s — mean planar root speed ceiling for a pose run
GAP_MERGE_FRAMES = 3  # penetration runs separated by <= this are one run


def knee_base_positions(dof_pos):
    """Knee (calf-joint) positions in the base frame, (N, 12) -> (N, 4, 3)."""
    dof = np.asarray(dof_pos, dtype=np.float64).reshape(len(dof_pos), 4, 3)
    knees = np.zeros((len(dof), 4, 3))
    for leg in range(4):
        q1, q2 = dof[:, leg, 0], dof[:, leg, 1]
        x = -ik.THIGH_LENGTH * np.sin(q2)
        z = -ik.THIGH_LENGTH * np.cos(q2)
        d = ik.HIP_LATERAL[leg]
        knees[:, leg] = (
            np.stack([x, d * np.cos(q1) - z * np.sin(q1), d * np.sin(q1) + z * np.cos(q1)], axis=-1)
            + ik.HIP_OFFSETS[leg]
        )
    return knees


def _knees_z(root_pos, rot, dof):
    """World knee heights (N, 4)."""
    return (root_pos[:, None, :] + _apply_per_leg(rot, knee_base_positions(dof)))[..., 2]


def _lift_solve(root_pos, rot, foot_world, lift):
    """Dof and knee heights after lifting the root with feet world-pinned."""
    lifted = root_pos.copy()
    lifted[:, 2] += lift
    dof, _ = ik.clamp_to_limits(ik.ik(foot_base_positions(lifted, rot, foot_world)))
    return _knees_z(lifted, rot, dof), dof


HYSTERESIS_W = 5.0  # rad-per-rad cost of moving the q2 pick between frames
TAPER_FRAMES = 5    # linear blend of the correction into neighboring frames


def _chain_pick(cost_rows, grid, forward):
    """Sequential argmin over rows with a strong continuity penalty.

    cost_rows: (R, G) per-frame candidate costs (inf = infeasible). Returns
    the picked grid values (R,). The chain is run forward or backward; a
    strong penalty (HYSTERESIS_W per rad of movement) forbids side flips
    unless feasibility forces one.
    """
    idx = range(len(cost_rows)) if forward else range(len(cost_rows) - 1, -1, -1)
    picks = np.empty(len(cost_rows))
    prev = None
    for i in idx:
        cost = cost_rows[i].copy()
        if prev is not None:
            cost = cost + HYSTERESIS_W * np.abs(grid - prev)
        j = int(cost.argmin())
        picks[i] = grid[j]
        if np.isfinite(cost_rows[i][j]):
            prev = picks[i]
    return picks


def _runs_of(mask):
    edges = np.flatnonzero(np.diff(mask))
    bounds = np.concatenate([[0], edges + 1, [len(mask)]])
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if mask[bounds[i]]
    ]


def _raise_thighs(root_pos, rot, dof, foot_world, need):
    """Joint-space knee clearing, one contiguous run at a time.

    Per penetrating run of each leg: grid-search thigh pitch q2 for the
    feasible (knee world z >= CLEARANCE_Z) value nearest the original,
    chained with a strong hysteresis so one run stays on one side of the
    infeasible band (forward and backward chains are tried, cheapest
    kept — a per-frame argmin flip measured 145 rad/s on the trot
    crouch); then grid-search calf q3 for the foot nearest its original
    target (an analytic arctan2 solution wraps and bound-clamps
    discontinuously). The correction tapers linearly into TAPER_FRAMES
    neighboring clear frames so run edges never pop. q1 is untouched.
    """
    out = dof.copy()
    if not need.any():
        return out
    n = len(dof)
    dof4 = dof.reshape(n, 4, 3)
    out3 = out.reshape(n, 4, 3)
    R = rot.as_matrix()
    for leg in range(4):
        if not need[:, leg].any():
            continue
        d = ik.HIP_LATERAL[leg]
        lo2, hi2 = ik.JOINT_LIMITS[leg][1]
        c_lo, c_hi = ik.JOINT_LIMITS[leg][2]
        grid = np.linspace(lo2, hi2, 361)
        q3_grid = np.linspace(c_lo, c_hi, 361)
        for s, e in _runs_of(need[:, leg]):
            rows = np.arange(s, e)
            q1 = dof4[rows, leg, 0]
            # knee world height for every candidate q2 (rows x grid)
            x = np.broadcast_to(-ik.THIGH_LENGTH * np.sin(grid)[None, :], (len(rows), len(grid)))
            z = -ik.THIGH_LENGTH * np.cos(grid)[None, :]
            y_out = d * np.cos(q1)[:, None] - z * np.sin(q1)[:, None]
            z_out = d * np.sin(q1)[:, None] + z * np.cos(q1)[:, None]
            knee_base = np.stack([x, y_out, z_out], axis=-1) + ik.HIP_OFFSETS[leg]
            knee_w = np.einsum("nij,ngj->ngi", R[rows], knee_base) + root_pos[rows, None, :]
            cost2 = np.abs(grid[None, :] - dof4[rows, leg, 1][:, None])
            cost2[knee_w[..., 2] < CLEARANCE_Z] = np.inf
            fwd = _chain_pick(cost2, grid, forward=True)
            bwd = _chain_pick(cost2, grid, forward=False)
            take = np.abs(np.diff(fwd)).sum() <= np.abs(np.diff(bwd)).sum()
            q2 = fwd if take else bwd

            # calf: foot nearest the original target given the chosen q2
            foot_b = foot_base_positions(root_pos, rot, foot_world)[rows, leg]
            a = q2[:, None] + q3_grid[None, :]
            fx = -ik.THIGH_LENGTH * np.sin(q2)[:, None] - ik.CALF_LENGTH * np.sin(a)
            fz = -ik.THIGH_LENGTH * np.cos(q2)[:, None] - ik.CALF_LENGTH * np.cos(a)
            fy_out = d * np.cos(q1)[:, None] - fz * np.sin(q1)[:, None]
            fz_out = d * np.sin(q1)[:, None] + fz * np.cos(q1)[:, None]
            cand = np.stack([fx, fy_out, fz_out], axis=-1) + ik.HIP_OFFSETS[leg]
            dist = np.linalg.norm(cand - foot_b[:, None, :], axis=-1)
            q3 = _chain_pick(dist, q3_grid, forward=True)

            delta = np.zeros((n, 2))
            delta[rows, 0] = q2 - dof4[rows, leg, 1]
            delta[rows, 1] = q3 - dof4[rows, leg, 2]
            # taper into the neighbors: clear frames next to the run get a
            # shrinking share of the edge correction (raising a clear knee
            # keeps it clear), so entering/leaving the run never pops
            for k in range(1, TAPER_FRAMES + 1):
                w = 1.0 - k / (TAPER_FRAMES + 1)
                if s - k >= 0 and not need[s - k, leg]:
                    delta[s - k] = w * delta[s]
                if e - 1 + k < n and not need[e - 1 + k, leg]:
                    delta[e - 1 + k] = w * delta[e - 1]
            out3[:, leg, 1] += delta[:, 0]
            out3[:, leg, 2] += delta[:, 1]
    return out


def project_clearance(motion):
    """Project a §7 motion dict onto knee-above-ground poses.

    Returns (motion_out, report). Root xy, orientation and timing are
    untouched; root z (capped lift) and dof_pos (re-IK / thigh raise)
    change, and only on frames that needed it. Feet move only on frames
    where the joint-space raise ran (reported as foot_shift_max).
    """
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    rot = Rotation.from_quat(np.asarray(motion["root_rot"]))
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = float(motion["fps"])
    n = len(root_pos)

    foot_world = root_pos[:, None, :] + _apply_per_leg(rot, ik.fk(dof_pos))
    depth = np.maximum(0.0, CLEARANCE_Z - _knees_z(root_pos, rot, dof_pos).min(axis=1))

    # pose gate: project only sustained (>= MIN_POSE_S after merging
    # <= GAP_MERGE_FRAMES gaps) AND quasi-static (mean planar root speed
    # <= POSE_SPEED_MAX) runs; fast/short swing dips stay untouched
    below = depth > 0
    merged = below.copy()
    for s, e in _runs_of(~below):
        if 0 < s and e < n and e - s <= GAP_MERGE_FRAMES:
            merged[s:e] = True
    speed = np.linalg.norm(np.gradient(root_pos[:, :2], axis=0) * fps, axis=-1)
    pose_mask = np.zeros(n, dtype=bool)
    for s, e in _runs_of(merged):
        if (e - s) / fps >= MIN_POSE_S and speed[s:e].mean() <= POSE_SPEED_MAX:
            pose_mask[s:e] = True
    depth = np.where(pose_mask, depth, 0.0)

    report = {
        "frames_below": int(below.sum()),
        "pose_frames": int((depth > 0).sum()),
        "max_depth": float(depth.max()),
        "max_lift": 0.0,
        "lift_fraction": 0.0,
        "raise_fraction": 0.0,
        "max_dq": 0.0,
        "foot_shift_max": 0.0,
    }
    if not (depth > 0).any():
        return motion, report

    # stage 1 — root lift, bisected per frame on [0, MAX_LIFT] (knee height
    # is monotone in lift); frames the cap cannot clear end up at MAX_LIFT
    lo = np.zeros(n)
    hi = np.where(depth > 0, MAX_LIFT, 0.0)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        kz, _ = _lift_solve(root_pos, rot, foot_world, mid)
        ok = kz.min(axis=1) >= CLEARANCE_Z
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    lift = hi
    lift = np.maximum(lift, gaussian_filter1d(lift, LIFT_SMOOTH_S * fps, mode="nearest"))
    lifted_root = root_pos + np.stack([np.zeros(n), np.zeros(n), lift], axis=-1)

    kz, dof_lifted = _lift_solve(root_pos, rot, foot_world, lift)
    # untouched frames keep their original joints bit-identical (the IK
    # round-trip is only exact where it actually had to run)
    dof_lifted[lift <= 1e-9] = dof_pos[lift <= 1e-9]

    # stage 2 — joint-space thigh raise where the capped lift wasn't enough
    # (pose runs only — transients stay by the duration gate above); close
    # 1-3 frame gaps so one contiguous run gets one hysteresis chain (split
    # runs may resolve to opposite sides of the infeasible band)
    need = (kz < CLEARANCE_Z) & pose_mask[:, None]
    for leg in range(4):
        for s, e in _runs_of(~need[:, leg]):
            if 0 < s and e < n and e - s <= 3:
                need[s:e, leg] = True
    dof_final = _raise_thighs(lifted_root, rot, dof_lifted, foot_world, need)

    kz_final = _knees_z(lifted_root, rot, dof_final)
    feet_final = lifted_root[:, None, :] + _apply_per_leg(rot, ik.fk(dof_final))

    out = dict(motion)
    out["root_pos"] = lifted_root
    out["dof_pos"] = dof_final
    report.update(
        max_lift=float(lift.max()),
        lift_fraction=float((lift > 1e-6).mean()),
        raise_fraction=float(need.any(axis=1).mean()),
        max_dq=float(np.abs(dof_final - dof_pos).max()),
        residual_depth=float(np.maximum(0.0, CLEARANCE_Z - kz_final.min(axis=1)).max()),
        foot_shift_max=float(np.linalg.norm(feet_final - foot_world, axis=-1).max()),
    )
    return out, report
