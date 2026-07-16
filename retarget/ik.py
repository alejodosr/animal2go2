"""Analytic FK/IK for the Unitree Go2 legs.

Each Go2 leg is a 3-DOF chain: hip abduction (about +x), hip pitch / "thigh"
(about +y), knee / "calf" (about +y). Reaching a 3D foot point with such a
chain has a closed-form solution:

  1. Express the foot target in the hip frame (base axes, origin at the hip
     abduction joint).
  2. The abduction angle comes from the y-z geometry: the foot must lie in the
     leg's sagittal plane, which sits at a fixed lateral offset `d` from the
     hip x-axis (the thigh body offset in the MJCF).
  3. Inside that plane the remaining thigh+calf pair is a planar two-link arm:
     knee angle from the law of cosines, thigh angle from atan2 minus the
     interior angle.
  4. We always pick the knee-backward branch (calf angle negative), matching
     Go2's build and its calf joint range.

Conventions (same as the rest of the project):
  - Leg order FR, FL, RR, RL; per-leg angle order (hip, thigh, calf).
  - Foot positions are in the *base frame* (origin and axes of the trunk
    body), meters. The foot point is the end of the calf link, i.e. the
    center of the foot sphere up to a 2 mm fore-aft offset we ignore.
  - Zero configuration is the leg stretched straight down.

Geometry and joint limits are transcribed from assets/unitree_go2/go2.xml;
tests/test_ik.py cross-checks them against the loaded MuJoCo model so they
cannot silently drift.
"""

import numpy as np

LEG_ORDER = ["FR", "FL", "RR", "RL"]

# Hip abduction joint origin in the base frame (MJCF <body name="*_hip" pos>).
HIP_OFFSETS = np.array([
    [0.1934, -0.0465, 0.0],   # FR
    [0.1934, 0.0465, 0.0],    # FL
    [-0.1934, -0.0465, 0.0],  # RR
    [-0.1934, 0.0465, 0.0],   # RL
])

# Lateral offset of the leg plane from the hip joint, along the hip y-axis
# (MJCF <body name="*_thigh" pos>). Sign encodes the left/right mirroring —
# it is the only thing that differs between legs in the FK/IK math.
HIP_LATERAL = np.array([-0.0955, 0.0955, -0.0955, 0.0955])  # FR, FL, RR, RL

THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
MAX_REACH = THIGH_LENGTH + CALF_LENGTH

# (leg, joint, lo/hi) from the MJCF joint classes: abduction shared, thigh
# differs front/back, knee shared.
_ABDUCTION_RANGE = (-1.0472, 1.0472)
_FRONT_THIGH_RANGE = (-1.5708, 3.4907)
_BACK_THIGH_RANGE = (-0.5236, 4.5379)
_KNEE_RANGE = (-2.7227, -0.83776)
JOINT_LIMITS = np.array([
    [_ABDUCTION_RANGE, _FRONT_THIGH_RANGE, _KNEE_RANGE],  # FR
    [_ABDUCTION_RANGE, _FRONT_THIGH_RANGE, _KNEE_RANGE],  # FL
    [_ABDUCTION_RANGE, _BACK_THIGH_RANGE, _KNEE_RANGE],   # RR
    [_ABDUCTION_RANGE, _BACK_THIGH_RANGE, _KNEE_RANGE],   # RL
])

# Margin kept inside the workspace boundary when clamping targets, so the
# law-of-cosines argument stays strictly inside [-1, 1].
_REACH_EPS = 1e-4


def leg_fk(q, leg):
    """Foot position from joint angles for one leg.

    q: (..., 3) angles (hip, thigh, calf) in radians.
    leg: index into LEG_ORDER.
    Returns (..., 3) foot position in the base frame.
    """
    q = np.asarray(q, dtype=np.float64)
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    d = HIP_LATERAL[leg]

    # Two-link arm in the sagittal plane (x forward, z up), zero = straight
    # down, positive pitch swings the foot backward.
    x = -THIGH_LENGTH * np.sin(q2) - CALF_LENGTH * np.sin(q2 + q3)
    z = -THIGH_LENGTH * np.cos(q2) - CALF_LENGTH * np.cos(q2 + q3)

    # Abduction rotates the (y=d, z) pair about the hip x-axis.
    y_out = d * np.cos(q1) - z * np.sin(q1)
    z_out = d * np.sin(q1) + z * np.cos(q1)

    foot = np.stack([x, y_out, z_out], axis=-1)
    return foot + HIP_OFFSETS[leg]


def leg_ik(foot_pos, leg):
    """Joint angles reaching a foot position for one leg.

    foot_pos: (..., 3) target in the base frame.
    leg: index into LEG_ORDER.
    Returns (..., 3) angles (hip, thigh, calf), knee-backward branch.

    Targets outside the reachable workspace are clamped to it first (closest
    reachable point along the leg-plane direction), so the result is always
    finite; use clamp_to_limits() afterwards to detect joint-limit violations.
    """
    p = np.asarray(foot_pos, dtype=np.float64) - HIP_OFFSETS[leg]
    d = HIP_LATERAL[leg]
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]

    # Abduction: in the y-z plane the foot is the point (d, -L) rotated by q1,
    # where L >= 0 is the in-plane distance below the hip axis. A target
    # closer to the hip x-axis than |d| is unreachable; clamp it onto the
    # cylinder of radius |d|.
    r = np.maximum(np.hypot(py, pz), np.abs(d) + _REACH_EPS)
    leg_plane_dist = np.sqrt(r**2 - d**2)
    q1 = np.arctan2(pz, py) - np.arctan2(-leg_plane_dist, d)
    q1 = _wrap_pi(q1)

    # Clamp the in-plane target to the workspace boundary.
    rho = np.hypot(px, leg_plane_dist)
    scale = np.where(rho > MAX_REACH - _REACH_EPS, (MAX_REACH - _REACH_EPS) / rho, 1.0)
    px = px * scale
    leg_plane_dist = leg_plane_dist * scale

    # Planar two-link arm: knee from the law of cosines (negative branch),
    # thigh from the target direction minus the interior angle.
    cos_knee = (px**2 + leg_plane_dist**2 - THIGH_LENGTH**2 - CALF_LENGTH**2) / (
        2.0 * THIGH_LENGTH * CALF_LENGTH
    )
    q3 = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
    q2 = np.arctan2(-px, leg_plane_dist) - np.arctan2(
        CALF_LENGTH * np.sin(q3), THIGH_LENGTH + CALF_LENGTH * np.cos(q3)
    )
    # The thigh range extends past pi (e.g. rear: [-0.52, 4.54]); prefer the
    # 2*pi-equivalent angle that falls inside the limits.
    lo, hi = JOINT_LIMITS[leg, 1]
    q2 = _prefer_in_range(_wrap_pi(q2), lo, hi)

    return np.stack([q1, q2, q3], axis=-1)


def fk(dof_pos):
    """Foot positions for all four legs.

    dof_pos: (..., 12) angles, order FR, FL, RR, RL x (hip, thigh, calf).
    Returns (..., 4, 3) foot positions in the base frame.
    """
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    q = dof_pos.reshape(*dof_pos.shape[:-1], 4, 3)
    return np.stack([leg_fk(q[..., leg, :], leg) for leg in range(4)], axis=-2)


def ik(foot_pos):
    """Joint angles for all four legs.

    foot_pos: (..., 4, 3) foot targets in the base frame, leg order FR, FL, RR, RL.
    Returns (..., 12) angles.
    """
    foot_pos = np.asarray(foot_pos, dtype=np.float64)
    q = np.stack([leg_ik(foot_pos[..., leg, :], leg) for leg in range(4)], axis=-2)
    return q.reshape(*foot_pos.shape[:-2], 12)


def clamp_to_limits(dof_pos):
    """Clip angles to the MJCF joint limits.

    dof_pos: (..., 12) angles.
    Returns (clamped, violated): same-shape clipped angles and a bool mask of
    which entries were outside their limits (for the clamp-rate report).
    """
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    lo = JOINT_LIMITS[:, :, 0].reshape(12)
    hi = JOINT_LIMITS[:, :, 1].reshape(12)
    clamped = np.clip(dof_pos, lo, hi)
    violated = (dof_pos < lo) | (dof_pos > hi)
    return clamped, violated


def _wrap_pi(angle):
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _prefer_in_range(angle, lo, hi):
    """Shift by 2*pi toward [lo, hi] when the wrapped angle falls outside it."""
    two_pi = 2.0 * np.pi
    angle = np.where((angle < lo) & (angle + two_pi <= hi), angle + two_pi, angle)
    angle = np.where((angle > hi) & (angle - two_pi >= lo), angle - two_pi, angle)
    return angle
