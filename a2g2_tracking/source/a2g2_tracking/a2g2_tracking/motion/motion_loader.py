"""§7 motion pkl → torch tensors, with every convention fix in one place.

This module is deliberately free of Isaac Lab imports: it is pure torch/numpy
so it can be unit-tested without launching the sim, and so nothing
Isaac-specific leaks into the motion representation.

Conventions handled HERE and nowhere else:
  - quaternion xyzw (§7 pkl) → wxyz (Isaac Lab)
  - dof order canonical FR, FL, RR, RL × (hip, thigh, calf) → the simulator's
    joint order, mapped by *name* via ``make_dof_index_map``
  - global ground z-offset between the MuJoCo-aligned pkls and the Isaac Go2
    collision geometry (``GROUND_Z_OFFSET``, measured in Phase 2)
  - velocities by central finite differences (quaternion log-map for angular
    velocity); the pkls were already smoothed in Milestone 1, so no smoothing
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Canonical leg order of the §7 pkls (video2robot / Milestone 1).
LEG_ORDER = ("FR", "FL", "RR", "RL")

# Canonical dof names in pkl order: FR, FL, RR, RL × (hip, thigh, calf).
CANONICAL_DOF_NAMES = tuple(
    f"{leg}_{part}_joint" for leg in LEG_ORDER for part in ("hip", "thigh", "calf")
)

# Constant z-shift applied to root_pos so the pkls (z-aligned to the MuJoCo
# foot sphere) sit on the Isaac Go2 collision geometry. Measured in Phase 2:
# the foot collider is a sphere of radius 0.022 m, but a PD-held drop test
# settles with the foot center at z = 0.0239 m (PhysX effective contact
# surface sits ~2 mm above the geometric radius). The pkls replay stance-foot
# centers at a 0.0240 m median — aligned to 0.1 mm, so no shift is needed.
GROUND_Z_OFFSET = 0.0

# Empirical resting height of the foot link origin on Isaac's flat ground
# (settle test, Phase 2). The replay gate compares stance feet against this.
FOOT_REST_CENTER_Z = 0.0239


def make_dof_index_map(joint_names: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Permutation from canonical dof order to the simulator's joint order.

    Args:
        joint_names: the simulator's joint names in its own order — for Isaac
            Lab, ``Articulation.joint_names``, the single source of truth.

    Returns:
        Long tensor ``perm`` of shape (12,) such that
        ``sim_vec = canonical_vec[..., perm]``.
    """
    if len(joint_names) != 12:
        raise ValueError(f"expected 12 joint names, got {len(joint_names)}: {joint_names}")
    if set(joint_names) != set(CANONICAL_DOF_NAMES):
        raise ValueError(
            f"joint names do not match canonical names.\n  got: {sorted(joint_names)}\n"
            f"  expected: {sorted(CANONICAL_DOF_NAMES)}"
        )
    canonical_index = {name: i for i, name in enumerate(CANONICAL_DOF_NAMES)}
    return torch.tensor([canonical_index[name] for name in joint_names], dtype=torch.long)


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) xyzw → wxyz."""
    return q[..., [3, 0, 1, 2]]


# -- small wxyz quaternion helpers (batched over leading dims) ----------------


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_log_map(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle, rad) of a unit wxyz quaternion."""
    # shortest arc: the double cover q/-q is the same rotation
    q = torch.where(q[..., :1] < 0, -q, q)
    v_norm = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, q[..., :1])
    axis = q[..., 1:] / v_norm.clamp(min=1e-12)
    return torch.where(v_norm > 1e-8, axis * angle, 2.0 * q[..., 1:])


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Batched slerp between wxyz quaternions; ``blend`` broadcastable (..., 1)."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    # fall back to lerp for nearly-parallel quaternions
    use_lerp = sin_theta < 1e-6
    w0 = torch.where(use_lerp, 1.0 - blend, torch.sin((1.0 - blend) * theta) / sin_theta.clamp(min=1e-12))
    w1 = torch.where(use_lerp, blend, torch.sin(blend * theta) / sin_theta.clamp(min=1e-12))
    return quat_normalize(w0 * q0 + w1 * q1)


# -- loading ------------------------------------------------------------------


@dataclass
class MotionClip:
    """One retargeted clip, conventions already fixed for the simulator.

    All tensors live on the target device. Quaternions are wxyz. Dof vectors
    are in the simulator's joint order (per the ``joint_names`` passed to
    ``load_motion``). Foot contacts stay in canonical leg order FR, FL, RR, RL
    — they are matched by name to sensor bodies, never by dof position.
    Velocities are world-frame.
    """

    name: str
    fps: float
    num_frames: int
    root_pos: torch.Tensor  # (N, 3)
    root_rot: torch.Tensor  # (N, 4) wxyz, sign-continuous
    dof_pos: torch.Tensor  # (N, 12) sim joint order
    root_lin_vel: torch.Tensor  # (N, 3)
    root_ang_vel: torch.Tensor  # (N, 3)
    dof_vel: torch.Tensor  # (N, 12)
    foot_contacts: torch.Tensor  # (N, 4) float 0/1, canonical FR, FL, RR, RL


def _central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Central differences over dim 0, one-sided at the endpoints."""
    v = torch.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def _quat_ang_vel(q: torch.Tensor, dt: float) -> torch.Tensor:
    """World-frame angular velocity from a wxyz quaternion trajectory.

    Central log-map differences: ω(t) ≈ log(q_{t+1} ⊗ q_{t-1}⁻¹) / (2 dt).
    """
    def rel_rotvec(q_to: torch.Tensor, q_from: torch.Tensor) -> torch.Tensor:
        return quat_log_map(quat_mul(q_to, quat_conjugate(q_from)))

    w = torch.empty(q.shape[0], 3, dtype=q.dtype, device=q.device)
    w[1:-1] = rel_rotvec(q[2:], q[:-2]) / (2.0 * dt)
    w[0] = rel_rotvec(q[1:2], q[0:1]) / dt
    w[-1] = rel_rotvec(q[-1:], q[-2:-1]) / dt
    return w


def load_motion(
    path: str | Path,
    joint_names: list[str] | tuple[str, ...],
    device: str | torch.device = "cpu",
    z_offset: float = GROUND_Z_OFFSET,
    dtype: torch.dtype = torch.float32,
) -> MotionClip:
    """Load a §7 pkl and fix every convention. See module docstring.

    Args:
        path: pkl file in §7 format.
        joint_names: simulator joint names (``Articulation.joint_names``).
        device: target device for all tensors.
        z_offset: constant added to root z (ground alignment, Phase 2).
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    fps = float(data["fps"])
    dt = 1.0 / fps
    perm = make_dof_index_map(joint_names)

    root_pos = torch.as_tensor(np.asarray(data["root_pos"]), dtype=dtype)
    root_rot = torch.as_tensor(np.asarray(data["root_rot"]), dtype=dtype)
    dof_pos = torch.as_tensor(np.asarray(data["dof_pos"]), dtype=dtype)

    n = root_pos.shape[0]
    if not (root_rot.shape == (n, 4) and dof_pos.shape == (n, 12)):
        raise ValueError(f"{path.name}: inconsistent shapes {root_pos.shape} {root_rot.shape} {dof_pos.shape}")
    if "foot_contacts" not in data:
        raise ValueError(
            f"{path.name}: no foot_contacts — re-export with Milestone 1 stance segments; "
            "the contact-matching reward needs them"
        )
    foot_contacts = torch.as_tensor(np.asarray(data["foot_contacts"]), dtype=dtype)

    # conventions: xyzw → wxyz, canonical dof order → sim order, ground offset
    root_rot = quat_normalize(quat_xyzw_to_wxyz(root_rot))
    dof_pos = dof_pos[:, perm]
    root_pos = root_pos.clone()
    root_pos[:, 2] += z_offset

    # sign continuity across the double cover, else finite diffs and slerp
    # see a spurious 2π flip
    flip = ((root_rot[1:] * root_rot[:-1]).sum(dim=-1) < 0).to(dtype)
    sign = torch.cat([torch.ones(1, dtype=dtype), 1.0 - 2.0 * (torch.cumsum(flip, dim=0) % 2)])
    root_rot = root_rot * sign.unsqueeze(-1)

    clip = MotionClip(
        name=str(data.get("source", path.stem)),
        fps=fps,
        num_frames=n,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        root_lin_vel=_central_diff(root_pos, dt),
        root_ang_vel=_quat_ang_vel(root_rot, dt),
        dof_vel=_central_diff(dof_pos, dt),
        foot_contacts=foot_contacts,
    )
    for field in ("root_pos", "root_rot", "dof_pos", "root_lin_vel", "root_ang_vel", "dof_vel", "foot_contacts"):
        t = getattr(clip, field)
        if not torch.isfinite(t).all():
            raise ValueError(f"{path.name}: non-finite values in {field}")
        setattr(clip, field, t.to(device))
    return clip
