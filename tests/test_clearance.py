"""Unit tests for kinematic clearance projection (retarget/clearance.py)."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retarget import clearance as cl  # noqa: E402
from retarget import ik  # noqa: E402
from retarget.postprocess import _apply_per_leg  # noqa: E402

FPS = 50.0


def _motion_from_feet(root_z, foot_dx=0.0, n=100):
    """Clip with root at root_z and feet at ground height (world z 0.022),
    displaced foot_dx forward of the hips — a forward crouch drives the
    knee-backward IK branch down toward the floor."""
    root_pos = np.zeros((n, 3))
    root_pos[:, 0] = np.linspace(0, 0.2, n)  # quasi-static: pose-gate compatible
    root_pos[:, 2] = root_z
    root_rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))  # identity xyzw
    rot = Rotation.from_quat(root_rot)
    feet_base = ik.HIP_OFFSETS[None, :, :] + np.array([foot_dx, 0.0, 0.022 - root_z])
    dof, _ = ik.clamp_to_limits(ik.ik(np.tile(feet_base, (n, 1, 1))))
    return {
        "fps": FPS,
        "num_frames": n,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof,
        "foot_contacts": np.ones((n, 4), dtype=bool),
        "source": "synthetic",
    }, rot


def _knees_world(motion):
    rot = Rotation.from_quat(motion["root_rot"])
    return (
        motion["root_pos"][:, None, :]
        + _apply_per_leg(rot, cl.knee_base_positions(motion["dof_pos"]))
    )[..., 2]


def _feet_world(motion):
    rot = Rotation.from_quat(motion["root_rot"])
    return motion["root_pos"][:, None, :] + _apply_per_leg(rot, ik.fk(motion["dof_pos"]))


def test_clear_pose_is_identity():
    motion, _ = _motion_from_feet(root_z=0.32)
    out, rep = cl.project_clearance(motion)
    assert rep["frames_below"] == 0
    np.testing.assert_array_equal(out["dof_pos"], motion["dof_pos"])
    np.testing.assert_array_equal(out["root_pos"], motion["root_pos"])


def test_mild_case_lift_only_feet_preserved():
    # shallow forward crouch: knees a few cm under, fixable inside the lift cap
    motion, _ = _motion_from_feet(root_z=0.20, foot_dx=0.22)
    assert 0 < cl.CLEARANCE_Z - _knees_world(motion).min() < cl.MAX_LIFT  # premise
    out, rep = cl.project_clearance(motion)
    assert _knees_world(out).min() >= cl.CLEARANCE_Z - 1e-4
    assert rep["raise_fraction"] < 1e-6  # lift alone was enough
    np.testing.assert_allclose(_feet_world(out), _feet_world(motion), atol=1e-4)


def test_deep_case_thigh_raise_engages_and_clears():
    # deep forward crouch: penetration exceeds what the capped lift can fix
    motion, _ = _motion_from_feet(root_z=0.12, foot_dx=0.28)
    assert cl.CLEARANCE_Z - _knees_world(motion).min() > cl.MAX_LIFT  # premise
    out, rep = cl.project_clearance(motion)
    assert _knees_world(out).min() >= cl.CLEARANCE_Z - 1e-3
    assert rep["residual_depth"] < 1e-3
    assert rep["max_lift"] <= cl.MAX_LIFT + 1e-9
    assert rep["raise_fraction"] > 0.5  # the thigh raise did the rest
    # feet stay near the original targets (the calf projects the foot onto
    # its reachable circle, so a bounded shift is expected, not exactness)
    assert rep["foot_shift_max"] < 0.30


def test_mixed_clip_touches_only_where_needed():
    good, _ = _motion_from_feet(root_z=0.32, n=60)
    bad, _ = _motion_from_feet(root_z=0.12, foot_dx=0.28, n=60)
    motion = dict(good)
    motion["num_frames"] = 120
    for k in ("root_pos", "dof_pos", "foot_contacts"):
        motion[k] = np.concatenate([good[k], bad[k]])
    motion["root_rot"] = np.concatenate([good["root_rot"], bad["root_rot"]])
    out, rep = cl.project_clearance(motion)
    # early clear frames (beyond the smoothing kernel) are untouched
    np.testing.assert_array_equal(out["dof_pos"][:35], motion["dof_pos"][:35])
    np.testing.assert_array_equal(out["root_pos"][:35], motion["root_pos"][:35])
    assert _knees_world(out).min() >= cl.CLEARANCE_Z - 1e-3


def test_transient_dip_is_left_alone():
    # a 0.4 s penetration dip inside a clear clip is a swing artifact, not a
    # pose: the duration gate must leave the whole clip untouched
    good, _ = _motion_from_feet(root_z=0.32, n=200)
    bad, _ = _motion_from_feet(root_z=0.12, foot_dx=0.28, n=20)
    motion = dict(good)
    motion["num_frames"] = 200
    for k in ("root_pos", "dof_pos"):
        motion[k] = np.concatenate([good[k][:90], bad[k], good[k][110:]])
    out, rep = cl.project_clearance(motion)
    assert rep["frames_below"] == 20 and rep["pose_frames"] == 0
    np.testing.assert_array_equal(out["dof_pos"], motion["dof_pos"])
    np.testing.assert_array_equal(out["root_pos"], motion["root_pos"])
