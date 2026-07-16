"""Unit tests for the analytic Go2 leg FK/IK (retarget/ik.py).

Covers the acceptance criterion FK(IK(p)) < 1 mm on random reachable targets
for all four legs, plus cross-checks of the hardcoded geometry and of the
analytic FK against MuJoCo's own kinematics.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retarget import ik  # noqa: E402

GO2_XML = REPO_ROOT / "assets" / "unitree_go2" / "go2.xml"

RNG = np.random.default_rng(0)
NUM_SAMPLES = 2000


def _model():
    return mujoco.MjModel.from_xml_path(str(GO2_XML))


def _sample_dof_pos(n, margin=1e-3):
    """Random angles strictly inside the joint limits, (n, 12)."""
    lo = ik.JOINT_LIMITS[:, :, 0].reshape(12) + margin
    hi = ik.JOINT_LIMITS[:, :, 1].reshape(12) - margin
    return RNG.uniform(lo, hi, size=(n, 12))


def test_geometry_matches_mjcf():
    model = _model()
    for leg, name in enumerate(ik.LEG_ORDER):
        np.testing.assert_allclose(model.body(f"{name}_hip").pos, ik.HIP_OFFSETS[leg])
        thigh_pos = model.body(f"{name}_thigh").pos
        np.testing.assert_allclose(thigh_pos, [0.0, ik.HIP_LATERAL[leg], 0.0])
        calf_pos = model.body(f"{name}_calf").pos
        np.testing.assert_allclose(calf_pos, [0.0, 0.0, -ik.THIGH_LENGTH])
        for j, suffix in enumerate(["hip", "thigh", "calf"]):
            jnt_range = model.joint(f"{name}_{suffix}_joint").range
            np.testing.assert_allclose(jnt_range, ik.JOINT_LIMITS[leg, j])


def test_fk_matches_mujoco():
    """Analytic FK == MuJoCo FK of the calf endpoint, all legs, random angles."""
    model = _model()
    data = mujoco.MjData(model)
    dof_pos = _sample_dof_pos(200)
    foot_local = np.array([0.0, 0.0, -ik.CALF_LENGTH])

    # The MJCF declares legs in FL, FR, RL, RR order, so canonical dof vectors
    # must be scattered through the joint qpos addresses, not copied as a block.
    qpos_adr = np.array([
        model.joint(f"{name}_{suffix}_joint").qposadr[0]
        for name in ik.LEG_ORDER
        for suffix in ["hip", "thigh", "calf"]
    ])

    for q in dof_pos:
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity: world frame == base frame
        data.qpos[qpos_adr] = q
        mujoco.mj_forward(model, data)
        expected = np.empty((4, 3))
        for leg, name in enumerate(ik.LEG_ORDER):
            calf = data.body(f"{name}_calf")
            expected[leg] = calf.xpos + calf.xmat.reshape(3, 3) @ foot_local
        np.testing.assert_allclose(ik.fk(q), expected, atol=1e-9)


def test_fk_ik_roundtrip_reachable_targets():
    """FK(IK(p)) within 1 mm for random reachable targets (acceptance §10)."""
    for leg in range(4):
        # Reachable by construction: FK of random in-limit angles.
        q = _sample_dof_pos(NUM_SAMPLES).reshape(-1, 4, 3)[:, leg, :]
        targets = ik.leg_fk(q, leg)
        recovered = ik.leg_fk(ik.leg_ik(targets, leg), leg)
        err = np.linalg.norm(recovered - targets, axis=-1)
        assert err.max() < 1e-3, f"leg {ik.LEG_ORDER[leg]}: max error {err.max():.2e} m"


def test_ik_fk_angle_roundtrip():
    """IK recovers the exact angles when the foot is below the hip axis.

    The knee-backward branch plus foot-below-hip makes the solution unique, so
    IK(FK(q)) == q there. Configurations with the foot swung above the hip
    axis map to the mirrored abduction branch and are excluded.
    """
    for leg in range(4):
        q = _sample_dof_pos(NUM_SAMPLES).reshape(-1, 4, 3)[:, leg, :]
        below = (
            ik.THIGH_LENGTH * np.cos(q[:, 1])
            + ik.CALF_LENGTH * np.cos(q[:, 1] + q[:, 2])
        ) > 0.02
        q = q[below]
        assert len(q) > 100
        recovered = ik.leg_ik(ik.leg_fk(q, leg), leg)
        np.testing.assert_allclose(recovered, q, atol=1e-9)


def test_home_keyframe():
    """IK of the home stance recovers the home keyframe angles."""
    model = _model()
    home_q = model.key("home").qpos[7:].copy()
    feet = ik.fk(home_q)
    # Home: feet directly below the hips, trunk 0.27 m up -> feet slightly
    # below z=-0.26 in the base frame.
    np.testing.assert_allclose(feet[:, 0], ik.HIP_OFFSETS[:, 0], atol=1e-9)
    assert np.all(feet[:, 2] < -0.26)
    np.testing.assert_allclose(ik.ik(feet), home_q, atol=1e-9)


def test_unreachable_targets_are_clamped():
    """Far / degenerate targets give finite angles whose FK is in-workspace."""
    targets = np.array([
        [1.0, 1.0, -1.0],     # far outside reach
        [0.1934, -0.0465, 0.0],  # exactly at the FR hip joint
        [0.0, 0.0, 0.0],      # base origin
        [0.2, -0.05, 0.5],    # above the body
    ])
    for leg in range(4):
        q = ik.leg_ik(targets, leg)
        assert np.all(np.isfinite(q))
        feet = ik.leg_fk(q, leg) - ik.HIP_OFFSETS[leg]
        # Reach is bounded inside the leg plane, which sits HIP_LATERAL away
        # from the hip joint: ||foot||^2 = rho^2 + d^2 with rho <= MAX_REACH.
        rho = np.sqrt(np.linalg.norm(feet, axis=-1) ** 2 - ik.HIP_LATERAL[leg] ** 2)
        assert np.all(rho <= ik.MAX_REACH + 1e-9)


def test_clamp_to_limits():
    q = np.zeros(12)  # calf zero is outside its range [-2.72, -0.84]
    clamped, violated = ik.clamp_to_limits(q)
    calf_idx = np.arange(2, 12, 3)
    assert violated[calf_idx].all()
    assert not np.delete(violated, calf_idx).any()
    np.testing.assert_allclose(clamped[calf_idx], ik.JOINT_LIMITS[:, 2, 1])
    inside, violated = ik.clamp_to_limits(clamped)
    np.testing.assert_allclose(inside, clamped)
    assert not violated.any()
