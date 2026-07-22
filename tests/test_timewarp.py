"""Unit tests for the speed-aware time warp (retarget/timewarp.py)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retarget import timewarp as tw  # noqa: E402

FPS = 50.0


def _motion(speed, n=300, yaw_rate=0.0):
    """Straight-line clip at constant planar speed with a gait-like dof wave."""
    t = np.arange(n) / FPS
    root_pos = np.stack([speed * t, np.zeros(n), np.full(n, 0.32)], axis=1)
    yaw = yaw_rate * t
    root_rot = np.stack(
        [np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)], axis=1
    )  # xyzw
    dof_pos = 0.5 * np.sin(2 * np.pi * 2.0 * t)[:, None] * np.ones((1, 12))
    contacts = np.zeros((n, 4), dtype=bool)
    contacts[(np.arange(n) // 10) % 2 == 0] = True  # 0.2 s stance / 0.2 s swing
    return {
        "fps": FPS,
        "num_frames": n,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "foot_contacts": contacts,
        "source": "synthetic",
    }


def test_slow_motion_passes_through():
    motion = _motion(speed=1.0)
    out, report = tw.timewarp(motion)
    assert report["min_rate"] == 1.0
    assert report["slowed_fraction"] == 0.0
    assert out["num_frames"] == motion["num_frames"]
    np.testing.assert_allclose(out["root_pos"], motion["root_pos"], atol=1e-9)
    np.testing.assert_allclose(out["dof_pos"], motion["dof_pos"], atol=1e-9)
    np.testing.assert_array_equal(out["foot_contacts"], motion["foot_contacts"])


def test_fast_motion_slowed_to_cap():
    motion = _motion(speed=4.5)
    out, report = tw.timewarp(motion, cap=3.2)
    # steady-state interior speed lands on the cap (edges see filter ramps)
    vel = np.gradient(out["root_pos"][:, :2], axis=0) * FPS
    speed = np.linalg.norm(vel, axis=-1)
    interior = speed[len(speed) // 4 : -len(speed) // 4]
    assert np.all(interior < 3.2 * 1.05)
    assert np.all(interior > 3.2 * 0.9)
    # duration stretches by ~speed/cap
    stretch = report["duration_after"] / report["duration_before"]
    assert 0.9 * (4.5 / 3.2) < stretch < 1.1 * (4.5 / 3.2)
    # dof velocities scale down with the same factor
    assert report["dof_vel_peak_after"] < report["dof_vel_peak_before"] * 0.85


def test_warp_preserves_path_and_endpoints():
    motion = _motion(speed=4.5, yaw_rate=0.5)
    out, _ = tw.timewarp(motion)
    # same geometric path: endpoints match, no frame is dropped
    np.testing.assert_allclose(out["root_pos"][0], motion["root_pos"][0], atol=1e-6)
    np.testing.assert_allclose(out["root_pos"][-1], motion["root_pos"][-1], atol=1e-3)
    q = out["root_rot"]
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-9)
    # yaw is monotone in the source; the warp must keep it monotone (the final
    # grid sample clamps onto the last source frame, so allow a zero step)
    yaw = np.unwrap(2 * np.arctan2(q[:, 2], q[:, 3]))
    assert np.all(np.diff(yaw) >= -1e-9)


def test_contact_duty_fraction_preserved():
    motion = _motion(speed=4.5)
    _, report = tw.timewarp(motion)
    assert abs(report["contact_fraction_after"] - report["contact_fraction_before"]) < 0.05


def test_burst_only_slows_locally():
    """A clip that is slow except for a middle burst only stretches the middle."""
    n = 400
    t = np.arange(n) / FPS
    speed = np.where((t > 3.0) & (t < 5.0), 4.5, 1.0)
    x = np.concatenate([[0.0], np.cumsum(speed[:-1] / FPS)])
    motion = _motion(speed=1.0, n=n)
    motion["root_pos"][:, 0] = x
    out, report = tw.timewarp(motion)
    assert report["slowed_fraction"] < 0.5  # only the burst neighborhood
    # slow head of the clip is untouched (rate 1 => identity resample there)
    np.testing.assert_allclose(
        out["root_pos"][:100], motion["root_pos"][:100], atol=1e-6
    )
