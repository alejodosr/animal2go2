"""Seam validator: passes real clips, catches the silent killers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.seam import canonical_points, load_and_validate, validate_keypoints

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
GOLDEN = ["D1_007_KAN01_001.npz", "D1_009_KAN01_002.npz"]  # walk, trot


@pytest.fixture(scope="module")
def kp():
    path = PROCESSED / GOLDEN[0]
    if not path.exists():
        pytest.skip(f"{path} missing")
    return load_and_validate(path)


def test_golden_clips_pass():
    for name in GOLDEN:
        path = PROCESSED / name
        if not path.exists():
            pytest.skip(f"{path} missing")
        load_and_validate(path)


def test_canonical_points_shape(kp):
    pts = canonical_points(kp)
    assert pts.shape == (int(kp["num_frames"]), 10, 3)
    np.testing.assert_array_equal(pts[:, 0], kp["root_pos"])
    np.testing.assert_array_equal(pts[:, 6], kp["toe_pos"][:, 0])


def test_missing_key_fails(kp):
    d = dict(kp)
    del d["toe_pos"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_keypoints(d)


def test_fr_fl_mount_swap_fails(kp):
    d = dict(kp)
    d["leg_root_pos"] = kp["leg_root_pos"][:, [1, 0, 2, 3]]
    d["toe_pos"] = kp["toe_pos"][:, [1, 0, 2, 3]]
    with pytest.raises(ValueError, match="FR/FL swap"):
        validate_keypoints(d)


def test_front_rear_swap_fails(kp):
    d = dict(kp)
    d["leg_root_pos"] = kp["leg_root_pos"][:, [2, 3, 0, 1]]
    d["toe_pos"] = kp["toe_pos"][:, [2, 3, 0, 1]]
    with pytest.raises(ValueError, match="front"):
        validate_keypoints(d)


def test_toe_only_swap_fails(kp):
    d = dict(kp)
    d["toe_pos"] = kp["toe_pos"][:, [1, 0, 2, 3]]
    with pytest.raises(ValueError, match="toe"):
        validate_keypoints(d)


def test_centimeters_fail(kp):
    d = dict(kp)
    for k in ("root_pos", "chest_pos", "toe_pos", "leg_root_pos"):
        d[k] = kp[k] * 100.0
    with pytest.raises(ValueError, match="units"):
        validate_keypoints(d)


def test_y_up_fails(kp):
    d = dict(kp)
    for k in ("root_pos", "chest_pos", "toe_pos", "leg_root_pos"):
        d[k] = kp[k][..., [0, 2, 1]]  # swap y/z: ground plane becomes x-z
    with pytest.raises(ValueError):
        validate_keypoints(d)


def test_bad_quaternion_fails(kp):
    d = dict(kp)
    d["root_rot_xyzw"] = kp["root_rot_xyzw"] * 1.5
    with pytest.raises(ValueError, match="unit-norm"):
        validate_keypoints(d)


def test_nan_fails(kp):
    d = dict(kp)
    bad = kp["root_pos"].copy()
    bad[3, 1] = np.nan
    d["root_pos"] = bad
    with pytest.raises(ValueError, match="NaN"):
        validate_keypoints(d)
