"""Projection harness: camera geometry + render/reproject round trip."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.camera import PinholeCamera
from v2k.render import KEYPOINT_COLORS, default_camera, render_frame_array
from v2k.seam import KEYPOINT_NAMES, canonical_points, load_and_validate

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_look_at_center_projection():
    cam = PinholeCamera.look_at(eye=(0, -3, 1), target=(0, 0, 1),
                                width=640, height=480)
    uv, depth = cam.project(np.array([0.0, 0.0, 1.0]))
    assert depth == pytest.approx(3.0)
    assert uv[0] == pytest.approx(320.0)
    assert uv[1] == pytest.approx(240.0)


def test_look_at_axes_orientation():
    """World +Z (up) must project above center; camera-right maps right."""
    cam = PinholeCamera.look_at(eye=(0, -3, 1), target=(0, 0, 1),
                                width=640, height=480)
    up_uv, _ = cam.project(np.array([0.0, 0.0, 1.5]))
    assert up_uv[1] < 240.0  # smaller v = higher in image
    # Camera looks along +Y; world +X is to its... check consistency instead:
    right_uv, _ = cam.project(np.array([0.5, 0.0, 1.0]))
    left_uv, _ = cam.project(np.array([-0.5, 0.0, 1.0]))
    assert abs(right_uv[0] - 320) == pytest.approx(abs(left_uv[0] - 320), abs=1e-6)
    assert right_uv[0] != left_uv[0]


def test_params_round_trip():
    cam = PinholeCamera.look_at(eye=(1, -2, 1.5), target=(0.2, 0.1, 0.4))
    cam2 = PinholeCamera.from_params(cam.params())
    pts = np.random.default_rng(0).uniform(-1, 1, size=(20, 3))
    uv1, d1 = cam.project(pts)
    uv2, d2 = cam2.project(pts)
    np.testing.assert_allclose(uv1, uv2)
    np.testing.assert_allclose(d1, d2)


def _detect_markers(frame, tol=40, min_px=20):
    """Centroid of each keypoint's unique marker color; NaN if not found."""
    out = np.full((len(KEYPOINT_COLORS), 2), np.nan)
    img = frame.astype(np.int16)
    for i, c in enumerate(KEYPOINT_COLORS):
        mask = (np.abs(img - c.astype(np.int16)) < tol).all(axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) >= min_px:
            out[i] = xs.mean(), ys.mean()
    return out


def test_render_grid_round_trip():
    """Phase 0 gate, geometry half: with no occlusion every marker centroid
    must land on its input 2D position (catches y-flip / dpi / offset bugs)."""
    w, h = 960, 540
    gx, gy = np.meshgrid(np.linspace(150, w - 150, 5), np.linspace(120, h - 120, 2))
    uv = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    frame, fig_state = render_frame_array(uv, np.ones(10), w, h)
    det = _detect_markers(frame)
    assert not np.isnan(det).any(), "not all 10 markers detected on the grid"
    err = np.linalg.norm(det - uv, axis=-1)
    assert err.max() < 1.0, f"max grid reprojection err {err.max():.2f} px"


@pytest.mark.parametrize("view", ["side", "three_quarter"])
def test_render_clip_round_trip(view):
    """Phase 0 gate, clip half: on a real clip, every cleanly visible marker
    reprojects onto the mocap 2D; occluded markers may drop out (crouch
    frames stack all 10 points into a few dozen pixels)."""
    path = PROCESSED / "D1_007_KAN01_001.npz"
    if not path.exists():
        pytest.skip(f"{path} missing")
    kp = load_and_validate(path)
    cam = default_camera(kp, view=view, width=1280, height=720)
    pts = canonical_points(kp)
    uv, depth = cam.project(pts)

    frames = np.linspace(0, len(uv) - 1, 5).astype(int)
    errs, coverage, fig_state = [], np.zeros(10, dtype=bool), None
    for f in frames:
        frame, fig_state = render_frame_array(uv[f], depth[f],
                                              cam.width, cam.height, fig_state)
        det = _detect_markers(frame, min_px=180)  # full cores only (~200 px)
        found = ~np.isnan(det[:, 0])
        coverage |= found
        errs.append(np.linalg.norm(det[found] - uv[f][found], axis=-1))
    err = np.concatenate(errs)
    assert coverage.sum() >= 7, f"only {coverage.sum()}/10 keypoints ever visible"
    assert len(err) >= 15, "too few visible markers to score"
    assert np.median(err) < 1.5, f"median reprojection err {np.median(err):.2f} px"
    assert err.max() < 4.0, f"max reprojection err {err.max():.2f} px"
