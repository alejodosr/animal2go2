"""Unit tests for support-aware re-grounding (retarget/reground.py)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retarget import reground as rg  # noqa: E402

FPS = 50.0


def _feet_z(n, base=0.022):
    return np.full((n, 4), base)


def test_flat_clip_is_identity():
    feet = _feet_z(200)
    offset, segments = rg.support_offset(feet, FPS)
    assert segments == []
    assert (offset == 0).all()


def test_short_flight_untouched():
    # a 300 ms real flight: feet up to 0.12 m, but too short to be support
    feet = _feet_z(200)
    feet[100:115] = 0.12
    offset, segments = rg.support_offset(feet, FPS)
    assert segments == []
    assert (offset == 0).all()


def test_platform_detected_and_grounded():
    # stand on a 0.12 m platform for 1.5 s
    feet = _feet_z(300)
    feet[100:175] = 0.12 + 0.022
    offset, segments = rg.support_offset(feet, FPS)
    assert len(segments) == 1
    s, e = segments[0]
    assert abs(s - 100) <= 2 and abs(e - 175) <= 2
    # deep inside the segment (>=4 sigma from the edges) the lowest foot is
    # put exactly on the ground; closer to the edges the ramp is already active
    np.testing.assert_allclose(offset[125:155], 0.12, atol=1e-4)
    # far from the segment nothing moves
    assert (offset[:80] == 0).all() and (offset[200:] == 0).all()
    # ramps are monotone into and out of the segment
    assert (np.diff(offset[85:120]) >= -1e-12).all()
    assert (np.diff(offset[160:195]) <= 1e-12).all()


def test_offset_never_exceeds_clearance():
    # offset may never push a foot below ground, even at ramp edges
    feet = _feet_z(300)
    feet[100:175] = 0.12 + 0.022
    offset, _ = rg.support_offset(feet, FPS)
    clearance = np.maximum(0.0, feet.min(axis=1) - rg.FOOT_RADIUS)
    assert (offset <= clearance + 1e-12).all()
