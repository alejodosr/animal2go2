"""Phase 1: the canonical keypoint map, the temporal filter, and the scorer.

The model itself is not exercised here — it needs the SSD perception venv and a
GPU. What is exercised is everything that can silently corrupt the seam without
throwing: mapping bodyparts by index instead of by name, an FR/FL transposition,
low-confidence frames turning into (0, 0), and a scorer that reports success on
frames the detector never saw.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.detect2d import (
    DEFAULT_MIN_CONF,
    SUPERANIMAL_QUADRUPED_MAP,
    gt_bboxes,
    smooth_2d,
    to_canonical,
)
from v2k.eval2d import gate, score, trunk_px
from v2k.seam import KEYPOINT_NAMES

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"
GOLDEN_GT = SYNTHETIC / "D1_007_KAN01_001_side_gt2d.npz"

# The 39 SuperAnimal-Quadruped bodyparts, in the order the model emits them.
SUPERANIMAL_BODYPARTS = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base",
    "right_antler_end", "left_eye", "left_earbase", "left_earend",
    "left_antler_base", "left_antler_end", "neck_base", "neck_end",
    "throat_base", "throat_end", "back_base", "back_end", "back_middle",
    "tail_base", "tail_end", "front_left_thai", "front_left_knee",
    "front_left_paw", "front_right_thai", "front_right_knee",
    "front_right_paw", "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw", "belly_bottom",
    "body_middle_right", "body_middle_left",
]


def test_map_covers_every_canonical_name():
    assert set(SUPERANIMAL_QUADRUPED_MAP) == set(KEYPOINT_NAMES)


def test_map_targets_exist_and_are_distinct():
    targets = [SUPERANIMAL_QUADRUPED_MAP[n] for n in KEYPOINT_NAMES]
    assert set(targets) <= set(SUPERANIMAL_BODYPARTS)
    assert len(set(targets)) == len(targets), "two canonical points share a bodypart"


def test_map_respects_left_right_and_front_rear():
    """FR/FL and front/rear must not be transposed — the silent, deadly one."""
    for canon, part in SUPERANIMAL_QUADRUPED_MAP.items():
        if canon.startswith("FR") or canon.startswith("RR"):
            assert "right" in part, f"{canon} -> {part} is not a right-side part"
        elif canon.startswith("FL") or canon.startswith("RL"):
            assert "left" in part, f"{canon} -> {part} is not a left-side part"
    for canon, part in SUPERANIMAL_QUADRUPED_MAP.items():
        if canon.startswith("F"):
            assert part.startswith("front"), f"{canon} -> {part} is not a front limb"
        elif canon.startswith("R") and canon != "root":
            assert part.startswith("back"), f"{canon} -> {part} is not a rear limb"


def test_to_canonical_maps_by_name_not_index():
    """Shuffling the detector's bodypart order must not change the output."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(5, len(SUPERANIMAL_BODYPARTS), 3))
    ref = to_canonical(raw, SUPERANIMAL_BODYPARTS)

    order = rng.permutation(len(SUPERANIMAL_BODYPARTS))
    shuffled_parts = [SUPERANIMAL_BODYPARTS[i] for i in order]
    shuffled_raw = raw[:, order, :]
    assert np.allclose(to_canonical(shuffled_raw, shuffled_parts), ref)


def test_to_canonical_picks_the_named_rows():
    raw = np.zeros((2, len(SUPERANIMAL_BODYPARTS), 3))
    for j, part in enumerate(SUPERANIMAL_BODYPARTS):
        raw[:, j, 0] = j
    out = to_canonical(raw, SUPERANIMAL_BODYPARTS)
    for i, name in enumerate(KEYPOINT_NAMES):
        expected = SUPERANIMAL_BODYPARTS.index(SUPERANIMAL_QUADRUPED_MAP[name])
        assert out[0, i, 0] == expected


def test_to_canonical_rejects_unknown_bodyparts():
    with pytest.raises(KeyError):
        to_canonical(np.zeros((2, 3, 3)), ["nose", "tail_base", "left_eye"])


def test_smooth_flags_low_confidence_without_zeroing():
    n = 120
    kp = np.zeros((n, len(KEYPOINT_NAMES), 3))
    kp[..., 0] = np.arange(n)[:, None] * 1.0
    kp[..., 1] = 50.0
    kp[..., 2] = 0.9
    kp[40:50, 3, 2] = 0.01          # a dropout on one keypoint

    out, valid = smooth_2d(kp, fps=60.0)
    assert valid[:40, 3].all() and valid[50:, 3].all()
    assert not valid[40:50, 3].any()
    # The gap is interpolated, not zeroed: an interpolated foot must stay near
    # the track, because (0, 0) reads as a real foot at the image corner.
    assert np.all(out[40:50, 3, 0] > 30.0)
    assert np.isfinite(out[40:50, 3, :2]).all()


def test_smooth_reduces_jitter_but_keeps_the_trend():
    rng = np.random.default_rng(1)
    n = 200
    truth = np.linspace(0, 100, n)
    kp = np.zeros((n, len(KEYPOINT_NAMES), 3))
    kp[..., 0] = (truth + rng.normal(0, 3.0, n))[:, None]
    kp[..., 1] = 20.0
    kp[..., 2] = 0.9

    out, _ = smooth_2d(kp, fps=60.0)
    raw_jitter = np.abs(np.diff(kp[:, 0, 0])).mean()
    smooth_jitter = np.abs(np.diff(out[:, 0, 0])).mean()
    assert smooth_jitter < 0.5 * raw_jitter
    assert np.abs(out[:, 0, 0] - truth).mean() < 2.0


def test_smooth_survives_a_fully_missing_keypoint():
    n = 60
    kp = np.zeros((n, len(KEYPOINT_NAMES), 3))
    kp[..., 2] = 0.9
    kp[:, 7, 2] = 0.0
    out, valid = smooth_2d(kp, fps=60.0)
    assert not valid[:, 7].any()
    assert np.isnan(out[:, 7, :2]).all(), "a never-seen keypoint must stay NaN"


@pytest.fixture(scope="module")
def gt():
    if not GOLDEN_GT.exists():
        pytest.skip(f"{GOLDEN_GT} missing — run v2k.render first")
    return dict(np.load(GOLDEN_GT, allow_pickle=True))


def test_gt_bboxes_enclose_the_visible_keypoints(gt):
    boxes = gt_bboxes(gt, margin=0.15)
    uv, depth = gt["uv"], gt["depth"]
    assert len(boxes) == len(uv)
    assert (boxes[:, 2] > 0).all() and (boxes[:, 3] > 0).all()
    for i in range(0, len(uv), 37):
        vis = depth[i] > 0.05
        if not vis.any():
            continue
        x0, y0, w, h = boxes[i]
        pts = uv[i][vis]
        inside = ((pts[:, 0] >= x0 - 1) & (pts[:, 0] <= x0 + w + 1)
                  & (pts[:, 1] >= y0 - 1) & (pts[:, 1] <= y0 + h + 1))
        # Points outside the image are clipped away by the frame bounds.
        in_frame = ((pts[:, 0] >= 0) & (pts[:, 0] < int(gt["cam_width"]))
                    & (pts[:, 1] >= 0) & (pts[:, 1] < int(gt["cam_height"])))
        assert inside[in_frame].all()


def test_gt_bboxes_stay_inside_the_frame(gt):
    boxes = gt_bboxes(gt)
    w, h = int(gt["cam_width"]), int(gt["cam_height"])
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 0] + boxes[:, 2] <= w + 1e-6).all()
    assert (boxes[:, 1] + boxes[:, 3] <= h + 1e-6).all()


def _perfect_detection(gt, conf=0.9):
    uv = np.asarray(gt["uv"])
    n = len(uv)
    return {
        "num_frames": n,
        "uv": uv.copy(),
        "uv_raw": uv.copy(),
        "conf": np.full((n, len(KEYPOINT_NAMES)), conf),
        "valid": np.ones((n, len(KEYPOINT_NAMES)), dtype=bool),
    }


def test_scorer_passes_a_perfect_detection(gt):
    metrics = score(_perfect_detection(gt), gt)
    assert metrics["median_trunk"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert gate(metrics)[0]


def test_scorer_fails_a_swapped_pair(gt):
    """FR/FL transposition must be caught by the numbers, not just by eye."""
    det = _perfect_detection(gt)
    fr, fl = KEYPOINT_NAMES.index("FR_toe"), KEYPOINT_NAMES.index("FL_toe")
    det["uv"][:, [fr, fl]] = det["uv"][:, [fl, fr]]
    passed, fails = gate(score(det, gt))
    assert not passed, f"a swapped toe pair passed the gate ({fails})"


def test_scorer_ignores_frames_the_detector_flagged(gt):
    """Garbage under the confidence threshold must not pollute the error."""
    det = _perfect_detection(gt)
    det["uv"][:50] += 500.0
    det["conf"][:50] = 0.0
    det["valid"][:50] = False
    metrics = score(det, gt)
    assert metrics["median_trunk"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["coverage"] < 1.0, "coverage must still record the dropout"


def test_gate_fails_on_low_coverage(gt):
    det = _perfect_detection(gt)
    det["valid"][: int(0.5 * len(det["valid"]))] = False
    passed, fails = gate(score(det, gt))
    assert not passed and any("coverage" in f for f in fails)


def test_trunk_px_is_the_root_chest_distance():
    uv = np.zeros((3, len(KEYPOINT_NAMES), 2))
    uv[:, KEYPOINT_NAMES.index("chest"), 0] = 30.0
    uv[:, KEYPOINT_NAMES.index("root"), 0] = 10.0
    assert np.allclose(trunk_px(uv), 20.0)


def test_min_conf_threshold_is_documented_value():
    assert 0.0 < DEFAULT_MIN_CONF < 1.0
