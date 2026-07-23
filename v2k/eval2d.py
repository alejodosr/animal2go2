"""Phase 1 gate: score detected 2D keypoints against the projected mocap.

Errors are reported **normalized by the projected trunk length** (root->chest
in pixels), not in raw pixels. A 10 px error is excellent on a subject that
fills the frame and meaningless on one that is 20 px across; only the ratio
transfers between clips, resolutions and camera distances.

Coverage is reported separately from accuracy, and accuracy is computed over
confident frames only. Averaging error across frames where the model said "I
don't see it" measures nothing.

Usage:
    python -m v2k.eval2d data/synthetic/D1_007_KAN01_001_side_det2d.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.seam import KEYPOINT_NAMES

# Gate thresholds. Held deliberately loose: Phase 2 lifts these points to 3D
# and Phase 3 re-anchors them, so what matters here is "the right body part,
# not a different leg", not sub-pixel precision.
MAX_MEDIAN_ERR_TRUNK = 0.25   # median error, in trunk lengths
MAX_P95_ERR_TRUNK = 0.75      # tail error, in trunk lengths
MIN_COVERAGE = 0.80           # fraction of keypoint-frames above confidence


def trunk_px(uv):
    """Per-frame projected trunk length (root->chest) in pixels."""
    r = KEYPOINT_NAMES.index("root")
    c = KEYPOINT_NAMES.index("chest")
    return np.linalg.norm(uv[:, c] - uv[:, r], axis=-1)


def score(det, gt, min_conf=None):
    """Compare a detect2d result dict against a render gt2d dict.

    Returns a dict of per-keypoint and aggregate metrics. `valid` from the
    detection is used as the confidence mask unless `min_conf` overrides it.
    """
    n = min(int(det["num_frames"]), len(gt["uv"]))
    uv_det = np.asarray(det["uv"])[:n]
    uv_gt = np.asarray(gt["uv"])[:n]
    conf = np.asarray(det["conf"])[:n]
    valid = (conf >= min_conf) if min_conf is not None else np.asarray(det["valid"])[:n]

    # Only score against GT points that are actually in front of the camera and
    # inside the image; a keypoint the camera cannot see is not a detector miss.
    depth = np.asarray(gt["depth"])[:n]
    w, h = int(gt["cam_width"]), int(gt["cam_height"])
    visible = ((depth > 0.05)
               & (uv_gt[..., 0] >= 0) & (uv_gt[..., 0] < w)
               & (uv_gt[..., 1] >= 0) & (uv_gt[..., 1] < h))

    scale = trunk_px(uv_gt)
    scale = np.where(scale > 1e-6, scale, np.nan)
    err_px = np.linalg.norm(uv_det - uv_gt, axis=-1)
    err_trunk = err_px / scale[:, None]

    def _median(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    def _p95(a):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 95)) if a.size else float("nan")

    def _mean(a):
        return float(a.mean()) if a.size else float("nan")

    scored = visible & valid
    per_kp = {}
    for j, name in enumerate(KEYPOINT_NAMES):
        m = scored[:, j]
        seen = visible[:, j]
        per_kp[name] = {
            "coverage": _mean(valid[seen, j]),
            "median_trunk": _median(err_trunk[m, j]),
            "p95_trunk": _p95(err_trunk[m, j]),
            "median_px": _median(err_px[m, j]),
            "mean_conf": _mean(conf[seen, j]),
        }

    all_e = err_trunk[scored]
    return {
        "num_frames": n,
        "median_trunk_px": float(np.nanmedian(scale)),
        "coverage": _mean(valid[visible]),
        "median_trunk": _median(all_e),
        "p95_trunk": _p95(all_e),
        "per_keypoint": per_kp,
    }


def gate(metrics):
    """Apply the Phase 1 pass criteria. Returns (passed, list of failures)."""
    fails = []
    if not (metrics["coverage"] >= MIN_COVERAGE):
        fails.append(f"coverage {metrics['coverage']:.1%} < {MIN_COVERAGE:.0%}")
    if not (metrics["median_trunk"] <= MAX_MEDIAN_ERR_TRUNK):
        fails.append(f"median error {metrics['median_trunk']:.2f} trunks "
                     f"> {MAX_MEDIAN_ERR_TRUNK}")
    if not (metrics["p95_trunk"] <= MAX_P95_ERR_TRUNK):
        fails.append(f"p95 error {metrics['p95_trunk']:.2f} trunks "
                     f"> {MAX_P95_ERR_TRUNK}")
    return (not fails), fails


def report(metrics, name=""):
    """Human-readable table; returns the printed string."""
    lines = [f"{name}  ({metrics['num_frames']} frames, "
             f"trunk {metrics['median_trunk_px']:.1f} px in image)"]
    lines.append(f"  {'keypoint':10s} {'cover':>7s} {'conf':>6s} "
                 f"{'med/trunk':>10s} {'p95/trunk':>10s} {'med px':>8s}")
    for name_kp, m in metrics["per_keypoint"].items():
        lines.append(f"  {name_kp:10s} {m['coverage']:7.1%} {m['mean_conf']:6.3f} "
                     f"{m['median_trunk']:10.3f} {m['p95_trunk']:10.3f} "
                     f"{m['median_px']:8.1f}")
    lines.append(f"  {'ALL':10s} {metrics['coverage']:7.1%} {'':6s} "
                 f"{metrics['median_trunk']:10.3f} {metrics['p95_trunk']:10.3f}")
    passed, fails = gate(metrics)
    lines.append(f"  GATE: {'PASS' if passed else 'FAIL'}"
                 + ("" if passed else "  — " + "; ".join(fails)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", nargs="+", type=Path,
                        help="*_det2d.npz files")
    parser.add_argument("--gt", type=Path, default=None,
                        help="explicit *_gt2d.npz (default: derived from name)")
    parser.add_argument("--min-conf", type=float, default=None)
    args = parser.parse_args()

    failed = 0
    for path in args.detections:
        det = dict(np.load(path, allow_pickle=True))
        gt_path = args.gt
        if gt_path is None:
            stem = path.name.replace("_det2d.npz", "")
            gt_path = path.with_name(f"{stem}_gt2d.npz")
        gt = dict(np.load(gt_path, allow_pickle=True))
        metrics = score(det, gt, min_conf=args.min_conf)
        print(report(metrics, name=str(path.name)))
        print()
        if not gate(metrics)[0]:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
