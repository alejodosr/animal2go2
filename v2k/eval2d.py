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

# Empirical tier (real footage, no ground truth). Spatial error is
# uncomputable without labels, so the gate is robustness only: keep the same
# coverage bar, and require no *major* track dropout — a full-body gap the
# downstream lifter can't interpolate through.
MAJOR_DROPOUT_SEC = 0.25      # a lost-animal gap longer than this fails the gate


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


def _longest_run(flags):
    """Length of the longest run of True in a 1-D boolean sequence."""
    best = cur = 0
    for x in flags:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def score_gtfree(det, min_conf=None):
    """Robustness metrics for a detection with no ground truth (real footage).

    Reports only what the detector said about itself — coverage, confidence and
    track dropout — never spatial error, which needs labels field footage
    doesn't have. `valid` from the detection is the confidence mask unless
    `min_conf` overrides it.
    """
    conf = np.asarray(det["conf"], dtype=float)
    n, _ = conf.shape
    valid = ((conf >= min_conf) if min_conf is not None
             else np.asarray(det["valid"])).astype(bool)
    fps = float(det["fps"])
    names = [str(x) for x in det["keypoint_names"]]

    def _pcts(a):
        a = a[np.isfinite(a) & (a > 0)]
        if not a.size:
            return float("nan"), float("nan"), float("nan")
        return tuple(float(np.percentile(a, p)) for p in (5, 50, 95))

    per_kp = {}
    for j, name in enumerate(names):
        _, p50, _ = _pcts(conf[:, j])
        per_kp[name] = {"coverage": float(valid[:, j].mean()), "conf_p50": p50}

    # A "dropout" is a frame with zero confident keypoints — the animal lost
    # entirely, not just one occluded joint.
    gap = _longest_run(~valid.any(axis=1))
    p05, p50, p95 = _pcts(conf[valid])
    return {
        "num_frames": n,
        "fps": fps,
        "coverage": float(valid.mean()),
        "frame_detection_rate": float(valid.any(axis=1).mean()),
        "conf_p05": p05, "conf_p50": p50, "conf_p95": p95,
        "dropout_frames": int(gap),
        "dropout_sec": (gap / fps) if fps else float("nan"),
        "per_keypoint": per_kp,
    }


def gate_gtfree(metrics):
    """Empirical-tier pass criteria. Returns (passed, list of failures)."""
    fails = []
    if not (metrics["coverage"] >= MIN_COVERAGE):
        fails.append(f"coverage {metrics['coverage']:.1%} < {MIN_COVERAGE:.0%}")
    max_gap = MAJOR_DROPOUT_SEC * metrics["fps"]
    if metrics["dropout_frames"] > max_gap:
        fails.append(f"track dropout {metrics['dropout_sec']:.2f}s "
                     f"({metrics['dropout_frames']} frames) > {MAJOR_DROPOUT_SEC}s")
    return (not fails), fails


def report_gtfree(metrics, name=""):
    """Human-readable robustness table; returns the printed string."""
    lines = [f"{name}  ({metrics['num_frames']} frames @ "
             f"{metrics['fps']:.1f} fps — empirical / no ground truth)"]
    lines.append(f"  {'keypoint':10s} {'cover':>7s} {'conf p50':>9s}")
    for name_kp, m in metrics["per_keypoint"].items():
        lines.append(f"  {name_kp:10s} {m['coverage']:7.1%} {m['conf_p50']:9.3f}")
    lines.append(f"  {'ALL':10s} {metrics['coverage']:7.1%} {metrics['conf_p50']:9.3f}")
    lines.append(f"  frames with a detection {metrics['frame_detection_rate']:7.1%}"
                 f"   conf p05/p95: {metrics['conf_p05']:.3f}/{metrics['conf_p95']:.3f}")
    lines.append(f"  longest track dropout: {metrics['dropout_frames']} frames "
                 f"({metrics['dropout_sec']:.2f}s)")
    passed, fails = gate_gtfree(metrics)
    lines.append(f"  GATE: {'PASS' if passed else 'FAIL'}"
                 + ("" if passed else "  — " + "; ".join(fails)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections", nargs="+", type=Path,
                        help="*_det2d.npz files")
    parser.add_argument("--gt", type=Path, default=None,
                        help="explicit *_gt2d.npz (default: derived from name)")
    parser.add_argument("--empirical", action="store_true",
                        help="force GT-free scoring even when a *_gt2d.npz exists")
    parser.add_argument("--min-conf", type=float, default=None)
    args = parser.parse_args()

    failed = 0
    for path in args.detections:
        det = dict(np.load(path, allow_pickle=True))
        gt_path = args.gt
        if gt_path is None:
            stem = path.name.replace("_det2d.npz", "")
            cand = path.with_name(f"{stem}_gt2d.npz")
            gt_path = cand if cand.exists() else None

        # No ground truth (real footage) or forced: robustness-only tier.
        if args.empirical or gt_path is None:
            metrics = score_gtfree(det, min_conf=args.min_conf)
            print(report_gtfree(metrics, name=str(path.name)))
            passed = gate_gtfree(metrics)[0]
        else:
            gt = dict(np.load(gt_path, allow_pickle=True))
            metrics = score(det, gt, min_conf=args.min_conf)
            print(report(metrics, name=str(path.name)))
            passed = gate(metrics)[0]
        print()
        if not passed:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
