"""Phase 1 — 2D keypoint extraction from monocular video.

Runs DeepLabCut's SuperAnimal-Quadruped (top-down: object detector -> HRNet
pose head) over a clip, maps its 39 bodyparts onto the 10 canonical M0 names
by **name, never by index**, and temporally filters the result.

Two bounding-box sources, because they answer different questions:

- ``detector``  the learned Faster R-CNN. This is the field-tier path and the
  only one that exists for real footage.
- ``gt``        boxes derived from the projected mocap in ``<clip>_gt2d.npz``.
  Scores the *pose head alone*, with box-finding taken out of the loop — the
  rendered tier's whole point is scoring phases in isolation.

Low-confidence keypoints are **flagged, not zeroed**: the returned array keeps
whatever the model said, and a boolean ``valid`` mask records where it was
confident. Silently writing zeros would look like a foot at the image origin
to every downstream phase.

Usage:
    python -m v2k.detect2d data/synthetic/D1_007_KAN01_001_side.mp4
    python -m v2k.detect2d data/synthetic/D1_007_KAN01_001_side.mp4 --bbox gt
    # writes data/synthetic/<clip>_det2d.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2k.seam import KEYPOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical name -> SuperAnimal-Quadruped bodypart. All 10 have a direct
# anatomical counterpart, which is why this detector was picked over
# ViTPose+/AP-10K (whose HF checkpoint ships a COCO-human head only).
#
# `root` is the mocap `Hips` and `chest` the mocap `Spine1`; SuperAnimal's
# `back_end`/`back_base` are the same two ends of the trunk. `tail_base` sits
# slightly aft of `Hips` — a known bias, measured in eval2d, not hidden here.
#
# "thai" is SuperAnimal's spelling of thigh; those four are the limb-to-trunk
# attachments, i.e. our leg mounts. Left/right are the *animal's* own, matching
# the FR/FL/RR/RL convention.
SUPERANIMAL_QUADRUPED_MAP = {
    "root": "back_end",
    "chest": "back_base",
    "FR_mount": "front_right_thai",
    "FL_mount": "front_left_thai",
    "RR_mount": "back_right_thai",
    "RL_mount": "back_left_thai",
    "FR_toe": "front_right_paw",
    "FL_toe": "front_left_paw",
    "RR_toe": "back_right_paw",
    "RL_toe": "back_left_paw",
}

DEFAULT_MIN_CONF = 0.3     # below this a keypoint is flagged invalid
DEFAULT_CUTOFF_HZ = 7.0    # mirrors retarget/postprocess.py CUTOFF_HZ


class SuperAnimalQuadruped:
    """DLC SuperAnimal-Quadruped as a plain (N, 39, 3) video -> keypoints call.

    Wraps the runners directly rather than ``video_inference_superanimal`` so a
    caller can supply its own boxes and so nothing is written to disk.
    """

    def __init__(self, model_name="hrnet_w32",
                 detector_name="fasterrcnn_resnet50_fpn_v2",
                 device="cuda", batch_size=8, detector_batch_size=4):
        from deeplabcut.pose_estimation_pytorch.apis.utils import (
            get_inference_runners,
        )
        from deeplabcut.pose_estimation_pytorch.modelzoo.utils import (
            get_super_animal_snapshot_path,
            load_super_animal_config,
        )

        self.superanimal = "superanimal_quadruped"
        self.model_name = model_name
        self.detector_name = detector_name

        self.model_cfg = load_super_animal_config(
            super_animal=self.superanimal,
            model_name=model_name,
            detector_name=detector_name,
        )
        pose_snapshot = get_super_animal_snapshot_path(
            dataset=self.superanimal, model_name=model_name, download=True)
        det_snapshot = get_super_animal_snapshot_path(
            dataset=self.superanimal, model_name=detector_name, download=True)

        self.bodyparts = list(self.model_cfg["metadata"]["bodyparts"])
        self.pose_runner, self.detector_runner = get_inference_runners(
            model_config=self.model_cfg,
            snapshot_path=pose_snapshot,
            max_individuals=1,
            num_bodyparts=len(self.bodyparts),
            num_unique_bodyparts=0,
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            detector_path=det_snapshot,
            device=device,
        )

    def detect(self, video_path, bboxes=None, max_frames=None):
        """Run the model over a video.

        Args:
            video_path: mp4 to analyze.
            bboxes: optional (N, 4) or (N, 1, 4) boxes in ``(x0, y0, w, h)``.
                When given, the learned detector is bypassed entirely.
            max_frames: analyze only the first N frames.

        Returns:
            (n_analyzed, 39, 3) array of ``(u, v, confidence)``. Frames where
            nothing was found are NaN in u/v and 0 in confidence.
        """
        from deeplabcut.pose_estimation_pytorch.apis.videos import (
            VideoIterator,
            video_inference,
        )

        video = VideoIterator(str(video_path))
        n_video = video.get_n_frames()
        n_frames = n_video if max_frames is None else min(n_video, max_frames)

        detector_runner = self.detector_runner
        if bboxes is not None:
            boxes = np.asarray(bboxes, dtype=float)
            if boxes.ndim == 2:
                boxes = boxes[:, None, :]
            if len(boxes) < n_frames:
                raise ValueError(
                    f"got {len(boxes)} bboxes for {n_frames} frames")
            # VideoIterator always walks the whole file and indexes the context
            # per frame, so the list has to cover every frame even when only the
            # first `max_frames` are kept. Past the supplied boxes, reuse the
            # last one; those frames are discarded below.
            video.set_context([
                {"bboxes": boxes[min(i, len(boxes) - 1)],
                 "bbox_scores": np.ones(len(boxes[min(i, len(boxes) - 1)]))}
                for i in range(n_video)
            ])
            detector_runner = None

        preds = video_inference(video, pose_runner=self.pose_runner,
                                detector_runner=detector_runner)

        out = np.full((n_frames, len(self.bodyparts), 3), np.nan)
        out[..., 2] = 0.0
        for i, p in enumerate(preds[:n_frames]):
            pose = np.asarray(p["bodyparts"])   # (n_individuals, n_bpt, >=3)
            if pose.size:
                out[i] = pose[0, :, :3]
        return out


def gt_bboxes(gt2d, margin=0.15, width=None, height=None):
    """Boxes enclosing the projected mocap, in ``(x0, y0, w, h)``.

    `margin` pads each side by a fraction of the box size, so the crop looks
    like what a detector would hand a top-down pose model rather than a
    pixel-tight hull.
    """
    uv = np.asarray(gt2d["uv"])            # (N, 10, 2)
    depth = np.asarray(gt2d["depth"])      # (N, 10)
    width = int(gt2d["cam_width"]) if width is None else width
    height = int(gt2d["cam_height"]) if height is None else height

    boxes = np.zeros((len(uv), 4))
    for i in range(len(uv)):
        vis = depth[i] > 0.05
        pts = uv[i][vis] if vis.any() else uv[i]
        lo, hi = pts.min(0), pts.max(0)
        pad = margin * np.maximum(hi - lo, 1.0)
        lo, hi = lo - pad, hi + pad
        lo = np.maximum(lo, [0, 0])
        hi = np.minimum(hi, [width, height])
        boxes[i] = [lo[0], lo[1], max(hi[0] - lo[0], 1.0),
                    max(hi[1] - lo[1], 1.0)]
    return boxes


def to_canonical(raw, bodyparts, mapping=None):
    """(N, n_bodyparts, 3) -> (N, 10, 3) in KEYPOINT_NAMES order, by name."""
    mapping = SUPERANIMAL_QUADRUPED_MAP if mapping is None else mapping
    missing = [n for n in KEYPOINT_NAMES if mapping.get(n) not in bodyparts]
    if missing:
        raise KeyError(
            f"detector has no bodypart for canonical {missing} "
            f"(mapped to {[mapping.get(n) for n in missing]})")
    idx = [bodyparts.index(mapping[n]) for n in KEYPOINT_NAMES]
    return np.asarray(raw)[:, idx, :]


def smooth_2d(kp2d, fps, cutoff=DEFAULT_CUTOFF_HZ, min_conf=DEFAULT_MIN_CONF):
    """Confidence-gate, gap-fill, then low-pass each keypoint track.

    Returns ``(smoothed, valid)`` where `smoothed` is (N, K, 3) with filtered
    u/v and the original confidences, and `valid` is the (N, K) boolean mask of
    frames the model was actually confident about. Gaps are interpolated so the
    filter has something continuous to work on, but they stay marked invalid —
    an interpolated foot is a guess, and Phase 2/3 need to know which.
    """
    kp2d = np.asarray(kp2d, dtype=float)
    n, k, _ = kp2d.shape
    conf = np.nan_to_num(kp2d[..., 2])
    valid = (conf >= min_conf) & np.isfinite(kp2d[..., 0]) & np.isfinite(kp2d[..., 1])

    out = kp2d.copy()
    t = np.arange(n)
    for j in range(k):
        m = valid[:, j]
        if m.sum() < 2:
            out[:, j, :2] = np.nan
            continue
        for c in range(2):
            out[:, j, c] = np.interp(t, t[m], kp2d[m, j, c])

    good = np.isfinite(out[..., :2]).all(axis=(0, 2))
    if good.any() and cutoff < 0.5 * fps and n > 12:
        b, a = butter(2, cutoff / (0.5 * fps))
        out[:, good, :2] = filtfilt(b, a, out[:, good, :2], axis=0)
    return out, valid


def detect_clip(video_path, gt2d_path=None, bbox_source="detector",
                max_frames=None, detector=None, **kwargs):
    """Full Phase 1 pass over one clip. Returns the result dict (see main)."""
    import imageio.v2 as imageio

    video_path = Path(video_path)
    gt = dict(np.load(gt2d_path, allow_pickle=True)) if gt2d_path else None

    bboxes = None
    if bbox_source == "gt":
        if gt is None:
            raise ValueError("--bbox gt needs a *_gt2d.npz")
        bboxes = gt_bboxes(gt)

    if detector is None:
        detector = SuperAnimalQuadruped(**kwargs)
    raw = detector.detect(video_path, bboxes=bboxes, max_frames=max_frames)

    with imageio.get_reader(str(video_path)) as r:
        fps = float(r.get_meta_data()["fps"])

    kp2d = to_canonical(raw, detector.bodyparts)
    smoothed, valid = smooth_2d(kp2d, fps)
    return {
        "source": video_path.stem,
        "fps": fps,
        "num_frames": len(kp2d),
        "keypoint_names": np.array(KEYPOINT_NAMES),
        "uv": smoothed[..., :2],
        "conf": kp2d[..., 2],
        "valid": valid,
        "uv_raw": kp2d[..., :2],
        "bbox_source": bbox_source,
        "detector": f"{detector.superanimal}/{detector.model_name}",
        "bodyparts": np.array(detector.bodyparts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--bbox", default="detector", choices=["detector", "gt"],
                        help="bounding-box source (gt = rendered tier only)")
    parser.add_argument("--gt", type=Path, default=None,
                        help="explicit *_gt2d.npz (default: alongside the video)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--model", default="hrnet_w32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-suffix", default="det2d")
    args = parser.parse_args()

    detector = SuperAnimalQuadruped(model_name=args.model, device=args.device)
    for video in args.videos:
        gt_path = args.gt or video.with_name(f"{video.stem}_gt2d.npz")
        if not Path(gt_path).exists():
            gt_path = None
        res = detect_clip(video, gt2d_path=gt_path, bbox_source=args.bbox,
                          max_frames=args.max_frames, detector=detector)
        out = video.with_name(f"{video.stem}_{args.out_suffix}.npz")
        np.savez_compressed(out, **res)
        cov = res["valid"].mean()
        print(f"wrote {out} — {res['num_frames']} frames, "
              f"{cov:.1%} of keypoints above confidence "
              f"(bbox source: {args.bbox})")


if __name__ == "__main__":
    main()
