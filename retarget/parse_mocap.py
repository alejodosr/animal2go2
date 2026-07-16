"""Parse dog mocap BVH clips into canonical keypoint trajectories.

Usage:
    python retarget/parse_mocap.py data/D1_001_KAN01_001.bvh [more.bvh ...]
    python retarget/parse_mocap.py --scan          # summarize every clip in data/

Writes data/processed/<clip>.npz with the keypoint dict of
`skeleton.extract_keypoints` (meters, Z-up, leg order FR FL RR RL).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.skeleton import LEG_ORDER, extract_keypoints, parse_bvh

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "processed"


def clip_stats(kp):
    """Human-readable sanity stats for one parsed clip."""
    n = kp["num_frames"]
    dur = n / kp["fps"]
    root = kp["root_pos"]
    horiz = root[:, :2]
    dist = np.linalg.norm(np.diff(horiz, axis=0), axis=1).sum()
    speed = dist / dur
    toe_z = kp["toe_pos"][..., 2]
    return {
        "frames": n,
        "dur_s": dur,
        "fps": kp["fps"],
        "avg_speed_mps": speed,
        "root_z_mean": root[:, 2].mean(),
        "toe_z_min": toe_z.min(),
        "toe_z_max": toe_z.max(),
    }


def process(path, save=True, verbose=True):
    clip = parse_bvh(path)
    kp = extract_keypoints(clip)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{clip.name}.npz"
        np.savez_compressed(out, **kp)
        if verbose:
            print(f"wrote {out.relative_to(REPO_ROOT)}")
    if verbose:
        s = clip_stats(kp)
        print(f"  {clip.name}: {s['frames']} frames @ {s['fps']:.0f} fps "
              f"({s['dur_s']:.1f} s), avg speed {s['avg_speed_mps']:.2f} m/s, "
              f"root z mean {s['root_z_mean']:.3f} m, "
              f"toe z [{s['toe_z_min']:.3f}, {s['toe_z_max']:.3f}] m")
    return kp


def scan():
    print(f"{'clip':<24} {'frames':>7} {'dur_s':>7} {'speed':>7} {'root_z':>7} {'toe_z_min':>9}")
    for path in sorted(DATA_DIR.glob("*.bvh")):
        try:
            kp = extract_keypoints(parse_bvh(path))
        except Exception as e:  # a broken clip should not kill the scan
            print(f"{path.stem:<24} FAILED: {e}")
            continue
        s = clip_stats(kp)
        print(f"{path.stem:<24} {s['frames']:>7} {s['dur_s']:>7.1f} "
              f"{s['avg_speed_mps']:>7.2f} {s['root_z_mean']:>7.3f} {s['toe_z_min']:>9.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", type=Path, help="BVH files to process")
    parser.add_argument("--scan", action="store_true",
                        help="print stats for every BVH in data/ (no output files)")
    args = parser.parse_args()

    if args.scan:
        scan()
        return
    if not args.clips:
        parser.error("give at least one BVH file, or use --scan")
    print(f"leg order: {', '.join(LEG_ORDER)}")
    for path in args.clips:
        process(path)


if __name__ == "__main__":
    main()
