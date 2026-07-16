"""Phase 5 batch export: retarget every mocap clip and render its playback.

For each source clip (default: every .bvh in data/): retarget + phase-4
post-process -> motions/<clip>.pkl (brief §7 format), then kinematic playback
render -> media/go2_<clip>.mp4. Ends with a summary table (scale, clamp rate,
stance-foot skate before/after post-processing) over the whole dataset.

Usage:
    MUJOCO_GL=egl uv run python export_all.py                    # everything
    MUJOCO_GL=egl uv run python export_all.py data/D1_007*.bvh   # a subset
    uv run python export_all.py --no-video                       # pkls only
"""

import argparse
import pickle
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from retarget.postprocess import postprocess  # noqa: E402
from retarget.retarget import (  # noqa: E402
    MOTIONS_DIR,
    load_keypoints,
    resample,
    retarget_clip,
)

MEDIA_DIR = REPO_ROOT / "media"


def export_clip(path, fps, render):
    kp = load_keypoints(path)
    motion, info = retarget_clip(kp)
    motion, report = postprocess(motion, info["foot_targets"])
    motion = resample(motion, fps)
    with open(MOTIONS_DIR / f"{motion['source']}.pkl", "wb") as f:
        pickle.dump(motion, f)
    if render is not None:
        model, data, render_video = render
        render_video(model, data, motion, MEDIA_DIR / f"go2_{motion['source']}.mp4")
    return {
        "clip": motion["source"],
        "frames": motion["num_frames"],
        "scale": info["scale"],
        "clamp": report["clamp_rate"],
        "skate_before": report["skate_before"],
        "skate_after": report["skate_after"],
        "stance": motion["foot_contacts"].mean(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="*", type=Path,
                        help="source clips (.bvh or processed .npz); default: data/*.bvh")
    parser.add_argument("--fps", type=float, default=50.0, help="output fps")
    parser.add_argument("--no-video", action="store_true",
                        help="write only the .pkl files (no MuJoCo rendering / GPU needed)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip clips whose outputs are already on disk")
    args = parser.parse_args()

    clips = args.clips or sorted((REPO_ROOT / "data").glob("*.bvh"))
    if not clips:
        sys.exit("no clips found — download the dataset first (see README)")
    MOTIONS_DIR.mkdir(exist_ok=True)

    render = None
    if not args.no_video:
        MEDIA_DIR.mkdir(exist_ok=True)
        from viz.playback import load_model, render_video
        model, data = load_model()
        render = (model, data, render_video)

    rows, failed = [], []
    for i, path in enumerate(clips, 1):
        if args.skip_existing:
            pkl = MOTIONS_DIR / f"{path.stem}.pkl"
            mp4 = MEDIA_DIR / f"go2_{path.stem}.mp4"
            if pkl.exists() and (args.no_video or mp4.exists()):
                print(f"[{i}/{len(clips)}] {path.stem}: outputs exist, skipped")
                continue
        print(f"[{i}/{len(clips)}] {path.name}")
        try:
            rows.append(export_clip(path, args.fps, render))
        except Exception:
            traceback.print_exc()
            failed.append(path.name)

    if rows:
        print(f"\n{'clip':<28}{'frames':>7}{'scale':>7}{'clamp%':>8}"
              f"{'skate m/s':>16}{'stance':>8}")
        for r in rows:
            flag = "  <-- clamp > 3%, check scaling" if r["clamp"] > 0.03 else ""
            print(f"{r['clip']:<28}{r['frames']:>7}{r['scale']:>7.3f}"
                  f"{100 * r['clamp']:>8.2f}"
                  f"{r['skate_before']:>9.3f} ->{r['skate_after']:>6.3f} "
                  f"{r['stance']:>7.2f}{flag}")
    if failed:
        sys.exit(f"\n{len(failed)} clip(s) FAILED: {', '.join(failed)}")


if __name__ == "__main__":
    main()
