# animal2go2

Dog mocap → Unitree Go2 kinematic retargeting (Milestone 1). See `brief_claude.md`
for the full plan. Status: **Phase 0 (environment) and Phase 1 (data) done.**

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync
```

The Menagerie Go2 model is vendored in `assets/unitree_go2/`
(BSD-3, see its `LICENSE`; pinned commit in `MENAGERIE_COMMIT.txt`).

Download the dog mocap dataset (AI4Animation, SIGGRAPH 2018 —
**CC BY-NC 4.0, not redistributed here**):

```bash
curl -L -o data/MotionCapture.zip https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
unzip -o data/MotionCapture.zip -d data/
```

## Phase 0 — Go2 smoke test

```bash
MUJOCO_GL=egl uv run python viz/smoke_test_go2.py          # joint table + sweep mp4 in media/
uv run python viz/smoke_test_go2.py --interactive           # same sweep, live viewer
```

Verified from the MJCF (not hardcoded): 12 joints in qpos order FL, FR, RL, RR
× (hip, thigh, calf) — note this differs from our canonical output order
FR, FL, RR, RL, so all indexing goes through joint *names*. Limits: abduction
±1.047, front thigh [−1.571, 3.491], rear thigh [−0.524, 4.538], knee
[−2.723, −0.838]. Thigh = calf = 0.213 m. Home keyframe: z = 0.27 m,
(0, 0.9, −1.8) per leg. Perturbing each joint moves only its own foot.

## Phase 1 — parse & visualize the dog data

```bash
uv run python retarget/parse_mocap.py --scan                        # stats for all 52 clips
uv run python retarget/parse_mocap.py data/D1_007_KAN01_001.bvh     # -> data/processed/*.npz
uv run python viz/viz_source.py data/D1_007_KAN01_001.bvh           # -> media/source_*.mp4
```

Everything downstream of `retarget/skeleton.py` is meters, Z-up, leg order
FR, FL, RR, RL. Source data is Y-up, centimeters, 60 fps, ZXY euler channels.

### Notes / things that broke (article fodder)

- **Root OFFSET is a trap.** Standard BVH forward kinematics adds the root
  joint's `OFFSET` to its position channels. In this dataset the position
  channels are absolute: adding the offset floats every clip above the ground
  by exactly `OFFSET.y` (7.7 cm for most clips, 50 cm for some). Detected
  because `min(toe_z)` per clip matched each file's `OFFSET.y` to the mm.
- The dog skeleton's front legs are the *arm* chains (`...Shoulder→Arm→
  ForeArm→Hand`), rear legs the *leg* chains; toes are BVH end sites.
- Parsing sanity was verified quantitatively, not just visually: the dog
  moves head-first (heading·spine ≈ +1, heading·tail ≈ −1), and stance
  diagrams show real gaits — D1_007 is a lateral-sequence walk (duty 0.62),
  D1_009_..._002 a trot (diagonal-pair sync +0.74), D1_010_..._004 a canter
  (duty 0.38, flight phases).
