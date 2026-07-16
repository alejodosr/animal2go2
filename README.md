# animal2go2

Dog mocap → Unitree Go2 kinematic retargeting (Milestone 1). See `brief_claude.md`
for the full plan. Status: **Phases 0–3 done (environment, data, leg IK, retarget v0).**

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

## Phase 2 — analytic leg IK

```bash
uv run pytest tests/
```

`retarget/ik.py`: closed-form FK/IK for the 3-DOF Go2 leg (abduction from the
y–z geometry, then thigh+calf as a planar two-link arm via law of cosines),
knee-backward branch, pure numpy, vectorized over frames. Foot targets are in
the base frame; unreachable targets are clamped to the workspace so IK never
returns NaN, and `clamp_to_limits()` reports joint-limit violations for the
phase-4 clamp-rate log. Geometry/limit constants are transcribed from the
MJCF and a test asserts them against the loaded model so they can't drift.

FK(IK(p)) round-trip over 400k random reachable targets: max error 0.1 mm
(only at the foot-level-with-hip-axis boundary, where the workspace clamp's
ε engages; machine precision elsewhere). The analytic FK is also checked
against MuJoCo's own kinematics of the calf endpoint — which is what caught
the FL, FR, RL, RR qpos-order trap from phase 0 a second time: canonical dof
vectors must be scattered through `jnt_qposadr`, never block-copied.

## Phase 3 — retarget v0 + playback

```bash
uv run python retarget/retarget.py data/processed/D1_007_KAN01_001.npz   # -> motions/*.pkl
MUJOCO_GL=egl uv run python viz/playback.py motions/D1_007_KAN01_001.pkl # -> media/go2_*.mp4
uv run python viz/playback.py motions/D1_007_KAN01_001.pkl --interactive # live viewer
```

Viewer keys: Space pause, `.` single-step, `[`/`]` speed. Pipeline per the
brief §3: rigid trunk frame fitted to the dog's hip+shoulder segment (x from
pelvis→chest, y from left−right leg roots), uniform scale = 0.27 / mean dog
leg-root height, toes re-anchored from the dog's mean leg mounts to the Go2
leg-plane origins, analytic IK, resample to 50 Hz. Output pkls follow §7
(root_rot **xyzw**; playback converts to MuJoCo wxyz). No post-processing
yet: foot-skate, jitter, and some ground penetration are expected until
phase 4. Clamp rates: walk 0.0%, trot 0.8%, canter 1.3%.

### Notes / things that broke (article fodder)

- **Root OFFSET is a trap.** Standard BVH forward kinematics adds the root
  joint's `OFFSET` to its position channels. In this dataset the position
  channels are absolute: adding the offset floats every clip above the ground
  by exactly `OFFSET.y` (7.7 cm for most clips, 50 cm for some). Detected
  because `min(toe_z)` per clip matched each file's `OFFSET.y` to the mm.
- The dog skeleton's front legs are the *arm* chains (`...Shoulder→Arm→
  ForeArm→Hand`), rear legs the *leg* chains; toes are BVH end sites.
- **The dog's anatomy leaks into the robot's posture.** The trunk frame's
  pelvis→chest axis carries a constant pitch bias (withers sit ~10 cm higher
  than the hip balls; −18° mean on the trot clip), which made the Go2 play
  back permanently nose-down/up. Fixed by removing the per-clip *median*
  trunk tilt (median, not mean — the trot clip contains a crouch segment
  that would drag the mean). Dynamic pitch stays: the walk clip really does
  start with the dog sniffing the ground, and the retarget should keep that.
- Parsing sanity was verified quantitatively, not just visually: the dog
  moves head-first (heading·spine ≈ +1, heading·tail ≈ −1), and stance
  diagrams show real gaits — D1_007 is a lateral-sequence walk (duty 0.62),
  D1_009_..._002 a trot (diagonal-pair sync +0.74), D1_010_..._004 a canter
  (duty 0.38, flight phases).
