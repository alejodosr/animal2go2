# animal2go2 — Milestone 1: Kinematic Retargeting (Dog Mocap → Unitree Go2)

## Context

This is the first technical milestone of **animal2go2**, an open-source pipeline inspired by
[AIM-Intelligence/video2robot](https://github.com/AIM-Intelligence/video2robot) (video → human pose → humanoid motion),
adapted to quadrupeds: animal motion → Unitree Go2 trajectories → (later) RL tracking in Isaac Lab → sim2sim validation in MuJoCo.

**This milestone is purely kinematic. No RL, no physics simulation of dynamics.** The goal is:
given dog motion capture data, produce physically plausible Go2 joint trajectories and play them back in the MuJoCo viewer.

The output of this milestone becomes the reference-motion input for the RL tracking milestone later,
so the output format matters (see §7).

Hardware available: Linux machine with RTX 3090. Everything in this milestone runs fine on CPU; the GPU is irrelevant here.

---

## 1. Deliverables

1. `parse_mocap.py` — load dog mocap clips, extract a canonical keypoint trajectory (root + 4 feet + reference joints).
2. `viz_source.py` — visualize the raw dog skeleton motion (matplotlib 3D or rerun.io) to sanity-check parsing.
3. `retarget.py` — core retargeting: dog keypoints → Go2 root pose + 12 joint angles per frame.
4. `playback.py` — kinematic playback of retargeted motion in the MuJoCo passive viewer + mp4 export.
5. `motions/*.pkl` — at least 3 retargeted clips (e.g. pace, trot, turn) in the output format of §7.
6. `README.md` — how to reproduce, with honest notes on what breaks (this feeds a Substack article; failures are content).

Suggested repo layout:

```
animal2go2/
├── retarget/
│   ├── parse_mocap.py
│   ├── skeleton.py          # source skeleton definition + keypoint extraction
│   ├── ik.py                # analytic 3-DOF leg IK for Go2
│   ├── retarget.py
│   └── postprocess.py       # contact fixing, filtering
├── viz/
│   ├── viz_source.py
│   └── playback.py
├── assets/                  # Go2 MJCF from Menagerie (git submodule or vendored)
├── motions/                 # output .pkl files (gitignored except samples)
└── data/                    # raw mocap (gitignored)
```

---

## 2. Data sources

### Primary: AI4Animation dog mocap dataset
- Repo: https://github.com/sebastianstarke/AI4Animation — the **SIGGRAPH 2018** project
  ("Mode-Adaptive Neural Networks for Quadruped Motion Control", Zhang, Starke et al.).
- The dog mocap dataset is linked from that section of the repo (download link in the README / project page).
  It contains BVH-style motion capture of a real dog: locomotion gaits (walk, pace, trot, canter), turns, sits, jumps.
- This is the same dataset Peng et al. 2020 used. Verify the license/terms noted in the repo before redistributing any of it.

### Key reference implementation: google-research/motion_imitation
- Repo: https://github.com/google-research/motion_imitation ("Learning Agile Robotic Locomotion Skills by Imitating Animals", Peng et al. 2020).
- Contains `retarget_motion.py` (PyBullet-based) that retargets exactly this dog dataset to Laikago/A1,
  plus already-retargeted motions (`dog_pace`, `dog_trot`, `dog_spin`).
- **Use it as a design reference, not a dependency.** We reimplement in MuJoCo, targeting Go2, cleanly and readably
  (the readability is part of the product — this becomes a tutorial).
- Their retargeted A1 motions are also useful as a cross-check: A1 and Go2 have the same joint topology.

### Robot model: MuJoCo Menagerie Go2
- https://github.com/google-deepmind/mujoco_menagerie → `unitree_go2/` (MJCF + meshes, joint limits included).
- 12 actuated DOF, 3 per leg, order per leg: **hip abduction/adduction (hip), hip pitch (thigh), knee (calf)**.
- Read joint limits and default pose directly from the XML — do not hardcode from memory.
- Leg link lengths (hip offset, thigh length, calf length) needed for analytic IK: extract from the MJCF body positions,
  or cross-check with the official Unitree Go2 URDF (unitreerobotics/unitree_ros).

---

## 3. Core design decision: retarget positions, not joint angles

Dog and Go2 morphologies do not correspond joint-by-joint:
- A dog's front leg bends backward at the elbow; the rear leg has a forward-bending stifle and backward-bending hock.
  Go2's legs all share one topology (hip/thigh/calf).
- The dog has an articulated spine; Go2's trunk is rigid.

So **do not map joint angles**. Follow the Peng et al. approach:

1. From the dog, extract only: root (pelvis/trunk) position + orientation, and the 4 toe positions
   (optionally shoulder/hip positions to disambiguate).
2. Map the root: Go2 base pose ← dog trunk pose, with spine bending projected out
   (fit a single rigid frame to the dog's hip+shoulder segment, e.g. average of pelvis and chest orientation, yaw+pitch+roll of that fitted frame).
3. Map the feet: Go2 target toe positions ← dog toe positions, expressed **relative to the root frame**, scaled.
4. Solve per-leg IK to get the 12 joint angles.

### Scaling
- Compute a uniform scale factor = (Go2 nominal leg length) / (dog leg length), where leg length = thigh + calf.
- Also scale the root height: Go2 standing height ≈ 0.27–0.34 m (read the default/home keyframe from the MJCF).
- Optionally scale front/rear and lateral offsets separately (dog is longer and narrower relative to leg length than Go2);
  start with uniform scale + a fixed offset correction per leg mounting point, iterate visually.

---

## 4. IK — what it is and what we need (primer for the author, who hasn't done IK before)

**Forward kinematics (FK)**: given joint angles, compute where the foot is. Just chained rotations/translations.
**Inverse kinematics (IK)**: given a desired foot position, find the joint angles. The inverse problem.

For a general robot this needs numerical methods, but each Go2 leg is a simple **3-DOF chain**
(hip abduction about x, hip pitch about y, knee about y) reaching a point in 3D — this has a **closed-form analytic solution**.
This is the standard quadruped leg IK, well documented; the derivation is roughly:

1. Express the target foot position in the hip frame.
2. The hip abduction angle comes from the lateral geometry (atan2 in the y–z plane, accounting for the hip's lateral offset).
3. Project into the leg's sagittal plane; the remaining 2-DOF (thigh + calf) is a planar two-link arm:
   knee angle from the law of cosines, thigh angle from atan2 minus the interior angle.
4. Pick the knee-backward solution branch (matching Go2's build), clamp to joint limits from the MJCF,
   and clamp the target to the reachable workspace (‖target‖ ≤ thigh + calf − ε) before solving to avoid NaNs.

Implement this in `ik.py` with a matching FK function, and **unit-test it**: FK(IK(p)) ≈ p for random reachable points, all four legs
(legs differ by mirroring — get the signs right per leg).

Alternative/fallback: numerical IK with [mink](https://github.com/kevinzakka/mink) (differential IK on the MuJoCo model).
Fine as a cross-check, but the analytic version is preferred: it's faster, deterministic, and pedagogically valuable for the article.

---

## 5. Post-processing (this is what separates "demo" from "usable")

1. **Contact detection on the source**: a dog toe is in stance when its height is below a threshold AND its horizontal
   velocity is below a threshold (tune per clip; the dataset's toe markers make this reliable for locomotion gaits).
2. **Foot-skate removal**: during a detected stance phase, pin the retargeted foot target to its position at touch-down
   (and to ground height z=0), blending in/out over ~2–3 frames to avoid pops.
3. **Ground alignment**: shift root z so that stance feet sit on z=0; the mocap ground plane is not MuJoCo's.
4. **Smoothing**: low-pass (Butterworth, ~6–8 Hz cutoff at the clip's fps) on joint trajectories AND root trajectory, applied after IK.
   Check that smoothing doesn't reintroduce foot-skate; if it does, filter foot targets before IK instead.
5. **Limit check**: assert every frame is within joint limits; log the clamp rate. A high clamp rate (>2–3% of frames)
   means the scaling/offsets are wrong — fix upstream rather than clamping harder.

---

## 6. Coordinate & convention pitfalls (check these FIRST, they cause 80% of the bugs)

- **Up axis**: BVH data is typically Y-up; MuJoCo is Z-up. Convert once, at parse time, and never again.
- **Units**: BVH is often in centimeters; MuJoCo is meters.
- **Quaternions**: MuJoCo uses **wxyz**; scipy `Rotation` uses **xyzw**. Wrap conversions in named helpers
  (`quat_wxyz_to_xyzw`, etc.) — do not inline them.
- **Euler order in BVH**: read the channel order from the file (often ZXY or ZYX), don't assume.
- **FPS**: record source fps; resample everything to a fixed target (e.g. 50 Hz — a typical RL control rate) at output time.
- **Leg ordering**: pick one canonical order (FR, FL, RR, RL — Unitree convention) and use it everywhere; document it.

---

## 7. Output format

Match the video2robot output shape so the whole pipeline stays consistent, extended with contacts:

```python
# motions/<clip_name>.pkl
{
    "fps": 50.0,
    "robot_type": "unitree_go2",
    "num_frames": N,
    "root_pos": np.ndarray,     # (N, 3), meters, world frame, Z-up
    "root_rot": np.ndarray,     # (N, 4), quaternion xyzw
    "dof_pos": np.ndarray,      # (N, 12), radians, order: FR, FL, RR, RL × (hip, thigh, calf)
    "foot_contacts": np.ndarray # (N, 4), bool, same leg order
    "source": str,              # source clip identifier
}
```

---

## 8. Visualization / playback

- `playback.py`: load the Go2 MJCF, set `qpos` directly each frame (root pos + quat + 12 dofs) using
  `mujoco.viewer.launch_passive` — **kinematic playback, do not step the dynamics**.
  Controls: pause, single-step, playback-speed.
- mp4 export via `mujoco.Renderer` + `imageio` (offscreen, 1080p, same fps as clip). These videos are article assets.
- Nice-to-have: side-by-side render of source dog skeleton (as MuJoCo sites/capsules or a second matplotlib panel)
  synced with the Go2 playback — this is the money shot for the article.

---

## 9. Suggested execution order

1. **Phase 0 — environment**: Python 3.10+, `mujoco`, `numpy`, `scipy`, `imageio[ffmpeg]`, `matplotlib`.
   Vendor/submodule the Menagerie Go2 model. Smoke test: open Go2 in the viewer, sweep each of the 12 joints
   through its range one at a time and confirm names/order/signs match §2.
2. **Phase 1 — data**: download the AI4Animation dog dataset; write the parser; visualize 2–3 clips of raw skeleton.
   Do not proceed until the dog visibly walks correctly in the source viz (up-axis/unit bugs die here).
3. **Phase 2 — IK**: implement + unit-test analytic leg IK (FK round-trip test on all 4 legs).
4. **Phase 3 — retarget v0**: root mapping + scaled foot targets + IK, no post-processing. Expect foot-skate and jitter; ship it to the viewer anyway.
5. **Phase 4 — post-process**: contacts, skate removal, ground alignment, smoothing, limit report.
6. **Phase 5 — outputs**: export 3+ clips as .pkl + mp4s; write the README.

## 10. Acceptance criteria

- FK(IK(p)) round-trip error < 1 mm on random reachable targets, all legs.
- Playback of trot/pace clips: no visible foot-skate during stance, no ground penetration, no joint-limit violations (clamp rate < 2%).
- Gait timing of the Go2 matches the source clip (same footfall pattern, verified visually side-by-side).
- Someone with the repo and the dataset can reproduce everything with ≤ 3 commands.

## 11. Explicit non-goals (do not build these now)

- No RL, no physics rollouts, no Isaac Lab.
- No video-based pose estimation (that is Milestone 3).
- No support for robots other than Go2.
- No GUI beyond the MuJoCo viewer.
