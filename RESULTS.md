# Milestone 2 — Phase 3 training runs

**STATUS (2026-07-19): CURRICULUM COMPLETE (stage7, all 3 clips, one
policy, 4000 iters) — walk 0.031 / 95.8% ✓✓, trot 0.089 / 99.5% ✓✓,
canter 0.090 joint (✓ under bar!) / 47.9% survival (✗, per the brief's
pre-authorized allowance: the clip contains a 4.1 m/s fully-airborne
gallop burst demanding 66 rad/s — 2.2× the actuator limit — plus a 167°
pirouette; physically infeasible segments, the article's "physics
disagrees" moment). Next: Phase 4/5 (assets, robustness, export) or
canter salvage options (see Observations).**

One row per run; one change per run (brief §4). Eval numbers from `eval.py`
(512 envs, deterministic, randomization off) on the run's final checkpoint.

| run | change vs. previous | iters | train reward | train ep len | eval mean joint err (rad) | eval ep len %max | outcome |
|---|---|---|---|---|---|---|---|
| `2026-07-17_14-09-42_stage1_trot` | — (baseline: single-clip trot `D1_009_KAN01_002`, brief §4 defaults) | 3000 | 65.7 | 218/500 | 0.310 | 29.4% | ✗ acceptance failed — see below |
| `2026-07-17_15-38-22_stage1b_trot_rootpos` | root height term → full 3D root-pos term (DeepMimic com: w 0.05→0.1, k 100→10, xy included) | 3000 | 80.4* | 226/500 | 0.267 | 30.3% | ✗ tracking −14%, but still 100% `root_pos` terminations (*reward not comparable: weights changed) |
| `2026-07-17_17-30-44_stage1c_trot_term2m` | `term_root_pos_err` 0.5 → 2.0 m (termination only; run-2 reward kept) | 3000 | 79.6* | 232/500 | 0.347 | 29.8% | ✗✗ **backfired** — drift 0.17→0.67 m/s, all metrics worse; REVERTED (*not comparable) |
| `2026-07-17_19-17-15_stage1d_trot_velx2` | bound back at 0.5 m; `rew_root_vel_w` 0.15 → 0.30 (velocity error is the drift source and is actor-observable via stance-leg dof_vel) | 3000 | 107.3* | 231/500 | 0.277 | 30.3% | ✗ xy err −20% (0.057), ori/height better, but ep len identical — led to finding the wrap bug (*not comparable) |
| `2026-07-18_14-22-09_stage1e_trot_wrapfix` | wrap fix only (+ 8192 envs, new speed default) | 1800 (stopped: plateau from ~1000) | ~100 | 225/500 | 0.271 | 31.6% | ✗ but diagnostic gold: `root_pos` wall GONE, terminations now 100% `root_ori` — unobservable heading drift hits the 45° bound at ~3 s. Joint plateau unchanged (expected: same actions/gains). Eval at model_1800, stage1-era cfg pinned via overrides |
| `2026-07-18_15-03-39_stage2_peng_trot` | Peng-alignment batch: ref-centered actions, kp=100/kd=1.0, vel-limit 30, ee reward (w.2 k40), merged pose kernel (20/10, w.15), split vel kernel (2/0.2, w.1) | 2200 (stopped: plateau from ~1300) | ~90* | 205/500 | **0.0746** | 30.7% | ✓✗ **joint acceptance SMASHED** (0.075 ≪ 0.15 bar, was 0.271; max 0.34); survival still 0% — 511/512 `root_ori` terminations (*reward not comparable) |
| `2026-07-18_16-03-12_stage3_heading_trot` | + heading-error obs (sin/cos ref-relative yaw; contract 80→82) | 3000 | ~98 | 210/500 | 0.0737 | 30.0% | ✗ **no change vs stage2** (len 150 vs 154, 512/512 `root_ori`) — and the yaw/tilt decomposition (added to eval.py) shows why: **yaw was never the problem** (end-of-episode yaw err 4.5°); end-of-episode TILT is 17° and accelerating ⇒ deaths are acute trip/fall events, not slow heading drift. The 45° ori bound is acting as a fall detector |
| `2026-07-18_17-45-50_stage4_acyclic_trot` | clips acyclic (episode = clip, end = truncation; root cause #2 fix) + injective acyclic phase encoding | 2000 (stopped: plateau from ~1300) | ~92 | 210 (cap ≈228) | 0.0905 | **survival 98.2%** (504/513 reach clip end; 9 `root_pos` deaths, all in the fast-trot final phase) | ✓✓ **STAGE-1 ACCEPTANCE MET**: joint err 0.09 < 0.15 AND survival ≫ 90%. `root_ori` deaths: 512→0. Video (one full episode, frame 0 → truncation): `videos/play/rl-video-step-0.mp4` |
| `2026-07-18_19-36-23_stage5_phasefree_trot` | phase-free tracker: phase obs removed, preview extended to steps [1,2,15,50] (20 ms–1 s, Peng-style); contract 82→116 actor / 124 critic | 2000 (stopped: plateau from ~1400) | ~93 | 212 | **0.0802** | survival 92.6% (474/512; 38 `root_pos` deaths, phase 0.8–1.0) | ✓ **parity: acceptance still met** (0.08 < 0.15, 92.6% > 90%) with *better* tracking; survival −5.6 pts vs stage4 — drift deaths in the fast section 9→38 (xy err 0.113→0.131). The phase-free contract is validated; the fast-section drift leak is the open item |
| `2026-07-18_20-36-50_stage6_multiclip_walktrot` | + walk clip (2-clip RSI, one policy, no clip ID — preview-only conditioning) | 3000 (still improving at 2200, ran full) | ~140* | ~230 | walk **0.0286** / trot 0.1085 | walk **95.4%** / trot **99.4%** | ✓✓ **STAGE-2 ACCEPTANCE MET** — both clips pass both bars from one policy. Interference trade: trot tracking −35% vs single-clip best (0.080→0.109) yet trot survival UP (92.6→99.4%; walk co-training regularizes). Walk deaths: 22 root_pos in its fast lead-out (phase 0.8–1.0). Per-clip videos in `videos/play/` (*2-clip reward, not comparable) |
| `2026-07-19_stage7_multiclip_all` (run dir dated 2026-07-18) | + canter clip (3-clip RSI, 4000 iters per user) | 4000 | — | — | walk 0.0306 / trot **0.0892** / canter 0.0900 | walk 95.8% / trot **99.5%** / **canter 47.9%** | ✓✓✗ walk+trot HOLD (trot tracking improved 0.109→0.089 with more data); canter joint err under the bar but survival fails — deaths cluster at phase 0.1–0.2 (the 4.1 m/s, 100%-flight gallop burst, ref dof_vel 66 rad/s = 2.2× actuator limit — infeasible) and 0.8–1.0 (sustained 2–2.5 m/s return canter). Per the brief: acceptable failure, article content. 3 per-clip videos in `videos/play/` |

## Peng 2020 comparison (2026-07-18)

Full numbers in memory note `peng2020-reference-numbers` (extracted from
arXiv:2004.00784 v3 + the released `motion_imitation` code). Deviations
found, ranked by suspected impact on the 0.27 rad plateau:

1. **Action parameterization** (confirmed): paper = absolute PD targets,
   bounds ±2π, low-pass filtered, fixed σ; ours was `default + 0.25·a`
   (needs |a| ≈ 5). Live stage1e logging confirms: `Action/abs_mean 1.25`,
   `abs_max 5.5`, 6% soft-limit clamp.
2. **PD gains**: paper kp=220 (Laikago); Go2 cfg kp=25 → open-loop floor
   τ_RMS/kp ≈ 0.15 rad. **pd_floor.py sweep** (512 envs, 600 steps, trot):

   | kp | kd | mean joint err floor (rad) |
   |---|---|---|
   | 25 | 0.50 | 0.168 |
   | 60 | 0.77 | 0.107 |
   | 100 | 1.00 | 0.091 |
   | 220 | 1.48 | 0.084 |

   Front-calf error ~0.26 rad persists at ALL kp → not stiffness-limited;
   suspect DCMotor `velocity_limit = 21 rad/s` derating torque during swing
   (ref dof_vel peaks 38; real Go2 spec ~30). **Open decision: raise
   velocity_limit to the Go2 datasheet 30 rad/s?** (contract-facing).
3. **Missing end-effector reward**: paper w=0.2, exp(−40·Σ‖Δx_ee‖²),
   root-relative — its 2nd-largest tracking term; we had none.
4. **Root vel kernel**: paper exp(−2‖Δv‖² − 0.2‖Δω‖²) w=0.1; ours had a
   shared k=2 (10× too stiff on ang vel → saturated kernel) at w=0.30.
5. **Termination inversion**: paper terminates on falls only and fights
   drift with a strong global root-pos reward (k=20); we had weak pos
   reward (k=10, w=0.1) + hard 0.5 m termination on an actor-unobservable
   quantity. stage2 adopts the paper reward (k=20 merged pose term); bound
   kept at 0.5 m for now.
6. Preview horizon: paper goals at t+1,2,10,30 (~1 s); ours t+1,2 (40 ms).
   Deferred to multi-clip (phase obs makes single-clip memorizable).
7. Minor: γ 0.95 vs ours 0.99; fixed vs learned σ; 512-256 nets (ours
   512-256-128); ~200M samples (ours ≈295M+). Same source mocap dataset
   (Zhang et al. [68] = KAN clips); paper's own sim trot return only 0.75.

**stage2 config** (implemented, smoke-tested; one paper-alignment batch):
`action_center = "ref"` (targets = ref_dof_pos(t+1) + 0.25·a; zero action
holds the reference, also at RSI), kp=100/kd=1.0, velocity_limit 21→30
rad/s (Go2 datasheet; user-approved), ee reward w=0.2 k=40
(root-frame foot pos vs sim-generated `motions/<clip>_feet.npz` caches —
`gen_feet_cache.py`, replay FK, root-relative → wrap-safe), merged root
pose term w=0.15 exp(−20·pos²−10·ori²), root vel w=0.1 exp(−2·lin−0.2·ang).
Kept from before: joint pos/vel terms, contact_match 0.1, action_rate,
torque penalties, all terminations, obs contract UNCHANGED. Deliberately
NOT in the batch: γ, preview horizon, termination redesign, velocity_limit.

## Open items (2026-07-18)

- **Action reach analysis** (offline, trot clip): max |q_ref − default| =
  1.28 rad (thigh), mean 0.42 rad ⇒ with `action_scale = 0.25` the policy
  needs |a| ≈ 5 at the extremes (≈ 5σ at init noise 1.0). Peng 2020
  parameterizes targets around the *reference* pose, not a static default —
  prime suspect for the 0.27 rad joint-error plateau. Torque is NOT the
  binding constraint: RMS ≈ 3.7 N·m/joint vs 23.7 N·m limit.
- Reference dof_vel peaks 38.4 rad/s > Go2 ~30 rad/s (peaks only).
- ~~Action-saturation logging~~ DONE (2026-07-18): `Action/abs_mean`,
  `abs_max`, `clamp_frac` logged from the env; stage1e live values 1.25 /
  5.5 / 6% confirm the reach analysis above.
- ~~Speed~~ DONE (2026-07-18): governor was already `performance`; 8192
  envs ≈ 156k steps/s (~2× of 4096), only ~7.3 GB VRAM — new default for
  all runs. NOTE: 8192 × 24 doubles samples/iter; the 1500-iter checkpoint
  is the sample-parity point vs the 4096-env stage1a–d runs.
- ~~velocity_limit~~ DECIDED (user, 2026-07-18): raised 21 → 30 rad/s (Go2
  datasheet) and included in stage2. Rationale: kp-independent front-calf
  PD-floor error (~0.26 rad) at the stock 21; ref peaks 38.4.
- Episode-length note: the trot clip is 9.12 s and max episode 10 s ⇒ under
  the wrap bug NO episode could survive (must cross a wrap); explains 0%
  survival everywhere.

## Observations

- **ROOT CAUSE #2 FOUND (2026-07-18, termination-phase diagnostic): the
  clips are not loops.** eval.py now logs death-phase + episode-length
  histograms: **511/512 stage3 deaths land in phase bin 0.9–1.0** — at the
  wrap seam, not spread over the gait. Clip forensics (offline, pkl):
  `D1_009` "trot" = 2.7 s motionless crouch (36% of the clip!) → stand-up
  transition → 5.5 s trot ending mid-flight. The cyclic wrap therefore
  demands trot-flight → crouch in one 20 ms frame: Δdof 1.34 rad, Δz
  −246 mm, Δyaw 20°, contacts 0→4. The *reference* teleports in
  orientation/pose space at every loop — ori error crosses 45° within
  steps regardless of policy quality (deaths are reference teleports, NOT
  physical falls; the "end tilt 17°" was tilt vs a reference slerping into
  the crouch). All three curriculum clips are diseased: walk seam 1.24 rad
  + 234 mm z (9% still), trot 1.34 rad + 246 mm (36% still), canter 1.03
  rad + **167° yaw** (the dog turns around mid-clip). Every episode-length
  / survival number in this table is seam-bounded (max achievable ep len ≈
  time-to-first-seam; ≈ 228 mean under uniform RSI). Same *class* of bug
  as the xy wrap teleport but in the clip CONTENT — M1 exported raw
  captures with lead-ins/outs; the "clean cyclic gait" premise was never
  true. FIX: trim clips to stride-cycle-aligned gait loops (offline tool),
  regen feet caches, retrain. The stage2/3 joint tracking (0.074) survived
  all this — the policy itself is likely fine.
- **stage3 eval + ori decomposition (2026-07-18, the "drift" narrative was
  wrong)**: eval.py now reports yaw and tilt separately (mean and
  end-of-episode). stage3: yaw err 3.7° mean / 4.5° at end — flat, benign;
  tilt 4.7° mean / **17° at end and accelerating** (45° crossed during the
  terminal step). Episodes end in acute tip-over events (~every 3 s), not
  accumulated heading drift — the ori bound is effectively a fall detector,
  so our termination is already ≈ the paper's falls-only in practice. The
  heading obs (stage3) was aimed at a non-problem; kept in the contract for
  now (2 dims, harmless, plausibly useful for turning clips; revisit at the
  Phase 5 freeze). Joint acceptance stands at 0.074. NEXT DIAGNOSTIC
  (cheap, before any new training): log clip phase at termination + episode
  length distribution — lethal-clip-segment hypothesis (a specific stride
  event trips the robot) vs uniform stochastic stumbling; then inspect
  swing-foot clearance in that segment (the M1 foot-skate removal may have
  left low clearance, and the k=40 ee kernel pulls feet onto those paths).
- **stage2 eval** (512 eps, model_2200): joint err 0.271 → **0.0746** in
  one batch — the paper-alignment diagnosis was right (parameterization +
  gains + missing ee term, compounding). Action noise std collapsed to
  0.16 (policy confident in small ref-relative corrections; abs_mean
  0.53). Height err 0.0175, xy 0.101. Remaining failure is singular:
  yaw drifts ~2°/s until the 45° `root_ori` bound at ~3 s (mean ori err
  6.56°, was 2.53° — the merged kernel trades ori for pos+joints once yaw
  is past its gradient range). Ghost video:
  `logs/.../stage2_peng_trot/videos/play/rl-video-step-0.mp4`.
  Next decision (single change): (A) add heading-error obs — sin/cos of
  ref-relative yaw, contract 80→82, what Peng does (observes IMU yaw);
  vs (B) tilt-only termination letting yaw drift free — but that breaks
  the global pos reward/termination as the path curves (they'd need a
  heading-aligned reformulation). A is the smaller coherent change.
- **stage1e eval** (513 eps, model_1800): joint err 0.271 (= stage1d, as
  expected — no action/gain change), xy err 0.070, ori err 2.53° mean, yet
  **100% `root_ori` terminations**: with the position teleport fixed, the
  next binding constraint is heading drift — yaw error is unobservable to
  the actor (no yaw/heading obs, by contract; Peng 2020 observes IMU
  roll-pitch-yaw) and random-walks to the 45° bound in ~3 s. Partial
  mitigation exists (ref root vel is expressed in the current base frame,
  so a yawed robot sees a rotated velocity command); if `root_ori`
  terminations still dominate after stage2, adding a heading-error obs is
  the paper-sanctioned contract change to consider.
- **Eval gotcha (added 2026-07-18)**: `eval.py` builds the env from the
  CURRENT registered cfg defaults (`@hydra_task_config`), not the run's
  dumped `env.yaml` — after stage2 changed the defaults (kp, action
  centering, velocity_limit), evaluating older runs requires pinning their
  era's values via hydra overrides (see stage1e eval command) or the
  numbers are invalid.

- **stage1_trot**: learned to trot without falling within ~300 iters
  (`base_contact` terminations → ~0), then plateaued from ~iter 700
  (ep len ~210–230/500, reward ~61–66). Dominant termination: `root_pos`
  drift > 0.5 m (~20/24 episode ends) — small velocity-tracking errors
  integrate into xy drift that nothing in the reward corrects (no xy term
  by design, brief §3). Eval + ghost video to discriminate: good gait that
  drifts vs. subtly wrong gait.
- **stage1_trot eval** (model_2999, 514 eps): mean ep len 147 (29.4% of max),
  survival 0% — every episode ends by `root_pos` drift. Root tracking is
  good (ori err 2.5°, height err 1.7 cm, mean xy err 0.09 m) but **mean
  joint err 0.310 rad** (bar: < 0.15) — the gait shape itself is off, not
  just drifting. Deterministic ep len (147) < stochastic training ep len
  (218): exploration noise was masking a systematic velocity deficit —
  drift to 0.5 m in ~2.9 s ⇒ ~0.17 m/s mean velocity error. Ghost video
  (`logs/.../stage1_trot/videos/play/rl-video-step-0.mp4`, side-by-side
  ghost at +1 m y): qualitatively a real trot, upright and rhythmic —
  verdict is "good-looking gait that drifts", not a broken gait.
  Candidate single changes for run 2 (pick ONE): add a soft root xy
  position term (DeepMimic/Peng 2020 have a com-position term; w≈0.1,
  fold with height into full root-pos tracking — reward-only, does not
  touch the frozen obs contract), relax `term_root_pos_err` 0.5→1.0 m,
  or raise `rew_joint_pos_k`/weight.
- **stage1b eval** (516 eps): joint err 0.310→0.267, xy err 0.090→0.071,
  ori 2.52°→2.24°, height 0.017→0.032 (z-kernel softening side effect) —
  every tracking metric improved except height, yet survival still 0% with
  100% `root_pos` terminations and ep len ~151. Structural conclusion: the
  actor cannot observe accumulated drift (no world pos / base lin vel in
  obs, by design), so reward alone cannot cancel it; and the brief's own
  philosophy says xy drift should be free — the 0.5 m `root_pos`
  termination bound contradicts it and truncates every episode. Next
  lever: the termination rule, not the reward.
- **stage1c eval** (512 eps): joint err 0.347, ori 3.55°, xy 0.114 — all
  worse than stage1b; ep len 149 *at a 4× looser bound* ⇒ deterministic
  drift ~0.67 m/s (4× stage1b). Lesson (Peng 2018/2020, confirmed here):
  **the termination bound is itself the anti-drift learning signal** — the
  exp position kernel has no gradient beyond ~0.6 m, so with the bound at
  2.0 m nothing opposes drift and the policy stops caring, degrading all
  root motion. Bound reverted to 0.5 m. Note: eval episode lengths are not
  comparable across runs with different bounds (eval uses the run's cfg).
- **ROOT CAUSE FOUND (2026-07-18): cyclic wrap teleport.** Four runs with
  different rewards and bounds all died at ~150 steps (3.0 s) — episode
  length was insensitive to every change, including a 4× looser bound
  (stage1c died at 2.0 m in the same time 0.5 m runs died — impossible for
  a steady drift process). Cause: `MotionLib.get_frame` interpolated raw
  stored `root_pos` at `t % duration`, so at every loop wrap the reference
  position teleported back to the loop start (meters, instantly) — the
  `root_pos` termination then fired at the first wrap regardless of policy
  quality or bound. ~150 steps ≈ mean time-to-first-wrap under uniform RSI
  phase. All stage1a–d episode-length/survival numbers (and the "drift"
  narrative) are artifacts of this; per-frame tracking errors (joint, ori)
  remain valid. Fixed: per-loop xy displacement accumulated across wraps
  (z periodic), wrap-seam lerp continues into the next loop; regression
  test `test_cyclic_root_pos_accumulates_across_loops` (14/14 pass).
- **Tooling gotcha (cost ~2 render cycles)**: the Go2 USD is instanceable →
  the ghost's `collision_enabled=False` and transparent material silently
  fail; an overlaid ghost ejects the robot at every RSI reset. Ghost now
  renders side-by-side (`ghost_y_offset = 1.0`). First two stage-1 videos
  were invalid for this reason.
- **Same gotcha, second bite (2026-07-19)**: for clips whose path doubles
  back (canter's 167° pirouette), a 1 m lateral offset is not enough — the
  return leg drives the robot through the solid ghost, and the first
  stage7 canter video showed constant robot–ghost collisions. Videos only
  (train/eval spawn no ghost — metrics unaffected). Refilmed with
  `env.ghost_y_offset=4.0`; rule of thumb: offset > the clip's lateral
  path spread.
