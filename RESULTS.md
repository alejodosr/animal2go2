# Milestone 2 — Phase 3 training runs

**Next up (planned, not run): `stage1e` = stage1d config + the wrap fix —
one change, first honest read of episode length/survival.** Before further
reward tuning: review action parameterization vs. Peng 2020 (see Open
items) and add action-saturation logging.

One row per run; one change per run (brief §4). Eval numbers from `eval.py`
(512 envs, deterministic, randomization off) on the run's final checkpoint.

| run | change vs. previous | iters | train reward | train ep len | eval mean joint err (rad) | eval ep len %max | outcome |
|---|---|---|---|---|---|---|---|
| `2026-07-17_14-09-42_stage1_trot` | — (baseline: single-clip trot `D1_009_KAN01_002`, brief §4 defaults) | 3000 | 65.7 | 218/500 | 0.310 | 29.4% | ✗ acceptance failed — see below |
| `2026-07-17_15-38-22_stage1b_trot_rootpos` | root height term → full 3D root-pos term (DeepMimic com: w 0.05→0.1, k 100→10, xy included) | 3000 | 80.4* | 226/500 | 0.267 | 30.3% | ✗ tracking −14%, but still 100% `root_pos` terminations (*reward not comparable: weights changed) |
| `2026-07-17_17-30-44_stage1c_trot_term2m` | `term_root_pos_err` 0.5 → 2.0 m (termination only; run-2 reward kept) | 3000 | 79.6* | 232/500 | 0.347 | 29.8% | ✗✗ **backfired** — drift 0.17→0.67 m/s, all metrics worse; REVERTED (*not comparable) |
| `2026-07-17_19-17-15_stage1d_trot_velx2` | bound back at 0.5 m; `rew_root_vel_w` 0.15 → 0.30 (velocity error is the drift source and is actor-observable via stance-leg dof_vel) | 3000 | 107.3* | 231/500 | 0.277 | 30.3% | ✗ xy err −20% (0.057), ori/height better, but ep len identical — led to finding the wrap bug (*not comparable) |

## Open items (2026-07-18)

- **Action reach analysis** (offline, trot clip): max |q_ref − default| =
  1.28 rad (thigh), mean 0.42 rad ⇒ with `action_scale = 0.25` the policy
  needs |a| ≈ 5 at the extremes (≈ 5σ at init noise 1.0). Peng 2020
  parameterizes targets around the *reference* pose, not a static default —
  prime suspect for the 0.27 rad joint-error plateau. Torque is NOT the
  binding constraint: RMS ≈ 3.7 N·m/joint vs 23.7 N·m limit.
- Reference dof_vel peaks 38.4 rad/s > Go2 ~30 rad/s (peaks only).
- No action-saturation logging (soft-limit clamp fraction, |a| histogram) —
  add to the env before more reward tuning.
- Speed: CPU governor `powersave` (20 cores) — set `performance` before
  runs; try 8192 envs (collection dominates: ~1.1 s vs 0.14 s learning per
  iteration at 4096).
- Episode-length note: the trot clip is 9.12 s and max episode 10 s ⇒ under
  the wrap bug NO episode could survive (must cross a wrap); explains 0%
  survival everywhere.

## Observations

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
