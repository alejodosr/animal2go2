# animal2go2 — Milestone 2: RL tracking policy in Isaac Lab

Brief for Claude Code. Continues from Milestone 1 (kinematic retargeting, done:
`motions/*.pkl` in §7 format, 50 Hz, foot-skate removed, clamp rates logged).

**Goal:** a PPO policy trained in Isaac Lab that tracks the retargeted dog
motions on the Unitree Go2 under physics — DeepMimic-style motion imitation
with reference state initialization and early termination. Deliverables: a
new package `a2g2_tracking/`, trained checkpoints, per-clip tracking metrics,
side-by-side videos (reference ghost vs. policy), and an exported policy +
frozen observation/action contract for Milestone 3 (MuJoCo sim2sim).

**Hardware:** single RTX 3090 (24 GB, Ampere sm_86 — fully supported),
Ubuntu 22.04/24.04.

**Design principle carried through everything:** the policy's observation and
action interface must be simulator-agnostic. No Isaac-specific state, no
privileged sim internals in the actor's observations. Milestone 3 ports this
policy to MuJoCo; every Isaac-flavored thing that leaks into the interface is
a bug we pay for later.

---

## §0 Conventions (read before writing any code)

Three convention mismatches between our pipeline, MuJoCo, and Isaac Lab. All
three have already bitten once in Milestone 1 (see its README). Do not index
by position anywhere; always map by joint *name*.

| Thing | Ours (§7 pkl) | MuJoCo Go2 | Isaac Lab Go2 |
|---|---|---|---|
| Quaternion | **xyzw** | wxyz | **wxyz** (`root_quat_w`) |
| Leg order | FR, FL, RR, RL | FL, FR, RL, RR (qpos order) | **breadth-first by USD traversal** — typically all hips, then all thighs, then all calves (e.g. `FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …`), NOT grouped per leg |
| Up axis / units | Z-up, meters, 50 Hz | Z-up, meters | Z-up, meters |

- **Trap:** Isaac Lab's `Articulation.joint_names` is the single source of
  truth for joint order. Build a name-based index map at startup
  (`{name: idx}`) and assert its length is 12. Write a unit test that
  round-trips a canonical dof vector through the map.
- **Trap:** quaternion convention. Our pkls store xyzw (video2robot §7).
  Isaac Lab wants wxyz. Convert once, at load time, inside the motion loader
  — never at use sites.
- Ground reference: our pkls were z-aligned so the foot *sphere* touches
  z = 0 in the MuJoCo model. Isaac's Go2 USD collision geometry differs
  slightly. Expect a constant few-mm root-height offset; measure it in
  Phase 2 (drop the robot at frame 0 and check penetration/float) and apply
  a single z-shift constant rather than re-exporting motions.

---

## §1 Phase 0 — Isaac Lab installation & smoke test

Use **Isaac Lab 2.3.x stable** with **Isaac Sim 5.1.0** via pip. Do NOT use
the Isaac Lab 3.0 beta / develop branch (Isaac Sim 6.0, Newton backend) — it
is in flux and the tracking task doesn't need it.

### Disk layout — set up BEFORE installing anything

Hard rule: **code lives in `$HOME`, everything heavy lives on the secondary
SSD.** Heavy = the venv (the isaacsim wheels alone are ~20 GB installed), the
Omniverse/Kit extension + shader caches, the Nucleus asset cache, pip/uv
caches, training logs & checkpoints, rendered media, and datasets.

**STATUS: this section is already DONE on this machine.** Phase 0 install is
complete and validated (Go2 velocity task trains at ~80k steps/s). Do not
re-create the venv or re-run the installs; the commands below are kept for
reproducibility/articles. What Claude Code needs is the map of where
everything lives:

```
SSD mount              /media/SHARED_DATA            (ext4, own partition — verify with: findmnt -T /media/SHARED_DATA)
A2G2 root              /media/SHARED_DATA/postcapitalistrobots/a2g2
venv                   $A2G2_SSD/venvs/env_isaaclab
Isaac Sim              $A2G2_SSD/venvs/env_isaaclab/lib/python3.11/site-packages/isaacsim
                       (pip-installed INSIDE the venv — there is no separate
                       ~/.local/share/ov/pkg binary install on this machine)
IsaacLab clone         ~/py_workspace/IsaacLab       (source checkout, code-only, in home)
animal2go2 repo        in home (code-only; data/, media/, logs/ are SSD symlinks)
Kit/Omniverse caches   ~/.cache/ov, ~/.local/share/ov, ~/.nvidia-omniverse → symlinks into $A2G2_SSD/caches/
```

Environment variables — persist in `~/.bashrc` (and export in any
non-interactive shell Claude Code spawns):

```bash
export A2G2_SSD=/media/SHARED_DATA/postcapitalistrobots/a2g2
export UV_CACHE_DIR=$A2G2_SSD/caches/uv
export PIP_CACHE_DIR=$A2G2_SSD/caches/pip
export OMNI_KIT_ACCEPT_EULA=YES

# Optional, only if a tool asks "where is Isaac Sim":
export ISAACSIM_PATH=$A2G2_SSD/venvs/env_isaaclab/lib/python3.11/site-packages/isaacsim
export ISAACSIM_PYTHON_EXE=$A2G2_SSD/venvs/env_isaaclab/bin/python
```

**Important:** because Isaac Sim is pip-installed, *the activated venv IS the
Isaac Sim selection mechanism*. `./isaaclab.sh` picks up whatever python is
active; no `ISAACSIM_PATH` is required for normal use (it exists for tooling
that expects the binary-install layout, e.g. some VSCode configs). The one
non-negotiable is therefore: **always activate the venv first** —

```bash
source $A2G2_SSD/venvs/env_isaaclab/bin/activate
```

— and with conda `base` deactivated (`conda deactivate`; ideally
`conda config --set auto_activate_base false`). A `(env_isaaclab) (base)`
double prompt has already caused one mis-installed package on this machine.

- **Trap (already hit and fixed, documented for the article):** the SSD's
  fstab entry originally used the `user` option, which implies `noexec` —
  Isaac Sim then fails at startup with
  `libcarb.so: failed to map segment from shared object`. Fixed by replacing
  the fstab line with `UUID=… /media/SHARED_DATA ext4 defaults,nofail 0 2`.
  If that error ever reappears, check
  `findmnt -T /media/SHARED_DATA -o OPTIONS` for `noexec` before debugging
  anything else.
- **Trap (already hit and fixed):** old sdists (`flatdict`) fail to build
  under new setuptools with `No module named 'pkg_resources'`. Fix:
  `pip install "setuptools<81" wheel && pip install <pkg> --no-build-isolation`.

Original setup commands (reference only — do not re-run):

Ask the user for the actual SSD mount point first; use `$A2G2_SSD` as the
placeholder throughout (e.g. `/mnt/ssd/a2g2`). Verify it's really a separate
filesystem: `df -h $A2G2_SSD` must show a different device than `df -h ~`.

```bash
# --- one-time setup ---
export A2G2_SSD=/mnt/ssd/a2g2        # <- confirm with user, then persist in ~/.bashrc
mkdir -p $A2G2_SSD/{venvs,caches/{ov,ov_data,pip,uv,omniverse_logs},logs,media,data}

# Omniverse/Kit caches: Kit writes to fixed paths under $HOME regardless of
# env vars, so symlinks are the only bulletproof redirect. Create them BEFORE
# the first isaacsim launch (Kit creates these dirs on first run otherwise).
mkdir -p ~/.cache ~/.local/share
ln -sfn $A2G2_SSD/caches/ov              ~/.cache/ov              # extension + shader cache (the big one)
ln -sfn $A2G2_SSD/caches/ov_data         ~/.local/share/ov        # kit data, downloaded exts
ln -sfn $A2G2_SSD/caches/omniverse_logs  ~/.nvidia-omniverse      # kit/isaac logs

# pip + uv caches honor env vars — persist these in ~/.bashrc too
export UV_CACHE_DIR=$A2G2_SSD/caches/uv
export PIP_CACHE_DIR=$A2G2_SSD/caches/pip
```

Code and repos (small, fast-access, backed up with home):

```
~/py_workspace/IsaacLab/       # vanilla clone, ~2 GB, fine in home
<animal2go2 repo in home>      # our repo incl. a2g2_tracking/ — CODE ONLY
```

Repo-internal heavy dirs become symlinks into the SSD so `git status` and
editors stay in home but bytes land on the SSD:

```bash
cd <animal2go2 repo>
mkdir -p $A2G2_SSD/{data,media,logs}/animal2go2
ln -sfn $A2G2_SSD/data/animal2go2   data      # mocap zips + processed npz
ln -sfn $A2G2_SSD/media/animal2go2  media     # rendered mp4s
ln -sfn $A2G2_SSD/logs/animal2go2   logs      # rsl_rl runs: checkpoints, tb, videos
# data/, media/, logs/ are already gitignored — keep it that way
```

- **Trap:** create the `~/.cache/ov` symlink before the *first* `isaacsim`
  launch. If Kit has already created a real directory there, move its
  contents to the SSD first (`mv ~/.cache/ov/* $A2G2_SSD/caches/ov/`), then
  replace the dir with the symlink.
- **Trap:** `ln -sfn`, not `ln -sf`, when the target may already exist as a
  directory — otherwise the link is created *inside* it.
- Optional but cheap: `export TMPDIR=$A2G2_SSD/tmp` (mkdir it) during the
  pip install step; some wheels unpack multi-GB temp files into `/tmp`.

**Verification (run after the Phase 0 smoke tests, gate for phase exit):**

```bash
du -sh ~/.cache/ov/ ~/.local/share/ov/ $A2G2_SSD/venvs/ 2>/dev/null  # big numbers, on SSD
df -h ~ ; df -h $A2G2_SSD    # home usage must be ~unchanged vs. pre-install
find ~ -maxdepth 3 -size +1G -not -path "*/code/IsaacLab/*" 2>/dev/null  # should return nothing new
```

Prerequisites to verify first (fail fast with clear messages):

```bash
nvidia-smi                 # driver ≥ 535 recommended for Isaac Sim 5.x
ldd --version              # GLIBC ≥ 2.35 (Ubuntu 22.04+ ok; 20.04 is NOT)
df -h .                    # ~50 GB free (isaacsim wheels + ext cache + assets)
```

Install (uv, Python 3.11 — Isaac Sim 5.x is built against 3.11 exactly):

```bash
# 1. Environment — the venv goes on the SSD (isaacsim wheels ≈ 20 GB installed)
uv venv --python 3.11 --seed $A2G2_SSD/venvs/env_isaaclab
source $A2G2_SSD/venvs/env_isaaclab/bin/activate
uv pip install --upgrade pip

# 2. Isaac Sim via pip (the [all,extscache] extras pre-cache kit extensions
#    — without extscache the first launch downloads for ~10 min)
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 3. CUDA-enabled PyTorch pinned to what Isaac Lab 2.3.x expects
#    (cu128 wheels run fine on Ampere/3090)
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Isaac Lab FROM SOURCE (the isaaclab pip package does NOT ship the
#    training scripts; we want the repo). Code → home, per the disk layout.
#    ALREADY DONE on this machine at ~/py_workspace/IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git ~/py_workspace/IsaacLab
cd ~/py_workspace/IsaacLab
git checkout <latest v2.3.x tag>     # pin it; record in A2G2 README
./isaaclab.sh --install rsl_rl       # only the framework we use

# 5. Headless EULA acceptance (needed on servers / in scripts)
export OMNI_KIT_ACCEPT_EULA=YES
```

Smoke tests, in order — each is a gate:

```bash
# a) Isaac Sim launches at all (first run pulls remaining extensions)
isaacsim

# b) Isaac Lab sees the GPU and can create an empty sim
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# c) The Go2 asset loads and a known-good task trains. This validates the
#    whole stack AND downloads the Go2 USD from Nucleus/S3 into the cache.
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
    --max_iterations 50
```

- **Trap:** asset loading. Isaac Lab assets live on AWS S3; first load of the
  Go2 USD can take minutes. If the network is slow, enable local asset
  caching per the Isaac Lab docs before blaming the code.
- **Trap:** if `import isaacsim` segfaults, check for a system
  `LD_LIBRARY_PATH` pointing at a different CUDA. Clean env vars.
- Record in the README: exact Isaac Lab tag, Isaac Sim version, driver
  version, torch version. Article fodder and reproducibility both.

**Acceptance:** velocity task runs 50 iterations headless without error, and
reward is increasing; the disk-layout verification block passes (home
partition usage ~unchanged, all heavy dirs resolve to the SSD). Commit
nothing to the IsaacLab clone; it stays vanilla.

---

## §2 Phase 1 — Project scaffolding & motion library

Create an **external project** so we never patch the IsaacLab tree:

```bash
cd IsaacLab && ./isaaclab.sh --new
# template: External project, Direct workflow, rsl_rl
# name: a2g2_tracking, place it inside the animal2go2 repo
```

Layout inside animal2go2:

```
a2g2_tracking/
├── a2g2_tracking/
│   ├── motion/
│   │   ├── motion_loader.py     # §7 pkl → GPU tensors, conventions fixed here
│   │   └── motion_lib.py        # multi-clip sampling, phase lookup
│   ├── tasks/
│   │   └── tracking_env.py      # DirectRLEnv
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py
│   └── export/
│       └── policy_contract.py   # obs/action spec → JSON (Milestone 3 input)
├── scripts/  (train.py, play.py, eval.py — thin wrappers)
└── tests/
```

`motion_loader.py` responsibilities (all conversions live HERE and only here):

1. Load §7 pkl: `fps`, `root_pos (N,3)`, `root_rot (N,4) xyzw`,
   `dof_pos (N,12)` in canonical FR, FL, RR, RL order.
2. Convert quat → wxyz. Reorder dofs canonical → Isaac joint order via the
   name map. Apply the global z-offset constant (measured in Phase 2).
3. Compute derivatives by central finite differences: `root_lin_vel`,
   `root_ang_vel` (from quaternion differences — use the proper log-map, not
   naive per-component diff), `dof_vel`. Smooth nothing — Milestone 1 already
   smoothed; double smoothing lags contacts.
4. Precompute per-frame **foot contact flags** from the Milestone 1 stance
   segments if present in the pkl; otherwise recompute with the same
   velocity+height heuristic. Needed for the contact-matching reward.
5. Everything to `torch` tensors on the sim device once, at init.

`motion_lib.py`: uniform sampling of (clip, frame) pairs for reference state
initialization; `get_frame(clip_idx, t)` with linear interpolation between
frames (slerp for the root quaternion); cyclic clips (walk/trot loops) wrap,
acyclic ones clamp-and-terminate.

- **Trap:** our motions are 50 Hz. Choose sim `dt = 0.005 s` (200 Hz physics)
  with `decimation = 4` → control at 50 Hz, exactly one reference frame per
  policy step. This removes an entire class of interpolation bugs. Write it
  down as a load-bearing constant.
- **Trap:** `--new` template names and registration (gym id like
  `A2G2-Tracking-Go2-v0`) — follow whatever the current template generates,
  don't fight it.

**Acceptance:** unit tests pass for (a) name-map round trip, (b) xyzw→wxyz on
a known quaternion, (c) velocities of a synthetic constant-velocity clip,
(d) loading every pkl in `motions/` without NaN.

---

## §3 Phase 2 — Tracking environment (DirectRLEnv)

Direct workflow, not manager-based — tracking rewards are tightly coupled to
the reference lookup and are clearer as one class.

**Scene:** flat plane, N Go2s from `UNITREE_GO2_CFG`
(`isaaclab_assets.robots.unitree`). Keep its actuator config (implicit PD).
Record the Kp/Kd it ships with into the policy contract JSON — MuJoCo must
replicate them in Milestone 3.

**Kinematic replay gate (do this before any RL):** a `--replay` mode on
`play.py` that force-sets root state + joint states from the reference every
step (physics kinematic puppet). This is the Phase 2 acceptance test: if the
replay looks wrong in Isaac, conventions are wrong, and no reward will fix
them. Use it to measure the ground z-offset from §0. Ghost-visualize with a
second, gravity-disabled, collision-disabled Go2 in a transparent material —
this ghost is reused in Phase 4 videos.

**Episode logic:**

- **Reference State Initialization (RSI):** every reset samples a random
  (clip, frame) and initializes the robot exactly there (root pose+vel, joint
  pos+vel). Without RSI, DeepMimic-style training mostly fails — the policy
  never sees the middle of a canter until it can survive the start.
  Add small noise (joint pos ±0.05 rad, root z +0–2 cm) for robustness.
- **Early termination:** terminate when (a) any tracking error exceeds a
  loose bound (root pos error > 0.5 m OR root orientation error > 45° OR
  mean joint error > 1.0 rad), (b) base/trunk contact with ground,
  (c) reference clip ends (acyclic). Early termination is not a nicety; it
  is half the learning signal (it truncates hopeless rollouts).
- **Phase tracking:** each env carries `(clip_idx, ref_t)`; `ref_t` advances
  by policy dt per step. Cyclic clips wrap `ref_t`.

**Observations (actor) — the sim-agnostic contract, frozen at end of phase:**

```
proj_gravity (3)              # base frame
root_ang_vel (3)              # base frame  — IMU-available
dof_pos - default (12)        # Isaac order, but SERIALIZED in canonical
dof_vel (12)                  #   FR,FL,RR,RL order in the contract
prev_action (12)
ref_target_t+1 … t+K:         # K=2 future reference frames, each:
  ref_dof_pos (12)
  ref_root_lin_vel, ref_root_ang_vel in CURRENT base frame (6)
phase (2)                     # sin/cos of normalized clip phase (cyclic only)
```

Deliberately excluded from the actor: base linear velocity (not directly
measurable on hardware; keep the door to real-robot deployment open) and any
world-frame positions. If asymmetric actor-critic is easy in the template,
give the **critic** privileged extras (base lin vel, foot contact states,
true root height); if it complicates things, skip it — symmetric is fine for
this milestone.

- **Trap:** express reference targets **relative to the robot's current base
  frame**, never in world frame. World-frame targets make the policy a
  position controller that breaks the moment the robot drifts; relative
  targets make tracking recoverable and transfer-friendly.
- **Trap:** observation normalization. rsl_rl's empirical normalizer is fine,
  but its running mean/std become part of the exported policy — they go in
  the contract JSON too.

**Actions:** 12 joint position targets = `default_pose + action_scale * a`,
`action_scale = 0.25` to start, tanh-squashed or clipped to joint limits.
PD via the articulation's implicit actuator. No torque control (harder to
port, harder to train).

**Rewards** (multiplicative exp-kernel style, DeepMimic weights as a start):

| term | form | weight |
|---|---|---|
| joint pos tracking | exp(−5·‖q − q_ref‖²) | 0.5 |
| joint vel tracking | exp(−0.1·‖q̇ − q̇_ref‖²) | 0.05 |
| root orientation | exp(−10·quat_err²) | 0.15 |
| root velocity (lin+ang) | exp(−2·‖v − v_ref‖²) | 0.15 |
| root height | exp(−100·(z − z_ref)²) | 0.05 |
| foot contact match | mean(contact == ref_contact) | 0.1 |
| action rate penalty | −0.01·‖aₜ − aₜ₋₁‖² | add |
| torque penalty | −1e−4·‖τ‖² | add |

Root *xy position* tracking is deliberately soft/absent — velocity + heading
tracking with free xy drift is much more robust and matches how these
policies deploy. Log every term separately from day one; the first week of
training debugging is entirely reading per-term reward curves.

**Acceptance:** kinematic replay of trot and walk looks correct in the Isaac
viewer (feet don't skate, robot doesn't float/penetrate); env steps 4096
parallel instances headless at > 50k steps/s aggregate; reward terms all
finite on random actions.

---

## §4 Phase 3 — Training (rsl_rl PPO)

Start config: 4096 envs (3090/24 GB handles this on flat terrain), 24-step
rollouts, PPO defaults from the Go2 velocity task as the base, entropy coef
0.005–0.01, lr 1e-3 adaptive.

Curriculum — one thing at a time:

1. **Single clip, trot** (`D1_009_…_002`, the cleanest cyclic gait). Target:
   survives full loops with mean joint error < 0.15 rad. Expect a few
   thousand iterations, roughly 1–3 h on the 3090.
2. **Add walk**, multi-clip sampling. The clip-conditioning path (phase +
   ref targets) proves itself here.
3. **Canter last.** Flight phases are genuinely harder; expect it to need
   more iterations and possibly a bumped contact-match weight. Failure here
   is a good article section, not a blocker.

Basic randomization ON from the start (these are cheap and prevent
overfitting to PhysX quirks — they are also the Milestone 3 insurance):
friction ∈ [0.5, 1.25], added base mass ∈ [−1, 3] kg, small random pushes
every 5–10 s. Motor strength / gains randomization deferred to Phase 5.

Practicalities:

- `--headless` always for training; `--video` flag (records periodic rollout
  clips to the log dir) for sanity — cheaper than babysitting a viewer.
- TensorBoard or wandb; log per-term rewards, episode length, and
  termination-cause histogram. A rising "root orientation termination" count
  says more than the total reward ever will.
- Checkpoint every ~200 iters, keep the config YAML next to the checkpoint.
- **All run output goes through the repo's `logs/` symlink** (SSD-backed, per
  the disk-layout section) — set rsl_rl's experiment/log dir there explicitly
  in the agent cfg; do not accept the default of writing into the IsaacLab
  tree or the package dir. Same for `--video` recordings. (The Phase 0 smoke
  test already left a throwaway run in
  `~/py_workspace/IsaacLab/logs/rsl_rl/unitree_go2_flat/` — safe to delete.)
- **Trap:** reward scales interact with episode length. If mean episode
  length is pinned at the max, early termination bounds are too loose (the
  policy is being rewarded for surviving, not tracking). If pinned near
  zero after 500+ iters, RSI or conventions are broken — go back to replay.
- **Trap:** don't tune more than one reward weight between runs. Keep a
  RESULTS.md table: run name, change, outcome, per-clip error. That table IS
  the article.

**Acceptance:** trot policy tracks with mean joint error < 0.15 rad and mean
episode length > 90% of max on 512 eval envs; walk and trot from one
multi-clip policy.

---

## §5 Phase 4 — Evaluation & article assets

`eval.py` (deterministic policy, no exploration noise) computes per clip over
≥ 512 episodes:

- mean/max joint tracking error (rad), per-joint breakdown
- root height error, orientation error (deg)
- velocity tracking error, xy drift per gait cycle
- contact timing F1 vs. reference contacts
- survival rate, mean episode length

Output one markdown table over all clips (the Milestone 1 summary-table
pattern — same spirit, physics edition).

`play.py` renders: policy Go2 + transparent reference ghost, side by side or
overlaid, camera following, → `media/rl_<clip>.mp4`. These videos next to the
Milestone 1 kinematic ones are the core visual of the article ("kinematics
says it's possible; physics disagrees; RL negotiates").

**Acceptance:** metrics table generated by one command; at least trot + walk
videos exported.

---

## §6 Phase 5 — Robustness & export (the Milestone 3 handoff)

1. **Extended randomization**, then fine-tune or retrain: motor Kp/Kd ±20 %,
   friction range widened, comms delay of 1 policy step (action buffer),
   observation noise (gyro 0.02 rad/s, joint pos 0.01 rad, joint vel
   0.5 rad/s). This is cheap sim2sim/sim2real insurance and its ablation
   (with vs. without) is a ready-made article experiment for Milestone 3.
2. **Export** via rsl_rl's exporter → TorchScript **and** ONNX. Run the
   exported artifact on CPU on 100 recorded observation vectors and assert
   outputs match the live policy to 1e-5.
3. **Freeze the policy contract** — `policy_contract.json`:
   - obs layout (names, sizes, order, canonical leg order used in
     serialization), action layout and scale, default joint pose
   - normalization mean/std arrays
   - control rate (50 Hz), PD gains, torque limits
   - quaternion convention of every field
   - motion pkl(s) the eval numbers refer to, checkpoint hash

   Milestone 3's MuJoCo runner consumes ONLY the ONNX + this JSON + the
   motion pkls. If it needs to import anything from Isaac Lab, Phase 5 has
   failed.

**Acceptance:** exported ONNX reproduces live policy outputs; contract JSON
validates against a schema test; a `README` section documents the full
train→eval→export loop in ≤ 10 commands.

---

## §7 Known references (for design questions, not for vendoring)

- Peng et al. 2020, "Learning Agile Robotic Locomotion Skills by Imitating
  Animals" — the reward structure, RSI, and early termination here follow it.
- DeepMimic (Peng et al. 2018) — exp-kernel tracking rewards.
- AMP (Peng et al. 2021) — explicitly OUT of scope for this milestone;
  tracking-based imitation is more debuggable and more explainable in
  articles. AMP is a candidate Milestone 4 comparison.
- Isaac Lab's `Isaac-Velocity-Flat-Unitree-Go2-v0` — PPO hyperparameter and
  actuator-config baseline.

## §8 Out of scope (do not drift)

- No AMP / adversarial anything.
- No terrain other than flat plane.
- No video-based pose extraction (that's Milestone 3+ front-end work).
- No real-robot deployment code.
- No Isaac Lab 3.0 / Newton backend migration.
