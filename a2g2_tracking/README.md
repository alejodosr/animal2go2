# a2g2_tracking — RL motion tracking for the Unitree Go2 (Milestone 2)

DeepMimic-style tracking of the Milestone 1 retargeted dog motions
(`motions/*.pkl`, §7 format) in Isaac Lab, trained with rsl_rl PPO.
Generated from the Isaac Lab external-project template (Direct workflow);
template docs kept below.

**Stack (pinned):** Isaac Lab 2.3.x @ `~/py_workspace/IsaacLab`, Isaac Sim
5.1.0 (pip venv `$A2G2_SSD/venvs/env_isaaclab`), torch 2.7.0+cu128,
driver 535.183.01, RTX 3090.

## Load-bearing constants

- sim dt = 0.005 s, decimation = 4 → control at 50 Hz = motion fps: exactly
  one reference frame per policy step (asserted at env init).
- Actor obs (80) and actions (12) are serialized in **canonical FR, FL, RR,
  RL** leg order; Isaac's joint order never leaks past the env boundary.
  Critic gets privileged extras (base lin vel, foot contacts, root height; 88).
- Reference targets (K=2 future frames) are expressed in the robot's
  **current base frame**, never world frame.
- Actuator: stock `UNITREE_GO2_CFG` DC motor PD — Kp 25.0, Kd 0.5, torque
  limit 23.5 N·m (goes into the Milestone 3 policy contract).
- `GROUND_Z_OFFSET` in `motion/motion_loader.py`: **0.0** — measured in
  Phase 2. The Isaac foot collider is a 0.022 m sphere, but a PD-held drop
  test settles the foot center at z = 0.0239 m (PhysX effective contact
  surface); the pkls replay stance centers at 0.0240 m, i.e. already aligned
  to 0.1 mm. Do not "fix" the offset from the geometric radius — that pushes
  feet into contact and depenetration lifts the whole robot ~2 mm.

## Phase 2 workflow

```bash
source $A2G2_SSD/venvs/env_isaaclab/bin/activate
cd ~/py_workspace/animal2go2

# kinematic replay gate (no policy): puppets the robot through the reference,
# prints ground z-offset stats; --video records to media/replay_<clip>/
python a2g2_tracking/scripts/rsl_rl/play.py --task Template-A2g2-Tracking-Direct-v0 \
    --replay --motion D1_009_KAN01_002 --headless [--video --video_length 300]

# train / play (logs → repo logs/ symlink, SSD-backed)
python a2g2_tracking/scripts/rsl_rl/train.py --task Template-A2g2-Tracking-Direct-v0 --headless
python a2g2_tracking/scripts/rsl_rl/play.py  --task Template-A2g2-Tracking-Direct-v0 --num_envs 32
```

---

# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/a2g2_tracking

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/a2g2_tracking/a2g2_tracking/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/a2g2_tracking"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```