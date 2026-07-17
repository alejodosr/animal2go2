# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments. Guarded: the motion subpackage is pure torch and
# must stay importable outside a running Isaac Sim (unit tests, Milestone 3
# tooling); isaaclab/omni only import inside the sim environment.
try:
    from .tasks import *

    # Register UI extensions.
    from .ui_extension_example import *
except ModuleNotFoundError as e:
    if e.name is None or not e.name.split(".")[0] in ("omni", "isaaclab", "isaaclab_tasks", "isaacsim", "carb", "pxr"):
        raise
