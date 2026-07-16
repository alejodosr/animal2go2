"""Source (dog) skeleton: BVH parsing, forward kinematics, keypoint extraction.

The AI4Animation SIGGRAPH 2018 dog dataset ships as BVH files:
  - Y-up, centimeters, 60 fps.
  - Rotation channels in ZXY order (read from the file, not assumed).
  - Front legs are the "arm" chains (Shoulder->Arm->ForeArm->Hand),
    rear legs the "leg" chains (UpLeg->Leg->Foot); toes are end sites.

Everything leaving this module is meters, Z-up. Conversions happen once,
at parse time (see `bvh_pos_to_zup_m`), and never again downstream.

Canonical leg order everywhere in this project: FR, FL, RR, RL
(Unitree convention: Front/Rear x Right/Left).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BVH_CM_TO_M = 0.01
LEG_ORDER = ["FR", "FL", "RR", "RL"]

# Dog joint names for each canonical keypoint. Toes are the end sites of these
# joints (FK exposes them as "<joint>_end"). Leg roots are the first mobile
# joint of each leg chain (shoulder ball / hip ball), used later to fit the
# trunk frame and to express foot targets per leg.
DOG_TOE_JOINTS = ["RightHand_end", "LeftHand_end", "RightFoot_end", "LeftFoot_end"]
DOG_LEG_ROOT_JOINTS = ["RightArm", "LeftArm", "RightUpLeg", "LeftUpLeg"]
DOG_ROOT_JOINT = "Hips"     # pelvis
DOG_CHEST_JOINT = "Spine1"  # front end of the trunk (shoulders attach here)


def bvh_pos_to_zup_m(pos):
    """BVH world position (Y-up, cm) -> project convention (Z-up, meters).

    Rotation of +90 deg about X: (x, y, z) -> (x, -z, y). Applied once at
    parse time.
    """
    pos = np.asarray(pos, dtype=np.float64)
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]
    out[..., 1] = -pos[..., 2]
    out[..., 2] = pos[..., 1]
    return out * BVH_CM_TO_M


def rot_yup_to_zup(rot):
    """Re-express a rotation given in the BVH Y-up world in the Z-up world."""
    r_x90 = Rotation.from_euler("x", 90, degrees=True)
    return r_x90 * rot * r_x90.inv()


def quat_xyzw_to_wxyz(q):
    """scipy quaternion order -> MuJoCo quaternion order."""
    q = np.asarray(q)
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q):
    """MuJoCo quaternion order -> scipy quaternion order."""
    q = np.asarray(q)
    return q[..., [1, 2, 3, 0]]


@dataclass
class BvhJoint:
    name: str
    offset: np.ndarray                 # (3,) local offset from parent, BVH units
    channels: list                     # e.g. ["Zrotation", "Xrotation", "Yrotation"]
    channel_index: int                 # index of first channel in the frame vector
    parent: int                        # index into BvhClip.joints, -1 for root
    end_offset: np.ndarray | None = None  # (3,) end-site offset, if leaf
    children: list = field(default_factory=list)


@dataclass
class BvhClip:
    name: str
    joints: list                       # list[BvhJoint], hierarchy (depth-first) order
    frames: np.ndarray                 # (num_frames, num_channels), degrees / BVH units
    frame_time: float

    @property
    def fps(self):
        return 1.0 / self.frame_time

    @property
    def num_frames(self):
        return len(self.frames)

    def joint_index(self, name):
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)


def parse_bvh(path):
    """Parse a BVH file into a BvhClip. No unit/axis conversion here."""
    tokens = Path(path).read_text().split()
    pos = 0

    def next_token():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def expect(tok):
        got = next_token()
        assert got == tok, f"{path}: expected {tok!r}, got {got!r}"

    expect("HIERARCHY")
    joints = []
    channel_count = 0

    def parse_joint(parent_index):
        nonlocal channel_count
        kind = next_token()               # ROOT / JOINT
        name = next_token()
        expect("{")
        expect("OFFSET")
        offset = np.array([float(next_token()) for _ in range(3)])
        expect("CHANNELS")
        n_ch = int(next_token())
        channels = [next_token() for _ in range(n_ch)]
        joint = BvhJoint(name=name, offset=offset, channels=channels,
                         channel_index=channel_count, parent=parent_index)
        channel_count += n_ch
        index = len(joints)
        joints.append(joint)
        if parent_index >= 0:
            joints[parent_index].children.append(index)

        while True:
            tok = next_token()
            if tok == "}":
                break
            if tok == "JOINT":
                pos_backup = pos - 1
                _reparse_joint(pos_backup, index)
            elif tok == "End":
                expect("Site")
                expect("{")
                expect("OFFSET")
                joint.end_offset = np.array([float(next_token()) for _ in range(3)])
                expect("}")
            else:
                raise ValueError(f"{path}: unexpected token {tok!r} in joint {name}")

    def _reparse_joint(token_pos, parent_index):
        nonlocal pos
        pos = token_pos
        parse_joint(parent_index)

    tok = next_token()
    assert tok == "ROOT", f"{path}: expected ROOT, got {tok!r}"
    pos -= 1
    parse_joint(-1)

    expect("MOTION")
    expect("Frames:")
    num_frames = int(next_token())
    expect("Frame")
    expect("Time:")
    frame_time = float(next_token())
    motion = np.array([float(t) for t in tokens[pos:]], dtype=np.float64)
    frames = motion.reshape(num_frames, channel_count)

    return BvhClip(name=Path(path).stem, joints=joints, frames=frames,
                   frame_time=frame_time)


def _local_rotation(joint, frames):
    """Rotation of `joint` relative to its parent for all frames."""
    rot_channels = [c for c in joint.channels if c.endswith("rotation")]
    order = "".join(c[0] for c in rot_channels)  # e.g. "ZXY"
    angles = np.stack(
        [frames[:, joint.channel_index + joint.channels.index(c)] for c in rot_channels],
        axis=-1,
    )
    # Uppercase seq = intrinsic: R = R_z @ R_x @ R_y for "ZXY", matching BVH.
    return Rotation.from_euler(order.upper(), angles, degrees=True)


def forward_kinematics(clip):
    """World positions for every joint and end site, all frames.

    Returns (positions, rotations):
      positions: dict name -> (N, 3) in meters, Z-up. End sites appear as
        "<joint>_end".
      rotations: dict name -> scipy Rotation (N,), world orientation, Z-up.
    """
    n = clip.num_frames
    positions = {}
    rotations = {}
    world_pos_bvh = {}   # joint index -> (N, 3) in raw BVH coords
    world_rot_bvh = {}

    for i, joint in enumerate(clip.joints):
        local_rot = _local_rotation(joint, clip.frames)
        offset = np.broadcast_to(joint.offset, (n, 3)).copy()

        pos_channels = [c for c in joint.channels if c.endswith("position")]
        if pos_channels:
            # In this dataset the root position channels are absolute: adding
            # the root OFFSET floats every clip above the ground by exactly
            # the OFFSET height. So the OFFSET is ignored when position
            # channels are present.
            trans = np.zeros((n, 3))
            for c in pos_channels:
                axis = "XYZ".index(c[0])
                trans[:, axis] = clip.frames[:, joint.channel_index + joint.channels.index(c)]
            offset = trans

        if joint.parent < 0:
            world_pos_bvh[i] = offset
            world_rot_bvh[i] = local_rot
        else:
            p_pos = world_pos_bvh[joint.parent]
            p_rot = world_rot_bvh[joint.parent]
            world_pos_bvh[i] = p_pos + p_rot.apply(offset)
            world_rot_bvh[i] = p_rot * local_rot

        positions[joint.name] = bvh_pos_to_zup_m(world_pos_bvh[i])
        rotations[joint.name] = rot_yup_to_zup(world_rot_bvh[i])

        if joint.end_offset is not None:
            end = world_pos_bvh[i] + world_rot_bvh[i].apply(
                np.broadcast_to(joint.end_offset, (n, 3)))
            positions[joint.name + "_end"] = bvh_pos_to_zup_m(end)

    return positions, rotations


def bone_list(clip):
    """(parent_name, child_name) pairs for drawing, end sites included."""
    bones = []
    for joint in clip.joints:
        if joint.parent >= 0:
            bones.append((clip.joints[joint.parent].name, joint.name))
        if joint.end_offset is not None:
            bones.append((joint.name, joint.name + "_end"))
    return bones


def extract_keypoints(clip):
    """Canonical keypoint trajectory for retargeting (meters, Z-up).

    Returns a dict:
      fps, num_frames, source,
      root_pos (N,3), root_rot_xyzw (N,4)   -- dog pelvis (Hips)
      chest_pos (N,3), chest_rot_xyzw (N,4) -- Spine1, front of the trunk
      toe_pos (N,4,3), leg_root_pos (N,4,3) -- leg order FR, FL, RR, RL
    """
    positions, rotations = forward_kinematics(clip)
    return {
        "fps": clip.fps,
        "num_frames": clip.num_frames,
        "source": clip.name,
        "root_pos": positions[DOG_ROOT_JOINT],
        "root_rot_xyzw": rotations[DOG_ROOT_JOINT].as_quat(),
        "chest_pos": positions[DOG_CHEST_JOINT],
        "chest_rot_xyzw": rotations[DOG_CHEST_JOINT].as_quat(),
        "toe_pos": np.stack([positions[j] for j in DOG_TOE_JOINTS], axis=1),
        "leg_root_pos": np.stack([positions[j] for j in DOG_LEG_ROOT_JOINTS], axis=1),
    }
