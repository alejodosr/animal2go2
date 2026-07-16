"""Post-processing of retargeted motions (phase 4): what separates "demo"
from "usable".

Takes the raw retarget output (root trajectory + world-frame foot targets +
source-side contacts) and produces clean joint trajectories:

  1. Contact refinement: stance/swing segments shorter than a few frames are
     flicker from the threshold detector, not gait — merge them away.
  2. Smoothing: low-pass (Butterworth, ~7 Hz) on the foot targets and the
     root trajectory *before* IK. Smoothing joint angles after IK would drag
     stance feet around and reintroduce foot-skate; smoothing the targets and
     solving IK against the already-smooth root cannot.
  3. Ground alignment: shift everything in z so the median stance foot
     center sits at the foot-sphere radius (i.e. the sphere touches z=0).
  4. Foot-skate removal: during each stance segment the foot target is
     pinned to its touch-down xy at ground height, blended in/out over a few
     swing frames to avoid pops.
  5. IK + limit report: solve, clamp to the MJCF limits, report the clamp
     rate (a high rate means the scaling/offsets upstream are wrong).

Order matters: smooth first (a filter would smear the pins), then align,
then pin, then IK.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from retarget import ik

FOOT_RADIUS = 0.022        # foot sphere size in the MJCF: center z at contact
CUTOFF_HZ = 7.0            # Butterworth low-pass cutoff (brief: ~6-8 Hz)
MIN_SEGMENT_S = 0.05       # stance/swing runs shorter than this are flicker
BLEND_S = 0.05             # skate-removal blend in/out duration (~3 frames @ 60)


def foot_world_positions(root_pos, root_rot, foot_base):
    """Base-frame foot points (N, 4, 3) -> world frame."""
    return root_pos[:, None, :] + _apply_per_leg(root_rot, foot_base)


def foot_base_positions(root_pos, root_rot, foot_world):
    """World-frame foot points (N, 4, 3) -> base frame."""
    return _apply_per_leg(root_rot.inv(), foot_world - root_pos[:, None, :])


def refine_contacts(contacts, min_len):
    """Merge away stance/swing runs shorter than min_len frames.

    Short swing gaps are filled first (a 1-frame liftoff inside a stance is
    detector noise, and filling it keeps the foot pinned), then short stance
    blips are dropped.
    """
    out = contacts.copy()
    for leg in range(out.shape[1]):
        col = out[:, leg]
        for value in (False, True):  # fill short swings, then short stances
            for start, end, val in _runs(col):
                if val == value and end - start < min_len:
                    col[start:end] = not value
    return out


def lowpass(x, fps, cutoff=CUTOFF_HZ):
    """Zero-phase Butterworth low-pass along axis 0."""
    if cutoff >= 0.5 * fps:
        return x.copy()
    b, a = butter(2, cutoff / (0.5 * fps))
    return filtfilt(b, a, x, axis=0)


def smooth_rotations(rot, fps, cutoff=CUTOFF_HZ):
    """Low-pass a Rotation sequence via its quaternions.

    Sign-continuous quaternions are filtered componentwise and renormalized —
    valid for smoothing because consecutive frames are close on the sphere.
    """
    q = rot.as_quat()
    flip = np.cumprod(np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1.0, 1.0))
    q[1:] *= flip[:, None]
    q = lowpass(q, fps, cutoff)
    return Rotation.from_quat(q / np.linalg.norm(q, axis=-1, keepdims=True))


def ground_align(root_pos, foot_world, contacts):
    """Shift root + feet in z so the median stance foot touches the ground.

    Returns the offset that was subtracted (positive = motion was floating).
    """
    stance_z = foot_world[..., 2][contacts]
    offset = np.median(stance_z) - FOOT_RADIUS
    root_pos = root_pos.copy()
    foot_world = foot_world.copy()
    root_pos[:, 2] -= offset
    foot_world[..., 2] -= offset
    return root_pos, foot_world, offset


def pin_stance_feet(foot_world, contacts, blend):
    """Remove foot-skate: pin each stance segment to its touch-down point.

    During stance the target is (touch-down xy, ground height); the `blend`
    swing frames on either side interpolate between the free target and the
    pin so liftoff/touch-down don't pop. Blending only touches swing frames,
    so neighboring stance segments stay exactly pinned.
    """
    out = foot_world.copy()
    n = len(out)
    for leg in range(out.shape[1]):
        stance = contacts[:, leg]
        for start, end, val in _runs(stance):
            if not val:
                continue
            pin = np.array([*foot_world[start, leg, :2], FOOT_RADIUS])
            out[start:end, leg] = pin
            for k in range(1, blend + 1):  # ease in before touch-down
                i = start - k
                if i < 0 or stance[i]:
                    break
                w = (blend + 1 - k) / (blend + 1)
                out[i, leg] = w * pin + (1 - w) * foot_world[i, leg]
            for k in range(1, blend + 1):  # ease out after liftoff
                i = end - 1 + k
                if i >= n or stance[i]:
                    break
                w = (blend + 1 - k) / (blend + 1)
                out[i, leg] = w * pin + (1 - w) * foot_world[i, leg]
    return out


def skate_speed(foot_world, contacts, fps):
    """Mean horizontal speed (m/s) of feet during stance — 0 means no skate.

    Only interior stance frames count: the central difference at a
    touch-down/liftoff frame picks up legitimate swing speed, not skate.
    """
    interior = contacts.copy()
    interior[1:] &= contacts[:-1]
    interior[:-1] &= contacts[1:]
    if not interior.any():
        return 0.0
    vel = np.gradient(foot_world[..., :2], axis=0) * fps
    return float(np.linalg.norm(vel, axis=-1)[interior].mean())


def postprocess(motion, foot_targets_base):
    """Full phase-4 pipeline on a raw retargeted motion (at the source fps).

    motion: §7-format dict from retarget_clip.
    foot_targets_base: (N, 4, 3) raw IK targets in the base frame.
    Returns (motion, report) with cleaned root/dof/contacts; report carries
    the clamp rate, ground offset and stance-foot skate speeds before/after.
    """
    fps = motion["fps"]
    rot = Rotation.from_quat(motion["root_rot"])
    foot_world = foot_world_positions(motion["root_pos"], rot, foot_targets_base)

    contacts = refine_contacts(
        motion["foot_contacts"], max(2, round(MIN_SEGMENT_S * fps))
    )
    skate_before = skate_speed(foot_world, contacts, fps)

    foot_world = lowpass(foot_world, fps)
    root_pos = lowpass(motion["root_pos"], fps)
    rot = smooth_rotations(rot, fps)

    root_pos, foot_world, ground_offset = ground_align(root_pos, foot_world, contacts)
    foot_world = pin_stance_feet(foot_world, contacts, max(2, round(BLEND_S * fps)))
    foot_world[..., 2] = np.maximum(foot_world[..., 2], FOOT_RADIUS)  # no swing dips

    dof_pos, violated = ik.clamp_to_limits(
        ik.ik(foot_base_positions(root_pos, rot, foot_world))
    )
    foot_realized = foot_world_positions(root_pos, rot, ik.fk(dof_pos))
    skate_after = skate_speed(foot_realized, contacts, fps)

    out = dict(motion)
    out["root_pos"] = root_pos
    out["root_rot"] = rot.as_quat()
    out["dof_pos"] = dof_pos
    out["foot_contacts"] = contacts
    report = {
        "clamp_rate": violated.mean(),
        "violated": violated,
        "ground_offset": ground_offset,
        "skate_before": skate_before,
        "skate_after": skate_after,
    }
    return out, report


def _apply_per_leg(rot, points):
    """Apply one Rotation per frame to (N, 4, 3) points (scipy pairs the i-th
    rotation with the i-th vector only)."""
    return np.stack(
        [rot.apply(points[:, leg]) for leg in range(points.shape[1])], axis=1
    )


def _runs(mask):
    """Consecutive runs of a 1-D bool array as (start, end, value) triples."""
    edges = np.flatnonzero(np.diff(mask))
    bounds = np.concatenate([[0], edges + 1, [len(mask)]])
    return [(bounds[i], bounds[i + 1], bool(mask[bounds[i]]))
            for i in range(len(bounds) - 1)]
