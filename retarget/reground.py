"""Support-aware re-grounding: feasibility projection for hidden terrain.

Retargeting flattens the source world to z = 0. When the source animal
stood on a raised object, the retargeted feet hover above the contact
threshold and the segment is labeled airborne — a physically impossible
sustained "flight" (jump clip D1_ex04_KAN02_003: 1740 ms at root z 0.43,
mean az -0.65 m/s^2 vs the -9.8 of real ballistics; the tracker rewarded
hovering). This module projects such segments back onto flat ground.

Detection is duration-based, not dynamics-based, on purpose: genuine
quadruped flights last <= ~350 ms, so any interval where even the LOWEST
foot stays above REGROUND_MIN_HEIGHT for REGROUND_MIN_S is elevated
support, not flight. (An az != -g test would misfire on time-warped clips,
whose real flights are legitimately sub-ballistic.)

The correction lowers root z by a smoothed per-frame offset that puts the
lowest foot back at ground height inside detected segments, ramping in and
out over ~2 sigma of a Gaussian so approach/landing stay pop-free; real
flights and grounded frames are untouched (offset 0 there by construction,
so clips without hidden terrain pass through bit-identical). Contacts are
relabeled from the corrected foot heights afterwards.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from retarget import ik
from retarget.postprocess import (
    FOOT_RADIUS,
    MIN_SEGMENT_S,
    foot_world_positions,
    relabel_contacts,
)

REGROUND_MIN_HEIGHT = 0.03  # m — lowest foot above this = no support at z=0
REGROUND_MIN_S = 0.5        # s — longer than any real quadruped flight
REGROUND_RAMP_S = 0.1       # Gaussian sigma for the offset ramps


def support_offset(foot_z, fps):
    """Per-frame ground offset (N,) from foot heights (N, 4).

    raw(t) = clearance of the LOWEST foot; frames where raw stays above
    REGROUND_MIN_HEIGHT for at least REGROUND_MIN_S form elevated-support
    segments. The returned offset follows raw inside those segments (lowest
    foot lands exactly at ground), ramps smoothly at their edges, and is 0
    elsewhere.
    """
    raw = np.maximum(0.0, foot_z.min(axis=1) - FOOT_RADIUS)
    elevated = raw > REGROUND_MIN_HEIGHT
    min_len = round(REGROUND_MIN_S * fps)
    target = np.zeros_like(raw)
    edges = np.flatnonzero(np.diff(elevated))
    bounds = np.concatenate([[0], edges + 1, [len(raw)]])
    segments = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if elevated[s] and e - s >= min_len:
            target[s:e] = raw[s:e]
            segments.append((int(s), int(e)))
    offset = gaussian_filter1d(target, REGROUND_RAMP_S * fps, mode="nearest")
    return np.minimum(offset, raw), segments


def reground(motion):
    """Re-ground a §7 motion dict. Returns (motion_out, report).

    Only root z and (via relabel) foot_contacts change; the motion comes
    back identical when no elevated-support segment is found.
    """
    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    rot = Rotation.from_quat(np.asarray(motion["root_rot"]))
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    contacts = np.asarray(motion["foot_contacts"], dtype=bool)

    feet = foot_world_positions(root_pos, rot, ik.fk(dof_pos))
    offset, segments = support_offset(feet[..., 2], fps)

    report = {
        "segments": [(s / fps, e / fps) for s, e in segments],
        "max_offset": float(offset.max()),
        "offset_fraction": float((offset > 1e-6).mean()),
        "contact_fraction_before": float(contacts.mean()),
        "contact_fraction_after": float(contacts.mean()),
    }
    if not segments:
        return motion, report

    out = dict(motion)
    out["root_pos"] = root_pos.copy()
    out["root_pos"][:, 2] -= offset
    feet_new = feet.copy()
    feet_new[..., 2] -= offset[:, None]
    out["foot_contacts"] = relabel_contacts(
        feet_new, max(2, round(MIN_SEGMENT_S * fps))
    )
    report["contact_fraction_after"] = float(out["foot_contacts"].mean())
    return out, report
