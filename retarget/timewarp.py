"""Speed-aware time warp: feasibility projection, stage v1.

The retarget pipeline can emit motions whose root speed exceeds what the
robot can track (canter D1_010_KAN01_004: 4.5 m/s peak vs a ~3-3.5 m/s
ceiling from 30 rad/s joints x ~0.35 m legs — 2026-07-20 audit). Rather
than trimming those segments, this module reparameterizes time: wherever
the planar root speed exceeds a cap, the clip plays back slower, so every
frame of the source motion is kept and only the clock stretches. Joint and
root velocities scale down by the same local factor, so a warp that fixes
root speed also relaxes dof_vel demands.

Deliberately NOT corrected: vertical dynamics inside flight phases. A
slowed flight arc shows sub-ballistic gravity (az = -g * rate^2); fixing it
would mean resynthesizing z. The tracker's height kernel is soft and the
flights in accepted clips are short (<= 240 ms), so v1 accepts the error
and reports flight durations instead.

The warp factor is built from a smoothed speed *envelope* (running max),
so slowdowns begin before a burst and release after it, and the playback
rate itself is low-passed — no acceleration pops at segment boundaries.
Because of that smoothing the achieved peak can sit slightly above the
cap; callers should check the report, not assume.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation, Slerp

from retarget.postprocess import MIN_SEGMENT_S, lowpass, refine_contacts

# Practical Go2 tracking ceiling (2026-07-20 canter audit): 30 rad/s joint
# velocity x ~0.35 m effective leg length puts sustainable root speed at
# ~3-3.5 m/s; 3.2 leaves margin without slowing feasible gait.
SPEED_CAP = 3.2
SPEED_SMOOTH_HZ = 2.0  # planar-speed low-pass before the envelope
ENVELOPE_S = 0.15      # running-max half-window: slow down *before* the burst
RATE_SMOOTH_S = 0.15   # Gaussian sigma on the rate: pop-free transitions.
                       # Gaussian, not Butterworth: monotone step response and
                       # compact support, so feasible segments far from a burst
                       # stay at rate exactly 1 (no IIR ringing tails).


def playback_rate(root_pos, fps, cap=SPEED_CAP):
    """Per-frame playback rate in (0, 1]: 1 = real time, <1 = slowed.

    rate(t) = cap / envelope(speed_xy(t)) clamped to <= 1, where the
    envelope is a running max over +-ENVELOPE_S of the smoothed speed.
    """
    vel = np.gradient(root_pos[:, :2], axis=0) * fps
    speed = lowpass(np.linalg.norm(vel, axis=-1), fps, SPEED_SMOOTH_HZ)
    half = max(1, round(ENVELOPE_S * fps))
    envelope = maximum_filter1d(speed, size=2 * half + 1)
    rate = np.minimum(1.0, cap / np.maximum(envelope, 1e-9))
    rate = gaussian_filter1d(rate, RATE_SMOOTH_S * fps, mode="nearest", truncate=4.0)
    rate = np.clip(rate, 0.05, 1.0)
    rate[rate > 1.0 - 1e-9] = 1.0  # snap fp noise so unwarped spans resample as identity
    return rate


def flight_durations(contacts, fps):
    """Durations (s) of all-airborne runs, longest first."""
    airborne = ~contacts.any(axis=1)
    runs = []
    edges = np.flatnonzero(np.diff(airborne))
    bounds = np.concatenate([[0], edges + 1, [len(airborne)]])
    for i in range(len(bounds) - 1):
        if airborne[bounds[i]]:
            runs.append((bounds[i + 1] - bounds[i]) / fps)
    return sorted(runs, reverse=True)


def timewarp(motion, cap=SPEED_CAP):
    """Warp a §7 motion dict so planar root speed stays near `cap`.

    Every source frame is kept; time stretches locally by 1/rate. Output is
    resampled onto a uniform grid at the source fps (more frames, longer
    clip). Returns (motion_out, report); if nothing exceeds the cap the
    motion comes back with zero warp (identity resample).
    """
    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    contacts = np.asarray(motion["foot_contacts"], dtype=bool)
    n = len(root_pos)
    dt = 1.0 / fps

    rate = playback_rate(root_pos, fps, cap)

    # warped timestamps of the source frames: dtau = dt / rate (trapezoid)
    inv = 1.0 / rate
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * dt)])

    # resample onto a uniform grid in warped time, at the source fps; ceil so
    # the grid covers tau[-1] and the final source frame is kept (np.interp
    # clamps the past-the-end sample onto it)
    m = int(np.ceil(tau[-1] * fps - 1e-9)) + 1
    frame = np.interp(np.arange(m) / fps, tau, np.arange(n))  # fractional src frame

    idx = np.arange(n)
    out = dict(motion)
    out["root_pos"] = CubicSpline(idx, root_pos)(frame)
    out["dof_pos"] = CubicSpline(idx, dof_pos)(frame)
    out["root_rot"] = Slerp(idx, Rotation.from_quat(root_rot))(frame).as_quat()
    nearest = np.clip(np.round(frame).astype(int), 0, n - 1)
    out["foot_contacts"] = refine_contacts(
        contacts[nearest], max(2, round(MIN_SEGMENT_S * fps))
    )
    out["num_frames"] = m

    def planar_peak(p):
        v = np.gradient(p[:, :2], axis=0) * fps
        s = np.linalg.norm(v, axis=-1)
        return float(s.max()), float(lowpass(s, fps, SPEED_SMOOTH_HZ).max())

    peak_raw, peak_smooth = planar_peak(root_pos)
    peak_raw_w, peak_smooth_w = planar_peak(out["root_pos"])
    report = {
        "duration_before": (n - 1) * dt,
        "duration_after": (m - 1) * dt,
        "min_rate": float(rate.min()),
        "slowed_fraction": float((rate < 0.99).mean()),
        "planar_speed_peak_before": (peak_raw, peak_smooth),
        "planar_speed_peak_after": (peak_raw_w, peak_smooth_w),
        "dof_vel_peak_before": float(np.abs(np.diff(dof_pos, axis=0)).max() * fps),
        "dof_vel_peak_after": float(np.abs(np.diff(out["dof_pos"], axis=0)).max() * fps),
        "contact_fraction_before": float(contacts.mean()),
        "contact_fraction_after": float(out["foot_contacts"].mean()),
        "flights_before": flight_durations(contacts, fps),
        "flights_after": flight_durations(out["foot_contacts"], fps),
    }
    return out, report
