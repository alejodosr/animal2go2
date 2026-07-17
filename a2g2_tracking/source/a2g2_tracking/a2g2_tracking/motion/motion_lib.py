"""Multi-clip motion library: RSI sampling and batched reference lookup.

Pure torch, no Isaac imports. All per-env queries are vectorized: ``clip_idx``
and ``t`` are (E,) tensors, results are (E, ...) tensors on the lib's device.

Time convention: a clip with N frames at ``fps`` covers
  - acyclic: t ∈ [0, (N-1)/fps], clamped; past the end the clip is "done"
    (the env terminates the episode).
  - cyclic:  period N/fps — frame N wraps to frame 0, and lookups blend
    across the seam. ``t`` is taken modulo the period.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .motion_loader import GROUND_Z_OFFSET, MotionClip, load_motion, quat_slerp


class MotionLib:
    """Stacks clips into flat tensors for batched (clip, t) lookups."""

    FIELDS = ("root_pos", "root_rot", "dof_pos", "root_lin_vel", "root_ang_vel", "dof_vel", "foot_contacts")
    # contacts are binary — nearest-frame, never blended
    _NEAREST = ("foot_contacts",)

    def __init__(self, clips: list[MotionClip], cyclic: list[bool], device: str | torch.device = "cpu"):
        if len(clips) == 0:
            raise ValueError("empty motion library")
        if len(cyclic) != len(clips):
            raise ValueError("cyclic flags and clips length mismatch")
        fps = {c.fps for c in clips}
        if len(fps) != 1:
            raise ValueError(f"clips have mixed fps: {fps}")
        self.device = torch.device(device)
        self.fps = clips[0].fps
        self.num_clips = len(clips)
        self.names = [c.name for c in clips]

        self.cyclic = torch.tensor(cyclic, dtype=torch.bool, device=self.device)
        lengths = torch.tensor([c.num_frames for c in clips], dtype=torch.long, device=self.device)
        self.num_frames = lengths
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), lengths.cumsum(0)[:-1]])
        # playable duration per clip (seconds): cyclic clips own the wrap
        # segment frame N-1 → frame 0, acyclic ones end at their last frame
        self.durations = torch.where(self.cyclic, lengths / self.fps, (lengths - 1) / self.fps)

        for field in self.FIELDS:
            flat = torch.cat([getattr(c, field) for c in clips], dim=0).to(self.device)
            setattr(self, field, flat)

    @classmethod
    def from_files(
        cls,
        paths: list[str | Path],
        joint_names: list[str] | tuple[str, ...],
        cyclic: list[bool] | None = None,
        device: str | torch.device = "cpu",
        z_offset: float = GROUND_Z_OFFSET,
    ) -> "MotionLib":
        """Load pkls and build the lib. ``cyclic`` defaults to all-cyclic
        (our walk/trot/canter clips are gait loops)."""
        clips = [load_motion(p, joint_names, device=device, z_offset=z_offset) for p in paths]
        if cyclic is None:
            cyclic = [True] * len(clips)
        return cls(clips, cyclic, device=device)

    # -- sampling (reference state initialization) ---------------------------

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform over (clip, frame) pairs — i.e. duration-weighted over
        clips — for RSI. Returns (clip_idx (E,) long, t (E,) float seconds)."""
        u = torch.rand(n, device=self.device) * self.durations.sum()
        edges = self.durations.cumsum(0)
        clip_idx = torch.searchsorted(edges, u, right=True).clamp(max=self.num_clips - 1)
        t_start = edges - self.durations
        return clip_idx, u - t_start[clip_idx]

    # -- lookup --------------------------------------------------------------

    def _frame_indices(self, clip_idx: torch.Tensor, t: torch.Tensor):
        """(global_idx0, global_idx1, blend) for interpolated lookup."""
        cyclic = self.cyclic[clip_idx]
        n = self.num_frames[clip_idx]
        dur = self.durations[clip_idx]
        t = torch.where(cyclic, t % dur, t.clamp(min=torch.zeros_like(dur), max=dur))
        f = t * self.fps
        i0 = f.floor().long()
        blend = (f - i0).unsqueeze(-1)
        i1 = i0 + 1
        i0 = torch.where(cyclic, i0 % n, i0.clamp(max=n - 1))
        i1 = torch.where(cyclic, i1 % n, i1.clamp(max=n - 1))
        off = self.offsets[clip_idx]
        return off + i0, off + i1, blend

    def get_frame(self, clip_idx: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reference state at continuous time ``t`` (E,) into clips (E,).

        Linear interpolation between frames, slerp for the root quaternion,
        nearest frame for contacts.
        """
        g0, g1, blend = self._frame_indices(clip_idx, t)
        out: dict[str, torch.Tensor] = {}
        for field in self.FIELDS:
            flat = getattr(self, field)
            if field == "root_rot":
                out[field] = quat_slerp(flat[g0], flat[g1], blend)
            elif field in self._NEAREST:
                out[field] = torch.where(blend < 0.5, flat[g0], flat[g1])
            else:
                out[field] = torch.lerp(flat[g0], flat[g1], blend)
        return out

    def phase(self, clip_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Normalized clip phase ∈ [0, 1) (E,) — sin/cos-encoded by the env."""
        dur = self.durations[clip_idx]
        cyclic = self.cyclic[clip_idx]
        p = torch.where(cyclic, (t % dur) / dur, (t / dur).clamp(max=1.0))
        return p

    def clip_done(self, clip_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """True where an acyclic clip has run past its last frame (E,) bool."""
        return ~self.cyclic[clip_idx] & (t >= self.durations[clip_idx])
