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
    # optional per-clip fields, included only when ALL clips provide them
    _OPTIONAL = ("feet_pos_root",)

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

        self._fields = list(self.FIELDS)
        for field in self._OPTIONAL:
            if all(getattr(c, field) is not None for c in clips):
                self._fields.append(field)
        for field in self._fields:
            flat = torch.cat([getattr(c, field) for c in clips], dim=0).to(self.device)
            setattr(self, field, flat)

        # per-loop xy root displacement for cyclic clips: without it the
        # reference teleports back to the loop start at every wrap, which
        # instantly exceeds any position-termination bound (RESULTS.md,
        # stage1a–d all died at the first wrap). Scaled by N/(N-1) so the
        # wrap segment (frame N-1 → frame 0 of the next loop) advances by the
        # clip's average per-frame step. z stays periodic (in-place gaits).
        first = self.offsets
        last = self.offsets + self.num_frames - 1
        loop_dp = self.root_pos[last] - self.root_pos[first]
        loop_dp[:, 2] = 0.0
        loop_dp *= (lengths / (lengths - 1).clamp(min=1)).to(loop_dp.dtype).unsqueeze(-1)
        self.loop_dp = torch.where(self.cyclic.unsqueeze(-1), loop_dp, torch.zeros_like(loop_dp))

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
        """(global_idx0, global_idx1, blend, loops, wrap_seg) for lookup.

        ``loops`` counts completed cycles of cyclic clips (0 for acyclic);
        ``wrap_seg`` marks lookups inside the frame N-1 → frame 0 seam, whose
        second endpoint belongs to the NEXT loop.
        """
        cyclic = self.cyclic[clip_idx]
        n = self.num_frames[clip_idx]
        dur = self.durations[clip_idx]
        loops = torch.where(cyclic, (t / dur).floor().clamp(min=0), torch.zeros_like(t))
        t = torch.where(cyclic, t % dur, t.clamp(min=torch.zeros_like(dur), max=dur))
        f = t * self.fps
        i0 = f.floor().long()
        blend = (f - i0).unsqueeze(-1)
        i1 = i0 + 1
        i0 = torch.where(cyclic, i0 % n, i0.clamp(max=n - 1))
        wrap_seg = cyclic & (i1 >= n)
        i1 = torch.where(cyclic, i1 % n, i1.clamp(max=n - 1))
        off = self.offsets[clip_idx]
        return off + i0, off + i1, blend, loops, wrap_seg

    def get_frame(self, clip_idx: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reference state at continuous time ``t`` (E,) into clips (E,).

        Linear interpolation between frames, slerp for the root quaternion,
        nearest frame for contacts.
        """
        g0, g1, blend = self._frame_indices(clip_idx, t)[:3]
        out: dict[str, torch.Tensor] = {}
        for field in self._fields:
            flat = getattr(self, field)
            if field == "root_rot":
                out[field] = quat_slerp(flat[g0], flat[g1], blend)
            elif field in self._NEAREST:
                out[field] = torch.where(blend < 0.5, flat[g0], flat[g1])
            elif flat.dim() == 3:  # (N, 4, 3) feet — broadcast blend over legs
                out[field] = torch.lerp(flat[g0], flat[g1], blend.unsqueeze(-1))
            else:
                out[field] = torch.lerp(flat[g0], flat[g1], blend)
        out["root_pos"] = self._unwrapped_root_pos(clip_idx, t)
        return out

    def _unwrapped_root_pos(self, clip_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Root position with loop displacement accumulated across wraps."""
        g0, g1, blend, loops, wrap_seg = self._frame_indices(clip_idx, t)
        dp = self.loop_dp[clip_idx]
        p1 = self.root_pos[g1] + torch.where(wrap_seg.unsqueeze(-1), dp, torch.zeros_like(dp))
        pos = torch.lerp(self.root_pos[g0], p1, blend)
        return pos + loops.unsqueeze(-1) * dp

    def phase(self, clip_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Normalized clip phase ∈ [0, 1) (E,) — sin/cos-encoded by the env."""
        dur = self.durations[clip_idx]
        cyclic = self.cyclic[clip_idx]
        p = torch.where(cyclic, (t % dur) / dur, (t / dur).clamp(max=1.0))
        return p

    def clip_done(self, clip_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """True where an acyclic clip has run past its last frame (E,) bool."""
        return ~self.cyclic[clip_idx] & (t >= self.durations[clip_idx])
