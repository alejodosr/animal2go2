"""Pinhole camera for the rendered evaluation tier (mocap 3D -> 2D).

Conventions: world is meters/Z-up (the seam's frame); camera follows OpenCV
(x right, y down, z forward; pixels u right, v down, origin top-left).
Static camera only (v0 scope decision Q4).
"""

import numpy as np


class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, R, t, width, height):
        """R (3,3), t (3,): p_cam = R @ p_world + t."""
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.width, self.height = int(width), int(height)

    @classmethod
    def look_at(cls, eye, target, fov_x_deg=50.0, width=1280, height=720,
                up=(0.0, 0.0, 1.0)):
        """Camera at `eye` looking at `target`, horizontal FOV, square pixels."""
        eye = np.asarray(eye, dtype=np.float64)
        z = np.asarray(target, dtype=np.float64) - eye
        z = z / np.linalg.norm(z)
        x = np.cross(z, np.asarray(up, dtype=np.float64))
        nx = np.linalg.norm(x)
        if nx < 1e-8:
            raise ValueError("look_at: view direction parallel to up")
        x = x / nx
        y = np.cross(z, x)
        R = np.stack([x, y, z])           # rows: cam axes in world coords
        t = -R @ eye
        fx = (width / 2.0) / np.tan(np.deg2rad(fov_x_deg) / 2.0)
        return cls(fx, fx, width / 2.0, height / 2.0, R, t, width, height)

    def project(self, pts):
        """World points (..., 3) -> pixel coords (..., 2) and depth (...)."""
        p = np.asarray(pts, dtype=np.float64)
        cam = p @ self.R.T + self.t
        depth = cam[..., 2]
        uv = np.empty(p.shape[:-1] + (2,))
        uv[..., 0] = self.fx * cam[..., 0] / depth + self.cx
        uv[..., 1] = self.fy * cam[..., 1] / depth + self.cy
        return uv, depth

    def params(self):
        """Dict of everything needed to reconstruct the camera (for the GT npz)."""
        return {
            "cam_fx": self.fx, "cam_fy": self.fy,
            "cam_cx": self.cx, "cam_cy": self.cy,
            "cam_R": self.R, "cam_t": self.t,
            "cam_width": self.width, "cam_height": self.height,
        }

    @classmethod
    def from_params(cls, d):
        return cls(float(d["cam_fx"]), float(d["cam_fy"]), float(d["cam_cx"]),
                   float(d["cam_cy"]), d["cam_R"], d["cam_t"],
                   int(d["cam_width"]), int(d["cam_height"]))
