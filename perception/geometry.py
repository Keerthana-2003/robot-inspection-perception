"""Camera geometry: intrinsics, extrinsics, projection, back-projection and metric measurement.

Conventions
  * Camera frame: x right, y down, z forward (OpenCV).
  * LiDAR points are (N, 3) in the LiDAR frame; T_cam_lidar maps them into the camera frame.
  * Boxes are (x1, y1, x2, y2) in pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...] = ()  # OpenCV distortion coefficients (k1, k2, p1, p2, k3)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int, dist=()) -> "CameraIntrinsics":
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), width, height, tuple(dist))

    def scaled(self, sx: float, sy: float | None = None) -> "CameraIntrinsics":
        """Intrinsics after resizing the image by (sx, sy), e.g. for model input resolution."""
        sy = sx if sy is None else sy
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                                round(self.width * sx), round(self.height * sy), self.dist)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def rotation_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from roll (x), pitch (y), yaw (z) in radians, applied as Rz @ Ry @ Rx."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def project_points(points_cam: np.ndarray, cam: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixels. Returns (uv (N,2), valid mask) where valid means
    in front of the camera and inside the image."""
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]
    in_front = z > 1e-6
    uv = np.full((len(pts), 2), np.nan)
    uv[in_front, 0] = cam.fx * pts[in_front, 0] / z[in_front] + cam.cx
    uv[in_front, 1] = cam.fy * pts[in_front, 1] / z[in_front] + cam.cy
    inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return uv, inside


def project_lidar(points_lidar: np.ndarray, T_cam_lidar: np.ndarray, cam: CameraIntrinsics):
    """LiDAR points -> pixels. Returns (uv, depth, valid) with depth = z in the camera frame."""
    pts_cam = transform_points(T_cam_lidar, points_lidar)
    uv, valid = project_points(pts_cam, cam)
    return uv, pts_cam[:, 2], valid, pts_cam


def backproject(uv: np.ndarray, depth: np.ndarray, cam: CameraIntrinsics) -> np.ndarray:
    """Pixels plus metric depth -> camera-frame 3D points."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    x = (uv[:, 0] - cam.cx) * d / cam.fx
    y = (uv[:, 1] - cam.cy) * d / cam.fy
    return np.stack([x, y, d], axis=1)


def box_metric_size(box: np.ndarray, depth_m: float, cam: CameraIntrinsics) -> tuple[float, float]:
    """Approximate width and height in metres of a fronto-parallel object at `depth_m`."""
    x1, y1, x2, y2 = box
    return float((x2 - x1) * depth_m / cam.fx), float((y2 - y1) * depth_m / cam.fy)


def depth_map_from_lidar(points_lidar: np.ndarray, T_cam_lidar: np.ndarray, cam: CameraIntrinsics) -> np.ndarray:
    """Sparse depth image (metres, 0 = no return) keeping the nearest point per pixel."""
    uv, depth, valid, _ = project_lidar(points_lidar, T_cam_lidar, cam)
    depth_img = np.zeros((cam.height, cam.width), dtype=np.float32)
    if not valid.any():
        return depth_img
    u = uv[valid, 0].astype(int)
    v = uv[valid, 1].astype(int)
    d = depth[valid]
    order = np.argsort(-d)  # write far points first so near points overwrite them
    depth_img[v[order], u[order]] = d[order]
    return depth_img
