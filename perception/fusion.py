"""Camera-LiDAR / camera-depth fusion: lift 2D detections into 3D and measure them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CameraIntrinsics, backproject, box_metric_size, project_lidar


@dataclass
class Detection3D:
    box: np.ndarray            # x1, y1, x2, y2 (pixels)
    score: float
    label: int
    depth_m: float | None      # robust distance along the camera z axis
    center_cam: np.ndarray | None  # 3D centre in the camera frame (m)
    size_m: tuple[float, float] | None  # approximate width, height (m)
    support_points: int        # LiDAR points / depth pixels used


def _shrink(box: np.ndarray, frac: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac / 2, (y2 - y1) * frac / 2
    return np.array([x1 + dx, y1 + dy, x2 - dx, y2 - dy])


def _robust_depth(depths: np.ndarray) -> float:
    """Depth of the nearest dense cluster: ignores background points that leak into the box."""
    d = np.sort(depths)
    if len(d) < 5:
        return float(np.median(d))
    q1 = np.percentile(d, 25)
    near = d[d <= q1 + max(0.5, 0.15 * q1)]
    return float(np.median(near))


def fuse_lidar(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, points_lidar: np.ndarray,
               T_cam_lidar: np.ndarray, cam: CameraIntrinsics, shrink: float = 0.2,
               min_points: int = 3) -> list[Detection3D]:
    """Assign projected LiDAR points to each 2D box (shrunk to avoid edge bleed) and estimate
    depth, 3D centre and metric size."""
    uv, depth, valid, pts_cam = project_lidar(points_lidar, T_cam_lidar, cam)
    out = []
    for box, score, label in zip(np.asarray(boxes).reshape(-1, 4), scores, labels):
        x1, y1, x2, y2 = _shrink(box, shrink)
        m = valid & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        if m.sum() < min_points:
            out.append(Detection3D(box, float(score), int(label), None, None, None, int(m.sum())))
            continue
        d = _robust_depth(depth[m])
        cluster = m & (np.abs(depth - d) < max(0.5, 0.15 * d))
        center = np.median(pts_cam[cluster], axis=0)
        out.append(Detection3D(box, float(score), int(label), d, center, box_metric_size(box, d, cam),
                               int(cluster.sum())))
    return out


def fuse_depth_image(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, depth_img: np.ndarray,
                     cam: CameraIntrinsics, shrink: float = 0.2, min_valid: int = 10) -> list[Detection3D]:
    """Same as fuse_lidar but from an aligned depth camera image (metres, 0 = invalid)."""
    out = []
    h, w = depth_img.shape
    for box, score, label in zip(np.asarray(boxes).reshape(-1, 4), scores, labels):
        x1, y1, x2, y2 = np.clip(_shrink(box, shrink), 0, [w - 1, h - 1, w - 1, h - 1]).astype(int)
        patch = depth_img[y1:y2 + 1, x1:x2 + 1]
        vals = patch[patch > 0]
        if len(vals) < min_valid:
            out.append(Detection3D(box, float(score), int(label), None, None, None, int(len(vals))))
            continue
        d = _robust_depth(vals)
        c = backproject([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], [d], cam)[0]
        out.append(Detection3D(box, float(score), int(label), d, c, box_metric_size(box, d, cam), int(len(vals))))
    return out
