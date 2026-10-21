"""Sensor calibration.

* Camera intrinsics from checkerboard images (OpenCV), with reprojection error.
* LiDAR-to-camera extrinsics from 3D-2D correspondences (e.g. board corners measured in both
  sensors): linear DLT initialisation followed by Gauss-Newton refinement of reprojection error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CameraIntrinsics, make_transform


@dataclass
class IntrinsicCalibration:
    camera: CameraIntrinsics
    rms_reprojection_error_px: float
    images_used: int


def calibrate_camera_checkerboard(images: list[np.ndarray], pattern: tuple[int, int],
                                  square_size_m: float) -> IntrinsicCalibration:
    """Estimate intrinsics and distortion from grayscale checkerboard images.

    pattern: inner corners per row and column, e.g. (9, 6).
    """
    import cv2

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size_m
    obj_points, img_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for img in images:
        found, corners = cv2.findChessboardCorners(img, pattern)
        if not found:
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
    if len(obj_points) < 3:
        raise ValueError(f"Checkerboard found in only {len(obj_points)} images; need at least 3.")
    h, w = images[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    return IntrinsicCalibration(CameraIntrinsics.from_matrix(K, w, h, dist.ravel()), float(rms), len(obj_points))


def _rodrigues(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * Kx @ Kx


def _project(R, t, pts, K):
    pc = pts @ R.T + t
    return (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def calibrate_lidar_camera(points_lidar: np.ndarray, pixels: np.ndarray, cam: CameraIntrinsics,
                           iterations: int = 20) -> tuple[np.ndarray, float]:
    """Estimate T_cam_lidar from >= 6 non-coplanar 3D-2D correspondences.

    Returns (4x4 transform, RMS reprojection error in pixels).
    """
    X = np.asarray(points_lidar, np.float64)
    uv = np.asarray(pixels, np.float64)
    if len(X) < 6:
        raise ValueError("Need at least 6 correspondences")
    K = cam.K
    # Normalised image coordinates, then DLT for [R|t] up to scale.
    xn = (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T).T
    A = []
    for (Xw, Yw, Zw), (u, v, _) in zip(X, xn):
        A.append([Xw, Yw, Zw, 1, 0, 0, 0, 0, -u * Xw, -u * Yw, -u * Zw, -u])
        A.append([0, 0, 0, 0, Xw, Yw, Zw, 1, -v * Xw, -v * Yw, -v * Zw, -v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    P = Vt[-1].reshape(3, 4)
    U, S, Vt3 = np.linalg.svd(P[:, :3])
    R = U @ Vt3
    scale = S.mean()
    t = P[:, 3] / scale
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    if np.mean(X @ R[2] + t[2]) < 0:  # points must be in front of the camera
        R, t = -R, -t
    # Gauss-Newton on reprojection error (rotation as a small-angle update).
    for _ in range(iterations):
        pred = _project(R, t, X, K)
        r = (pred - uv).ravel()
        J = np.zeros((len(r), 6))
        eps = 1e-6
        for j in range(6):
            d = np.zeros(6)
            d[j] = eps
            Rj = _rodrigues(d[:3]) @ R
            J[:, j] = ((_project(Rj, t + d[3:], X, K) - uv).ravel() - r) / eps
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        R = _rodrigues(step[:3]) @ R
        t = t + step[3:]
        if np.linalg.norm(step) < 1e-10:
            break
    rms = float(np.sqrt(np.mean(np.sum((_project(R, t, X, K) - uv) ** 2, axis=1))))
    return make_transform(R, t), rms
