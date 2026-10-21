from __future__ import annotations

import numpy as np
import pytest

from perception.calibration import calibrate_camera_checkerboard, calibrate_lidar_camera
from perception.geometry import (CameraIntrinsics, backproject, box_metric_size, depth_map_from_lidar, make_transform,
                                 project_lidar, project_points, rotation_from_euler, transform_points)

CAM = CameraIntrinsics(fx=800, fy=800, cx=640, cy=360, width=1280, height=720)
# LiDAR frame (x forward, y left, z up) -> camera frame (x right, y down, z forward), sensor 0.2 m above camera
R_CAM_LIDAR = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
T_CAM_LIDAR = make_transform(R_CAM_LIDAR, [0.0, -0.2, 0.0])


def test_project_backproject_roundtrip():
    pts = np.array([[0.5, -0.2, 4.0], [-1.0, 0.3, 10.0]])
    uv, valid = project_points(pts, CAM)
    assert valid.all()
    np.testing.assert_allclose(backproject(uv, pts[:, 2], CAM), pts, atol=1e-9)


def test_points_behind_camera_or_outside_image_are_invalid():
    _, valid = project_points(np.array([[0, 0, -1.0], [100, 0, 1.0]]), CAM)
    assert not valid.any()


def test_lidar_projection_uses_extrinsics():
    uv, depth, valid, _ = project_lidar(np.array([[5.0, 0.0, 0.0]]), T_CAM_LIDAR, CAM)  # 5 m straight ahead
    assert valid[0] and depth[0] == pytest.approx(5.0)
    assert uv[0, 0] == pytest.approx(640) and uv[0, 1] == pytest.approx(360 - 800 * 0.2 / 5)


def test_depth_map_keeps_nearest_return():
    pts = np.array([[10.0, 0, 0], [4.0, 0, 0.16]])  # both land on the same pixel row/col region
    depth = depth_map_from_lidar(pts, make_transform(R_CAM_LIDAR, [0, 0, 0]), CAM)
    assert depth[360, 640] == pytest.approx(10.0)
    assert depth.max() == pytest.approx(10.0) and depth[depth > 0].min() == pytest.approx(4.0)


def test_metric_size_and_intrinsics_scaling():
    w, h = box_metric_size(np.array([600, 300, 680, 460]), 4.0, CAM)
    assert w == pytest.approx(0.4) and h == pytest.approx(0.8)
    half = CAM.scaled(0.5)
    assert half.fx == 400 and half.width == 640


def test_lidar_camera_extrinsic_calibration_recovers_pose():
    rng = np.random.default_rng(0)
    R_true = rotation_from_euler(0.02, -0.03, 0.01) @ R_CAM_LIDAR
    T_true = make_transform(R_true, [0.05, -0.18, 0.12])
    pts = np.c_[rng.uniform(3, 12, 30), rng.uniform(-3, 3, 30), rng.uniform(-1, 1.5, 30)]
    uv, _ = project_points(transform_points(T_true, pts), CAM)
    uv_noisy = uv + rng.normal(0, 0.5, uv.shape)
    T_est, rms = calibrate_lidar_camera(pts, uv_noisy, CAM)
    assert rms < 1.0
    np.testing.assert_allclose(T_est[:3, 3], T_true[:3, 3], atol=0.03)
    angle = np.degrees(np.arccos(np.clip((np.trace(T_est[:3, :3].T @ R_true) - 1) / 2, -1, 1)))
    assert angle < 0.3


def _render_checkerboard(K, R, t, pattern=(9, 6), square=0.03, size=(640, 480)):
    import cv2

    img = np.full((size[1], size[0]), 255, np.uint8)
    cols, rows = pattern[0] + 1, pattern[1] + 1
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                continue
            corners = np.array([[j, i, 0], [j + 1, i, 0], [j + 1, i + 1, 0], [j, i + 1, 0]], float) * square
            corners -= [square, square, 0]  # inner-corner grid starts at the first inner corner
            pc = corners @ R.T + t
            uv = (pc[:, :2] / pc[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            cv2.fillConvexPoly(img, np.round(uv * 16).astype(np.int32), 0, lineType=cv2.LINE_AA, shift=4)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_checkerboard_intrinsic_calibration():
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    images = []
    for roll, pitch, yaw, tz in [(0.1, -0.2, 0.05, 0.6), (-0.15, 0.2, -0.1, 0.55), (0.2, 0.1, 0.0, 0.65),
                                 (-0.1, -0.25, 0.1, 0.6), (0.05, 0.3, -0.05, 0.7), (0.25, -0.1, 0.15, 0.6)]:
        R = rotation_from_euler(roll, pitch, yaw)
        images.append(_render_checkerboard(K, R, np.array([-0.12, -0.075, tz])))
    calib = calibrate_camera_checkerboard(images, (9, 6), 0.03)
    assert calib.images_used >= 5
    assert calib.rms_reprojection_error_px < 0.5
    assert calib.camera.fx == pytest.approx(600, rel=0.02) and calib.camera.cx == pytest.approx(320, abs=8)
