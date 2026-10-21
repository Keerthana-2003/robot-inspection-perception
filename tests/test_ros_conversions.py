"""ROS message conversion helpers, tested with lightweight stand-ins for the ROS message types."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "inspection_perception_ros"))
from inspection_perception_ros.node import image_to_numpy, pointcloud2_to_xyz  # noqa: E402


def test_bgr8_image_with_row_padding():
    rgb = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    bgr = rgb[..., ::-1]
    padded = np.zeros((4, 5 * 3 + 1), np.uint8)  # step larger than width * 3
    padded[:, :15] = bgr.reshape(4, 15)
    msg = SimpleNamespace(encoding="bgr8", height=4, width=5, step=16, data=padded.tobytes())
    np.testing.assert_array_equal(image_to_numpy(msg), rgb)


def test_thermal_32fc1_image():
    t = np.arange(12, dtype=np.float32).reshape(3, 4) + 20
    msg = SimpleNamespace(encoding="32FC1", height=3, width=4, step=16, data=t.tobytes())
    np.testing.assert_array_equal(image_to_numpy(msg), t)


def test_pointcloud2_xyz_with_intensity_and_nans():
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    pts = np.array([(1, 2, 3, 0.5), (np.nan, 0, 0, 0.1), (4, 5, 6, 0.9)], dtype=dtype)
    fields = [SimpleNamespace(name=n, offset=o) for n, o in [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)]]
    msg = SimpleNamespace(fields=fields, point_step=16, data=pts.tobytes())
    np.testing.assert_array_equal(pointcloud2_to_xyz(msg), [[1, 2, 3], [4, 5, 6]])
