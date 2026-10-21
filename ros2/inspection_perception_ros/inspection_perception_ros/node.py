"""ROS 2 node for the inspection pipeline.

Subscribes
  image    sensor_msgs/Image        RGB camera (rgb8 / bgr8)
  points   sensor_msgs/PointCloud2  LiDAR (x, y, z float32), latest cloud is fused with each image
  thermal  sensor_msgs/Image        radiometric temperature image (32FC1, degrees C)
Publishes
  inspection/results  std_msgs/String        JSON per frame: tracks, 3D objects, events, hotspots, latency
  ptz/command         geometry_msgs/Vector3  x = pan deg, y = tilt deg, z = zoom (when a PTZ target exists)

Message conversion is done with numpy (no cv_bridge dependency) so the node runs in a slim container.
"""
from __future__ import annotations

import json

import numpy as np

try:  # rclpy is only present inside a ROS 2 environment
    import rclpy
    from geometry_msgs.msg import Vector3
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - lets the conversion helpers be unit-tested without ROS
    rclpy = None
    Node = object


def image_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image -> numpy (RGB uint8 or float32 single channel)."""
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        return arr[..., ::-1].copy() if msg.encoding == "bgr8" else arr.copy()
    if msg.encoding == "32FC1":
        arr = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.step // 4)
        return arr[:, :msg.width].copy()
    raise ValueError(f"unsupported encoding {msg.encoding}")


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 with float32 x, y, z fields -> (N, 3), NaNs removed."""
    offsets = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    xyz = np.stack([raw[:, offsets[k]:offsets[k] + 4].copy().view(np.float32).ravel() for k in "xyz"], axis=1)
    return xyz[np.isfinite(xyz).all(1)]


def load_calibration(path: str):
    """YAML with camera intrinsics (fx, fy, cx, cy, width, height) and a 4x4 T_cam_lidar."""
    import yaml

    from perception.geometry import CameraIntrinsics

    with open(path) as f:
        c = yaml.safe_load(f)
    cam = CameraIntrinsics(**{k: c["camera"][k] for k in ("fx", "fy", "cx", "cy", "width", "height")})
    T = np.asarray(c["T_cam_lidar"], float) if "T_cam_lidar" in c else None
    return cam, T


class InspectionNode(Node):
    def __init__(self):
        super().__init__("inspection_perception")
        from perception.detector import ONNXDetector
        from perception.pipeline import InspectionPipeline
        from perception.ptz import PTZController

        model = self.declare_parameter("model", "/models/best.onnx").value
        num_classes = int(self.declare_parameter("num_classes", 1).value)
        calib = self.declare_parameter("calibration", "/config/calibration.yaml").value
        use_ptz = bool(self.declare_parameter("ptz", False).value)
        cam, T = load_calibration(calib)
        self.pipeline = InspectionPipeline(ONNXDetector(model, num_classes=num_classes), cam, T_cam_lidar=T,
                                           ptz=PTZController(cam) if use_ptz else None)
        self.latest_points = None
        self.latest_thermal = None
        self.zoom = 1.0
        self.pub = self.create_publisher(String, "inspection/results", 10)
        self.ptz_pub = self.create_publisher(Vector3, "ptz/command", 10)
        self.create_subscription(Image, "image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "points", self.on_points, qos_profile_sensor_data)
        self.create_subscription(Image, "thermal", self.on_thermal, qos_profile_sensor_data)
        self.get_logger().info(f"inspection node ready (model={model})")

    def on_points(self, msg):
        self.latest_points = pointcloud2_to_xyz(msg)

    def on_thermal(self, msg):
        self.latest_thermal = image_to_numpy(msg)

    def on_image(self, msg):
        try:
            res = self.pipeline.process(image_to_numpy(msg), lidar_points=self.latest_points,
                                        thermal_c=self.latest_thermal, current_zoom=self.zoom)
        except Exception as err:  # keep the node alive; field debugging happens from these logs
            self.get_logger().error(f"pipeline failed: {err}")
            return
        out = res.to_dict()
        out["stamp"] = {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec}
        self.pub.publish(String(data=json.dumps(out, default=float)))
        if res.ptz is not None and not res.ptz.centered:
            self.zoom = res.ptz.zoom
            self.ptz_pub.publish(Vector3(x=res.ptz.pan_deg, y=res.ptz.tilt_deg, z=res.ptz.zoom))


def main():
    rclpy.init()
    node = InspectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
