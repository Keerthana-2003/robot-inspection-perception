"""The inspection pipeline that runs on the robot, one frame at a time:

  RGB frame -> detector (YOLO / ONNX / TensorRT) -> ByteTrack -> LiDAR or depth fusion
            -> metric measurement -> zone events -> optional thermal hotspots and PTZ command

Each stage is optional so the same pipeline serves a fixed camera, a PTZ inspection head, or a
mobile robot with LiDAR. Per-stage latency is recorded for profiling on edge hardware.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detector import Detections
from .events import Event, ZoneMonitor
from .fusion import Detection3D, fuse_depth_image, fuse_lidar
from .geometry import CameraIntrinsics
from .ptz import PTZCommand, PTZController
from .thermal import Hotspot, find_hotspots
from .tracking import ByteTracker


@dataclass
class FrameResult:
    frame: int
    detections: Detections
    tracks: list[dict]
    objects_3d: list[Detection3D] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    hotspots: list[Hotspot] = field(default_factory=list)
    ptz: PTZCommand | None = None
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "tracks": self.tracks,
            "objects_3d": [{"label": o.label, "score": round(o.score, 3), "depth_m": o.depth_m,
                            "size_m": o.size_m, "center_cam": None if o.center_cam is None else o.center_cam.round(3).tolist()}
                           for o in self.objects_3d],
            "events": [{"type": e.type, "track_id": e.track_id, "zone": e.zone, "time_s": round(e.time_s, 2), **e.details}
                       for e in self.events],
            "hotspots": [{"box": h.box, "max_temp_c": round(h.max_temp_c, 1), "severity": h.severity} for h in self.hotspots],
            "ptz": None if self.ptz is None else vars(self.ptz),
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
        }


class InspectionPipeline:
    def __init__(self, detector, cam: CameraIntrinsics, tracker: ByteTracker | None = None,
                 zones: dict | None = None, fps: float = 15.0, T_cam_lidar: np.ndarray | None = None,
                 ptz: PTZController | None = None, ptz_target_label: int | None = None):
        self.detector, self.cam = detector, cam
        self.tracker = tracker or ByteTracker()
        self.zones = ZoneMonitor(zones, fps) if zones else None
        self.T_cam_lidar = T_cam_lidar
        self.ptz, self.ptz_target_label = ptz, ptz_target_label
        self.frame = 0

    def process(self, image_rgb: np.ndarray, lidar_points: np.ndarray | None = None,
                depth_image: np.ndarray | None = None, thermal_c: np.ndarray | None = None,
                current_zoom: float = 1.0) -> FrameResult:
        self.frame += 1
        lat = {}
        t0 = time.perf_counter()
        det = self.detector(image_rgb)
        lat["detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tracks = self.tracker.update(det.boxes, det.scores, det.labels)
        lat["track"] = (time.perf_counter() - t0) * 1000
        track_dicts = [{"id": t.track_id, "label": t.label, "score": round(t.score, 3),
                        "box": np.round(t.box, 1).tolist()} for t in tracks]
        result = FrameResult(self.frame, det, track_dicts)

        t0 = time.perf_counter()
        if lidar_points is not None and self.T_cam_lidar is not None:
            result.objects_3d = fuse_lidar(det.boxes, det.scores, det.labels, lidar_points, self.T_cam_lidar, self.cam)
        elif depth_image is not None:
            result.objects_3d = fuse_depth_image(det.boxes, det.scores, det.labels, depth_image, self.cam)
        lat["fusion"] = (time.perf_counter() - t0) * 1000

        if self.zones:
            result.events = self.zones.update(self.frame, [(t.track_id, t.box) for t in tracks])
        if thermal_c is not None:
            _, result.hotspots = find_hotspots(thermal_c)
        if self.ptz is not None:
            targets = [t for t in tracks if self.ptz_target_label is None or t.label == self.ptz_target_label]
            if targets:
                best = max(targets, key=lambda t: t.score)
                result.ptz = self.ptz.command(best.box, current_zoom)
        lat["total"] = sum(lat.values())
        result.latency_ms = lat
        return result
