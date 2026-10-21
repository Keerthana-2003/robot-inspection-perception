"""Active perception with a PTZ (pan-tilt-zoom) camera: centre and zoom on a target so the
inspection model sees it at the resolution it needs.

The controller converts the target's pixel offset into pan/tilt angles through the camera
intrinsics (not a fixed gain), applies a deadband and per-step limits, and chooses zoom so the
target fills a desired fraction of the frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CameraIntrinsics


@dataclass
class PTZCommand:
    pan_deg: float     # positive = right
    tilt_deg: float    # positive = up
    zoom: float        # absolute zoom factor
    centered: bool     # target within deadband and at the desired size


@dataclass
class PTZController:
    cam: CameraIntrinsics            # intrinsics at zoom 1.0
    deadband_px: float = 10.0
    max_step_deg: float = 15.0
    target_fill: float = 0.4         # desired target height / image height
    zoom_limits: tuple[float, float] = (1.0, 30.0)
    gain: float = 0.8                # < 1 damps overshoot from detector noise

    def command(self, box: np.ndarray, current_zoom: float) -> PTZCommand:
        x1, y1, x2, y2 = box
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        fx, fy = self.cam.fx * current_zoom, self.cam.fy * current_zoom  # focal length scales with zoom
        du, dv = u - self.cam.cx, v - self.cam.cy
        pan = np.degrees(np.arctan2(du, fx)) if abs(du) > self.deadband_px else 0.0
        tilt = -np.degrees(np.arctan2(dv, fy)) if abs(dv) > self.deadband_px else 0.0
        pan = float(np.clip(self.gain * pan, -self.max_step_deg, self.max_step_deg))
        tilt = float(np.clip(self.gain * tilt, -self.max_step_deg, self.max_step_deg))

        fill = (y2 - y1) / self.cam.height
        zoom = current_zoom
        if pan == 0.0 and tilt == 0.0 and fill > 0:  # zoom only once centred, so the target stays in view
            zoom = float(np.clip(current_zoom * self.target_fill / fill, *self.zoom_limits))
        size_ok = abs(fill - self.target_fill) / self.target_fill < 0.15 or zoom in self.zoom_limits
        return PTZCommand(pan, tilt, zoom, pan == 0.0 and tilt == 0.0 and size_ok)
