"""Video understanding over tracks: zone entry/exit, dwell time, and objects that stop moving.

Turns frame-level tracks into events an operator or robot behaviour can act on
(e.g. "person in restricted zone for 12 s", "pallet left in aisle").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def point_in_polygon(pt: tuple[float, float], poly: np.ndarray) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass
class Event:
    type: str        # zone_enter | zone_exit | dwell | stationary
    track_id: int
    zone: str | None
    frame: int
    time_s: float
    details: dict = field(default_factory=dict)


class ZoneMonitor:
    def __init__(self, zones: dict[str, np.ndarray], fps: float, dwell_s: float = 10.0,
                 stationary_s: float = 5.0, stationary_px: float = 5.0):
        self.zones = {k: np.asarray(v, float) for k, v in zones.items()}
        self.fps, self.dwell_s = fps, dwell_s
        self.stationary_s, self.stationary_px = stationary_s, stationary_px
        self._inside: dict[tuple[int, str], int] = {}     # (track, zone) -> entry frame
        self._dwell_reported: set[tuple[int, str]] = set()
        self._anchor: dict[int, tuple[np.ndarray, int]] = {}  # track -> (position, since frame)
        self._stationary_reported: set[int] = set()

    def update(self, frame: int, tracks: list[tuple[int, np.ndarray]]) -> list[Event]:
        """tracks: [(track_id, box)] for this frame. Uses the bottom-centre of the box (ground contact)."""
        events, seen = [], set()
        t = frame / self.fps
        for tid, box in tracks:
            foot = ((box[0] + box[2]) / 2, box[3])
            for name, poly in self.zones.items():
                key = (tid, name)
                if point_in_polygon(foot, poly):
                    seen.add(key)
                    if key not in self._inside:
                        self._inside[key] = frame
                        events.append(Event("zone_enter", tid, name, frame, t))
                    elif (frame - self._inside[key]) / self.fps >= self.dwell_s and key not in self._dwell_reported:
                        self._dwell_reported.add(key)
                        events.append(Event("dwell", tid, name, frame, t,
                                            {"seconds": (frame - self._inside[key]) / self.fps}))
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            anchor = self._anchor.get(tid)
            if anchor is None or np.linalg.norm(center - anchor[0]) > self.stationary_px:
                self._anchor[tid] = (center, frame)
                self._stationary_reported.discard(tid)
            elif (frame - anchor[1]) / self.fps >= self.stationary_s and tid not in self._stationary_reported:
                self._stationary_reported.add(tid)
                events.append(Event("stationary", tid, None, frame, t, {"seconds": (frame - anchor[1]) / self.fps}))
        for key in [k for k in self._inside if k not in seen]:
            events.append(Event("zone_exit", key[0], key[1], frame, t,
                                {"seconds": (frame - self._inside.pop(key)) / self.fps}))
            self._dwell_reported.discard(key)
        return events
