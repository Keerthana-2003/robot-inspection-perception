"""Multi-object tracking: ByteTrack-style association with a constant-velocity Kalman filter.

ByteTrack's key idea: associate high-confidence detections first, then try to rescue unmatched
tracks with low-confidence detections (often occluded or blurred objects) before giving up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix


def _to_cxcyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def _to_xyxy(s):
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBox:
    """Constant-velocity Kalman filter over (cx, cy, aspect, h) and their velocities."""

    def __init__(self, box):
        self.x = np.r_[_to_cxcyah(box), np.zeros(4)]
        h = self.x[3]
        self.P = np.diag([2 * h, 2 * h, 1e-2, 2 * h, 10 * h, 10 * h, 1e-5, 10 * h]) ** 2 / 100
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def predict(self):
        h = self.x[3]
        q = np.r_[[h / 20, h / 20, 1e-2, h / 20], [h / 160, h / 160, 1e-5, h / 160]] ** 2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag(q)
        return _to_xyxy(self.x)

    def update(self, box):
        h = self.x[3]
        R = np.diag([h / 20, h / 20, 1e-1, h / 20]) ** 2
        y = _to_cxcyah(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(8) - Kg @ self.H) @ self.P

    @property
    def box(self):
        return _to_xyxy(self.x)


@dataclass
class Track:
    track_id: int
    label: int
    score: float
    kf: KalmanBox
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    history: list = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box


class ByteTracker:
    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, match_iou: float = 0.3,
                 max_lost: int = 30, min_hits: int = 2):
        self.high, self.low, self.match_iou = high_thresh, low_thresh, match_iou
        self.max_lost, self.min_hits = max_lost, min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0

    def _match(self, tracks, boxes, thresh):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        cost = 1 - iou_matrix(np.array([t.box for t in tracks]), boxes)
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= 1 - thresh]
        mr, mc = {r for r, _ in matches}, {c for _, c in matches}
        return matches, [i for i in range(len(tracks)) if i not in mr], [j for j in range(len(boxes)) if j not in mc]

    def update(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray | None = None) -> list[Track]:
        """Feed one frame of detections; returns confirmed tracks updated this frame."""
        self.frame += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        labels = np.zeros(len(boxes), int) if labels is None else np.asarray(labels).reshape(-1)
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        hi = scores >= self.high
        lo = (scores >= self.low) & ~hi
        hi_idx, lo_idx = np.where(hi)[0], np.where(lo)[0]

        # 1) all tracks vs high-confidence detections
        m1, unmatched_t, unmatched_hi = self._match(self.tracks, boxes[hi_idx], self.match_iou)
        for ti, di in m1:
            self._apply(self.tracks[ti], boxes[hi_idx[di]], scores[hi_idx[di]])
        # 2) remaining tracks vs low-confidence detections (recovers occluded objects)
        remaining = [self.tracks[i] for i in unmatched_t]
        m2, _, _ = self._match(remaining, boxes[lo_idx], 0.5)
        for ti, di in m2:
            self._apply(remaining[ti], boxes[lo_idx[di]], scores[lo_idx[di]])
        # 3) unmatched high-confidence detections start new tracks
        for di in unmatched_hi:
            j = hi_idx[di]
            self.tracks.append(Track(self._next_id, int(labels[j]), float(scores[j]), KalmanBox(boxes[j])))
            self.tracks[-1].history.append((self.frame, boxes[j].copy()))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]
        return [t for t in self.tracks if t.time_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)]

    def _apply(self, track: Track, box, score):
        track.kf.update(box)
        track.hits += 1
        track.time_since_update = 0
        track.score = float(score)
        track.history.append((self.frame, np.asarray(box).copy()))
