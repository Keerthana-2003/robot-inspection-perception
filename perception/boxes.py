"""Box utilities: IoU, NMS and letterbox coordinate mapping (numpy, no framework dependency)."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in x1y1x2y2."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5,
        classes: np.ndarray | None = None) -> np.ndarray:
    """Greedy non-maximum suppression; class-aware when `classes` is given. Returns kept indices."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    if classes is not None:  # offset boxes per class so different classes never overlap
        boxes = boxes + (np.asarray(classes).reshape(-1, 1) * (boxes.max() + 1))
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.asarray(keep, dtype=int)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, np.float64)
    return np.concatenate([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], axis=-1)


def letterbox_params(src_hw: tuple[int, int], dst: int) -> tuple[float, float, float]:
    """Scale and padding used to fit an image into a dst x dst square keeping aspect ratio."""
    h, w = src_hw
    r = min(dst / h, dst / w)
    pad_x = (dst - round(w * r)) / 2
    pad_y = (dst - round(h * r)) / 2
    return r, pad_x, pad_y


def unletterbox(boxes: np.ndarray, src_hw: tuple[int, int], dst: int) -> np.ndarray:
    """Map boxes from letterboxed model coordinates back to the original image."""
    r, px, py = letterbox_params(src_hw, dst)
    out = (np.asarray(boxes, np.float64) - [px, py, px, py]) / r
    h, w = src_hw
    return np.clip(out, 0, [w, h, w, h])
