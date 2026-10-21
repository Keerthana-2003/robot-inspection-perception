"""Evaluation metrics: detection AP / mAP (COCO-style), segmentation IoU, tracking MOTA / ID switches,
and anomaly-detection AUROC."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import iou_matrix


def average_precision(pred_boxes, pred_scores, pred_image_ids, gt_boxes, gt_image_ids,
                      iou_threshold: float = 0.5) -> float:
    """AP for one class with COCO 101-point interpolation.

    pred_* / gt_* are flat arrays across the dataset; *_image_ids say which image each box is in.
    """
    pred_boxes = np.asarray(pred_boxes, np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(pred_scores))
    gt_ids = np.asarray(gt_image_ids)
    used = np.zeros(n_gt, bool)
    tp = np.zeros(len(order))
    for k, i in enumerate(order):
        img = pred_image_ids[i]
        cand = np.where(gt_ids == img)[0]
        if len(cand) == 0:
            continue
        ious = iou_matrix(pred_boxes[i:i + 1], gt_boxes[cand])[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_threshold and not used[cand[j]]:
            used[cand[j]] = True
            tp[k] = 1
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    # precision envelope, then sample at 101 recall points
    for k in range(len(precision) - 2, -1, -1):
        precision[k] = max(precision[k], precision[k + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


def mean_average_precision(preds: list[dict], gts: list[dict], iou_thresholds=(0.5,)) -> dict:
    """preds: [{"image_id", "box", "score", "label"}], gts: [{"image_id", "box", "label"}].
    Returns {"mAP": ..., "per_class": {label: AP}} averaged over classes and IoU thresholds."""
    labels = sorted({g["label"] for g in gts})
    per_class = {}
    for c in labels:
        p = [x for x in preds if x["label"] == c]
        g = [x for x in gts if x["label"] == c]
        aps = [average_precision([x["box"] for x in p], [x["score"] for x in p], [x["image_id"] for x in p],
                                 [x["box"] for x in g], [x["image_id"] for x in g], t) for t in iou_thresholds]
        per_class[c] = float(np.mean(aps))
    return {"mAP": float(np.mean(list(per_class.values()))) if per_class else float("nan"), "per_class": per_class}


COCO_IOUS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


def segmentation_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> dict:
    """Per-class IoU and mIoU for semantic segmentation label maps."""
    ious = {}
    for c in range(num_classes):
        p, g = pred_mask == c, gt_mask == c
        union = np.logical_or(p, g).sum()
        if union:
            ious[c] = float(np.logical_and(p, g).sum() / union)
    return {"per_class": ious, "mIoU": float(np.mean(list(ious.values()))) if ious else float("nan")}


def mota(gt_frames: list[dict], pred_frames: list[dict], iou_threshold: float = 0.5) -> dict:
    """CLEAR-MOT accuracy. Each frame dict maps object/track id -> box."""
    fn = fp = idsw = n_gt = 0
    last_match: dict = {}
    for gt, pr in zip(gt_frames, pred_frames):
        n_gt += len(gt)
        gids, pids = list(gt), list(pr)
        if gids and pids:
            cost = 1 - iou_matrix(np.array([gt[g] for g in gids]), np.array([pr[p] for p in pids]))
            r, c = linear_sum_assignment(cost)
            pairs = [(gids[i], pids[j]) for i, j in zip(r, c) if cost[i, j] <= 1 - iou_threshold]
        else:
            pairs = []
        for g, p in pairs:
            if g in last_match and last_match[g] != p:
                idsw += 1
            last_match[g] = p
        fn += len(gids) - len(pairs)
        fp += len(pids) - len(pairs)
    return {"MOTA": 1 - (fn + fp + idsw) / max(n_gt, 1), "FN": fn, "FP": fp, "IDSW": idsw}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U), labels 1 = anomalous."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.r_[pos, neg])) + 1.0
    # average ranks for ties
    allv = np.r_[pos, neg]
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
