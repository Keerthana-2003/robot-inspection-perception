"""Detectors behind one interface.

* YOLODetector  - Ultralytics YOLO (PyTorch) for training-time use and quick experiments.
* ONNXDetector  - deployment path: ONNX Runtime (CPU / CUDA / TensorRT execution providers) with
                  our own letterbox preprocessing, YOLO output decoding, NMS and instance masks.
                  Same code runs on a workstation and on an NVIDIA Jetson.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boxes import nms, unletterbox, xywh_to_xyxy


@dataclass
class Detections:
    boxes: np.ndarray                 # (N, 4) x1y1x2y2 in original image pixels
    scores: np.ndarray                # (N,)
    labels: np.ndarray                # (N,)
    masks: np.ndarray | None = None   # (N, H, W) bool instance masks (segmentation models)
    names: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls, names=None) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None, names or {})


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad_value, dtype=image.dtype)
    top, left = int((size - nh) / 2), int((size - nw) / 2)
    out[top:top + nh, left:left + nw] = resized
    return out


def decode_yolo(output: np.ndarray, num_classes: int, conf: float, iou: float,
                max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode a YOLOv8/YOLO11 head output of shape (1, 4 + nc [+ nm], A) in model-input pixels.
    Returns boxes (xyxy), scores, labels and mask coefficients (or None)."""
    pred = output[0].T  # (A, 4 + nc + nm)
    cls_scores = pred[:, 4:4 + num_classes]
    labels = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), labels]
    keep = scores >= conf
    pred, labels, scores = pred[keep], labels[keep], scores[keep]
    boxes = xywh_to_xyxy(pred[:, :4])
    k = nms(boxes, scores, iou, classes=labels)[:max_det]
    coeffs = pred[k, 4 + num_classes:] if pred.shape[1] > 4 + num_classes else None
    return boxes[k], scores[k], labels[k], coeffs


def masks_from_protos(coeffs: np.ndarray, protos: np.ndarray, boxes_in: np.ndarray, input_size: int,
                      src_hw: tuple[int, int]) -> np.ndarray:
    """Instance masks from YOLO-seg prototypes: sigmoid(coeffs @ protos), cropped to each box,
    resized back to the original image."""
    import cv2

    nm, mh, mw = protos.shape
    m = 1 / (1 + np.exp(-(coeffs @ protos.reshape(nm, -1)))).reshape(-1, mh, mw)
    out = np.zeros((len(m), *src_hw), dtype=bool)
    from .boxes import letterbox_params

    r, px, py = letterbox_params(src_hw, input_size)
    sx, sy = mw / input_size, mh / input_size
    for i, (mask, b) in enumerate(zip(m, boxes_in)):
        crop = np.zeros_like(mask)
        x1, y1, x2, y2 = (b * [sx, sy, sx, sy]).round().astype(int)
        crop[max(y1, 0):y2, max(x1, 0):x2] = mask[max(y1, 0):y2, max(x1, 0):x2]
        full = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        top, left = int(py), int(px)
        nh, nw = round(src_hw[0] * r), round(src_hw[1] * r)
        full = full[top:top + nh, left:left + nw]
        out[i] = cv2.resize(full, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_LINEAR) > 0.5
    return out


class ONNXDetector:
    def __init__(self, model_path: str, num_classes: int, input_size: int = 640, conf: float = 0.25,
                 iou: float = 0.5, providers: list[str] | None = None, names: dict | None = None, session=None):
        if session is None:
            import onnxruntime as ort

            available = ort.get_available_providers()
            wanted = providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(model_path, providers=[p for p in wanted if p in available])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.num_classes, self.input_size, self.conf, self.iou = num_classes, input_size, conf, iou
        self.names = names or {}

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        x = letterbox(image_rgb, self.input_size).astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def __call__(self, image_rgb: np.ndarray) -> Detections:
        outputs = self.session.run(None, {self.input_name: self.preprocess(image_rgb)})
        boxes_in, scores, labels, coeffs = decode_yolo(outputs[0], self.num_classes, self.conf, self.iou)
        if len(boxes_in) == 0:
            return Detections.empty(self.names)
        src_hw = image_rgb.shape[:2]
        masks = None
        if coeffs is not None and len(outputs) > 1:
            masks = masks_from_protos(coeffs, outputs[1][0], boxes_in, self.input_size, src_hw)
        return Detections(unletterbox(boxes_in, src_hw, self.input_size), scores, labels, masks, self.names)


class YOLODetector:
    def __init__(self, weights: str = "yolo11n-seg.pt", conf: float = 0.25, iou: float = 0.5, device=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.iou, self.device = conf, iou, device

    def __call__(self, image_rgb: np.ndarray) -> Detections:
        r = self.model.predict(image_rgb[..., ::-1], conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty(r.names)
        masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
        if masks is not None and masks.shape[1:] != image_rgb.shape[:2]:
            import cv2

            masks = np.stack([cv2.resize(m.astype(np.uint8), image_rgb.shape[1::-1]) > 0 for m in masks])
        return Detections(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                          r.boxes.cls.cpu().numpy().astype(int), masks, r.names)
