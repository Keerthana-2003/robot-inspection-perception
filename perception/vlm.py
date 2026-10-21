"""Open-vocabulary detection with a vision-language model (OWLv2 / OWL-ViT via Hugging Face).

Lets an operator ask for things the trained detector never saw, e.g.
["corroded pipe flange", "open electrical panel", "oil leak on floor"], without retraining.
"""
from __future__ import annotations

import numpy as np

from .detector import Detections


class OpenVocabDetector:
    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", threshold: float = 0.2,
                 device: str | None = None):
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.threshold = threshold

    def __call__(self, image_rgb: np.ndarray, queries: list[str]) -> Detections:
        from PIL import Image

        h, w = image_rgb.shape[:2]
        inputs = self.processor(text=[queries], images=Image.fromarray(image_rgb), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        # OWLv2 pads images to a square; post-process against the padded size, then clip.
        side = max(h, w)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=self.torch.tensor([[side, side]]))[0]
        boxes = res["boxes"].cpu().numpy()
        boxes = np.clip(boxes, 0, [w, h, w, h]) if len(boxes) else np.zeros((0, 4))
        return Detections(boxes, res["scores"].cpu().numpy(), res["labels"].cpu().numpy().astype(int),
                          None, dict(enumerate(queries)))
