"""Visual anomaly detection for inspection: a PatchCore implementation.

PatchCore (Roth et al., CVPR 2022) learns only from normal images:
  1. Extract mid-level patch features from a pretrained CNN (WideResNet-50 layer2 + layer3),
     locally averaged so each patch sees its neighbourhood.
  2. Keep a memory bank of normal patches, reduced by greedy coreset subsampling.
  3. Score a new image's patches by distance to the nearest memory patch; the image score is the
     maximum patch score and the patch scores form a pixel-level anomaly map.

The memory bank / scoring core is pure numpy (tested in CI); feature extraction uses PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def greedy_coreset(features: np.ndarray, ratio: float = 0.1, projection_dim: int = 128,
                   seed: int = 0) -> np.ndarray:
    """Indices of a coreset that approximately covers the feature set (k-center greedy),
    computed in a random low-dimensional projection for speed."""
    n = len(features)
    k = max(1, int(round(n * ratio)))
    rng = np.random.default_rng(seed)
    x = features
    if features.shape[1] > projection_dim:
        proj = rng.normal(size=(features.shape[1], projection_dim)) / np.sqrt(projection_dim)
        x = features @ proj
    selected = [int(rng.integers(n))]
    min_d = np.linalg.norm(x - x[selected[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(min_d))
        selected.append(i)
        min_d = np.minimum(min_d, np.linalg.norm(x - x[i], axis=1))
    return np.asarray(selected)


def nearest_distances(queries: np.ndarray, bank: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Euclidean distance from each query to its nearest neighbour in the bank."""
    out = np.empty(len(queries))
    bank_sq = (bank ** 2).sum(1)
    for s in range(0, len(queries), batch):
        q = queries[s:s + batch]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ bank.T + bank_sq[None, :]
        out[s:s + batch] = np.sqrt(np.clip(d2.min(1), 0, None))
    return out


@dataclass
class AnomalyResult:
    score: float               # image-level anomaly score
    patch_map: np.ndarray      # (h, w) patch-level scores
    is_anomalous: bool


class PatchCoreMemory:
    """Memory bank and scoring, independent of the feature extractor."""

    def __init__(self, coreset_ratio: float = 0.1):
        self.coreset_ratio = coreset_ratio
        self.bank: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, patch_features: np.ndarray) -> "PatchCoreMemory":
        """patch_features: (num_patches_total, D) from normal images only."""
        idx = greedy_coreset(patch_features, self.coreset_ratio)
        self.bank = patch_features[idx].astype(np.float32)
        return self

    def calibrate_threshold(self, normal_image_patches: list[np.ndarray], quantile: float = 0.99) -> float:
        """Set the decision threshold from held-out normal images (false-positive budget)."""
        scores = [self.score_patches(p, (1, len(p))).score for p in normal_image_patches]
        self.threshold = float(np.quantile(scores, quantile))
        return self.threshold

    def score_patches(self, patches: np.ndarray, grid_hw: tuple[int, int]) -> AnomalyResult:
        if self.bank is None:
            raise RuntimeError("fit() the memory bank first")
        d = nearest_distances(patches.astype(np.float32), self.bank)
        score = float(d.max())
        thr = self.threshold if self.threshold is not None else np.inf
        return AnomalyResult(score, d.reshape(grid_hw), score > thr)


class PatchCore:
    """End-to-end PatchCore with a torchvision WideResNet-50 backbone (requires torch)."""

    def __init__(self, coreset_ratio: float = 0.1, image_size: int = 224, device: str | None = None):
        import torch
        import torchvision

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        net = torchvision.models.wide_resnet50_2(weights=weights).eval().to(self.device)
        self.net = net
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.memory = PatchCoreMemory(coreset_ratio)

    def _features(self, images: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """images: (B, H, W, 3) uint8 RGB -> (B, h*w, D) patch features."""
        torch, F = self.torch, self.torch.nn.functional
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device) / 255
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        n = self.net
        with torch.no_grad():
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l2 = n.layer2(n.layer1(x))
            l3 = n.layer3(l2)
            l2 = F.avg_pool2d(l2, 3, 1, 1)
            l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([l2, l3], 1)
        b, d, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b, h * w, d).cpu().numpy(), (h, w)

    def fit(self, normal_images: np.ndarray, batch: int = 16) -> "PatchCore":
        feats = [self._features(normal_images[i:i + batch])[0] for i in range(0, len(normal_images), batch)]
        self.memory.fit(np.concatenate(feats).reshape(-1, feats[0].shape[-1]))
        return self

    def predict(self, image: np.ndarray) -> AnomalyResult:
        f, hw = self._features(image[None])
        return self.memory.score_patches(f[0], hw)
