"""Fit PatchCore on MVTec AD (industrial inspection benchmark) and report image AUROC and
pixel AUROC per category.

Dataset layout (download from https://www.mvtec.com/company/research/datasets/mvtec-ad or Kaggle):
  mvtec_ad/<category>/train/good/*.png
  mvtec_ad/<category>/test/<defect_type or good>/*.png
  mvtec_ad/<category>/ground_truth/<defect_type>/*_mask.png

  python scripts/train_anomaly.py --root mvtec_ad --categories bottle screw metal_nut
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load(paths, size):
    import cv2

    return np.stack([cv2.resize(cv2.imread(str(p))[..., ::-1], (size, size)) for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="mvtec_ad")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw", "metal_nut"])
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    import cv2

    from perception.anomaly import PatchCore
    from perception.metrics import auroc

    results = {}
    for cat in args.categories:
        root = Path(args.root) / cat
        train = sorted((root / "train" / "good").glob("*.png"))
        tests = sorted((root / "test").glob("*/*.png"))
        t0 = time.time()
        model = PatchCore(coreset_ratio=args.coreset, image_size=args.size).fit(load(train, args.size))
        fit_s = time.time() - t0

        img_scores, img_labels, pix_scores, pix_labels, latencies = [], [], [], [], []
        for p in tests:
            img = load([p], args.size)[0]
            t1 = time.perf_counter()
            r = model.predict(img)
            latencies.append((time.perf_counter() - t1) * 1000)
            defect = p.parent.name != "good"
            img_scores.append(r.score)
            img_labels.append(int(defect))
            amap = cv2.resize(r.patch_map.astype(np.float32), (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            if defect:
                gt = cv2.imread(str(root / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"), 0)
                gt = cv2.resize(gt, (args.size, args.size), interpolation=cv2.INTER_NEAREST) > 0
            else:
                gt = np.zeros((args.size, args.size), bool)
            pix_scores.append(amap[::4, ::4].ravel())  # subsample pixels to keep memory small
            pix_labels.append(gt[::4, ::4].ravel())
        results[cat] = {
            "train_images": len(train), "test_images": len(tests),
            "image_AUROC": round(auroc(img_scores, img_labels), 4),
            "pixel_AUROC": round(auroc(np.concatenate(pix_scores), np.concatenate(pix_labels).astype(int)), 4),
            "fit_seconds": round(fit_s, 1), "p50_latency_ms": round(float(np.median(latencies)), 1),
            "memory_bank_patches": int(len(model.memory.bank)),
        }
        print(cat, results[cat])
    results["mean_image_AUROC"] = round(float(np.mean([v["image_AUROC"] for v in results.values()])), 4)
    Path("results").mkdir(exist_ok=True)
    Path("results/anomaly_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
