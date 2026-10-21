"""Train and evaluate a YOLO11 detection + instance-segmentation model for inspection.

Default dataset: Ultralytics crack-seg (concrete/asphalt crack instance segmentation, downloads
automatically). Works with any Ultralytics-format dataset YAML.

  python scripts/train_detector.py --data crack-seg.yaml --model yolo11n-seg.pt --epochs 50
Outputs runs/<name>/weights/best.pt and results/detector_metrics.json (box and mask mAP).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="crack-seg.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="crack_seg")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    t0 = time.time()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name,
                device=args.device, seed=0, deterministic=True, patience=20, plots=True)
    train_min = (time.time() - t0) / 60

    best = YOLO(Path(model.trainer.save_dir) / "weights" / "best.pt")
    val = best.val(data=args.data, imgsz=args.imgsz, split="val", plots=True)
    metrics = {
        "data": args.data, "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
        "train_minutes": round(train_min, 1),
        "box_mAP50": round(float(val.box.map50), 4), "box_mAP50_95": round(float(val.box.map), 4),
        "box_precision": round(float(val.box.mp), 4), "box_recall": round(float(val.box.mr), 4),
        "weights": str(Path(model.trainer.save_dir) / "weights" / "best.pt"),
    }
    if getattr(val, "seg", None) is not None:
        metrics.update({"mask_mAP50": round(float(val.seg.map50), 4), "mask_mAP50_95": round(float(val.seg.map), 4)})
    Path("results").mkdir(exist_ok=True)
    Path("results/detector_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
