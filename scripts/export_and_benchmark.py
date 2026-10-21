"""Export a trained YOLO model for edge deployment and benchmark latency and memory.

Backends compared on the same images:
  * PyTorch (Ultralytics)              baseline
  * ONNX Runtime (our ONNXDetector)     CPU / CUDA
  * TensorRT FP16 and INT8 engines      NVIDIA GPU or Jetson (skipped automatically if unavailable)

  python scripts/export_and_benchmark.py --weights runs/segment/crack_seg/weights/best.pt --data crack-seg.yaml
Writes results/benchmark.json with p50 / p95 latency (ms), FPS, peak memory (MB), model size (MB)
and mAP after export, so accuracy loss from FP16 / INT8 is visible next to the speed-up.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if os.uname().sysname == "Linux" else r / (1024 * 1024)


def time_it(fn, images, warmup=10, runs=100):
    for i in range(warmup):
        fn(images[i % len(images)])
    lat = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(images[i % len(images)])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.asarray(lat)
    return {"p50_ms": round(float(np.median(lat)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "fps": round(1000 / float(np.mean(lat)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="crack-seg.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--images", default=None, help="folder of sample images (default: dataset val images)")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    from perception.detector import ONNXDetector

    model = YOLO(args.weights)
    names = model.names
    if args.images:
        paths = sorted(Path(args.images).glob("*.jpg"))[:50]
    else:
        from ultralytics.data.utils import check_det_dataset

        paths = sorted(Path(check_det_dataset(args.data)["val"]).glob("*.jpg"))[:50]
    images = [cv2.imread(str(p))[..., ::-1].copy() for p in paths]
    results = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "backends": {}}

    def add(name, fn, path, val_model=None):
        entry = time_it(fn, images, runs=args.runs)
        entry["peak_rss_mb"] = round(peak_rss_mb(), 1)
        entry["model_mb"] = round(Path(path).stat().st_size / 1e6, 1)
        if val_model is not None:
            v = val_model.val(data=args.data, imgsz=args.imgsz, verbose=False)
            entry["box_mAP50_95"] = round(float(v.box.map), 4)
            if getattr(v, "seg", None) is not None:
                entry["mask_mAP50_95"] = round(float(v.seg.map), 4)
        results["backends"][name] = entry
        print(name, entry)

    add("pytorch", lambda im: model.predict(im[..., ::-1], imgsz=args.imgsz, verbose=False), args.weights, model)

    onnx_path = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
    onnx_det = ONNXDetector(onnx_path, num_classes=len(names), input_size=args.imgsz, names=names,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    add("onnxruntime", onnx_det, onnx_path, YOLO(onnx_path, task=model.task))

    if torch.cuda.is_available():
        for label, kwargs in [("tensorrt_fp16", {"half": True}), ("tensorrt_int8", {"int8": True, "data": args.data})]:
            try:
                engine = YOLO(args.weights).export(format="engine", imgsz=args.imgsz, **kwargs)
                trt = YOLO(engine, task=model.task)
                add(label, lambda im, m=trt: m.predict(im[..., ::-1], imgsz=args.imgsz, verbose=False), engine, trt)
            except Exception as err:  # TensorRT not installed or unsupported layer
                results["backends"][label] = {"skipped": str(err)[:200]}
    base = results["backends"]["pytorch"]["p50_ms"]
    for b in results["backends"].values():
        if "p50_ms" in b:
            b["speedup_vs_pytorch"] = round(base / b["p50_ms"], 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
