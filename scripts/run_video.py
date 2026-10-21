"""Run the full inspection pipeline on a video: detection/segmentation, tracking, zone events,
and optional open-vocabulary queries. Writes an annotated video and a JSON-lines event log.

  python scripts/run_video.py --video site.mp4 --weights best.onnx --classes 1 --zone 100,400,900,400,900,700,100,700
  python scripts/run_video.py --video site.mp4 --vlm "oil leak on floor" "open electrical panel"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="yolo11n-seg.pt", help=".pt (PyTorch) or .onnx (ONNX Runtime)")
    ap.add_argument("--classes", type=int, default=80, help="number of classes (ONNX only)")
    ap.add_argument("--zone", default=None, help="polygon x1,y1,x2,y2,... in pixels")
    ap.add_argument("--vlm", nargs="*", default=None, help="open-vocabulary text queries")
    ap.add_argument("--out", default="results/annotated.mp4")
    args = ap.parse_args()

    import cv2

    from perception.detector import ONNXDetector, YOLODetector
    from perception.geometry import CameraIntrinsics
    from perception.pipeline import InspectionPipeline

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = CameraIntrinsics(fx=w, fy=w, cx=w / 2, cy=h / 2, width=w, height=h)  # replace with calibrated values

    if args.vlm:
        from perception.vlm import OpenVocabDetector

        vlm = OpenVocabDetector()
        detector = lambda im: vlm(im, args.vlm)  # noqa: E731
    elif args.weights.endswith(".onnx"):
        detector = ONNXDetector(args.weights, num_classes=args.classes)
    else:
        detector = YOLODetector(args.weights)
    zones = None
    if args.zone:
        pts = np.array([float(v) for v in args.zone.split(",")]).reshape(-1, 2)
        zones = {"zone": pts}
    pipe = InspectionPipeline(detector, cam, zones=zones, fps=fps)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    log = open(Path(args.out).with_suffix(".jsonl"), "w")
    latencies = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe.process(frame[..., ::-1].copy())
        latencies.append(res.latency_ms["total"])
        for t in res.tracks:
            x1, y1, x2, y2 = map(int, t["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f"#{t['id']} {t['score']:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if res.detections.masks is not None:
            overlay = frame.copy()
            overlay[res.detections.masks.any(0)] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        if zones:
            cv2.polylines(frame, [zones["zone"].astype(np.int32)], True, (0, 255, 0), 2)
        writer.write(frame)
        if res.events:
            log.write(json.dumps(res.to_dict()) + "\n")
    writer.release()
    log.close()
    print(f"frames={len(latencies)} p50={np.median(latencies):.1f}ms p95={np.percentile(latencies, 95):.1f}ms -> {args.out}")


if __name__ == "__main__":
    main()
