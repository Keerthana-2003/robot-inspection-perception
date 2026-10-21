# Robot Inspection Perception

Perception stack for inspection robots: find, segment, track, locate and measure things in the
physical world, flag what looks wrong, and run fast enough on edge hardware.

```
 RGB / PTZ camera ─► detector + instance segmentation (YOLO11-seg │ ONNX Runtime │ TensorRT)
                          │
                          ▼
                    ByteTrack tracker (Kalman + two-stage association)
                          │
 LiDAR / depth ──► calibration ─► fusion: 3D position + metric size of each detection
                          │
                          ▼
             zone events (enter / dwell / exit / stationary)  ──►  ROS 2 topics
 thermal camera ─► hotspot detection (warning / critical)
 normal images  ─► PatchCore visual anomaly detection + anomaly maps
 PTZ head       ─► active perception: pan / tilt / zoom to inspect the target
 operator text  ─► open-vocabulary detection with a vision-language model (OWLv2)
```

| Capability | Where | Notes |
|---|---|---|
| Object detection, classification, instance segmentation | `scripts/train_detector.py`, `perception/detector.py` | YOLO11-seg trained on crack-seg; own ONNX decoding, NMS and mask assembly |
| Multi-object tracking, video understanding | `perception/tracking.py`, `perception/events.py` | ByteTrack-style tracker with Kalman filter; zone and dwell events |
| Visual anomaly detection | `perception/anomaly.py`, `scripts/train_anomaly.py` | PatchCore implemented from the paper: WideResNet-50 features, k-center coreset, kNN scoring, anomaly maps |
| Sensor calibration | `perception/calibration.py` | Checkerboard intrinsics (OpenCV); LiDAR-camera extrinsics via DLT + Gauss-Newton |
| Sensor fusion and 3D | `perception/fusion.py`, `perception/geometry.py` | LiDAR or depth image → robust object depth, 3D centre, metric width / height |
| Thermal inspection | `perception/thermal.py` | Radiometric hotspots relative to ambient with severity |
| Active perception (PTZ) | `perception/ptz.py` | Intrinsics-based pan / tilt with deadband and limits, zoom to target fill |
| Vision-language model | `perception/vlm.py` | Open-vocabulary queries with OWLv2 |
| Edge optimisation | `scripts/export_and_benchmark.py`, `docker/Dockerfile.jetson` | ONNX, TensorRT FP16 / INT8; p50 / p95 latency, FPS, memory and mAP after export |
| ROS 2 | `ros2/inspection_perception_ros` | Node: Image + PointCloud2 + thermal in, JSON results and PTZ commands out |
| C++ | `cpp/` | Header-only C++17 kernels (IoU, NMS, LiDAR projection, box depth) with tests |
| Evaluation | `perception/metrics.py` | COCO-style mAP@0.5:0.95, mIoU, MOTA / ID switches, AUROC |

## Results

Produced by the scripts in `COLAB.md` on a free Colab T4 GPU. _(Fill in after running.)_

| Task | Dataset | Metric | Value |
|---|---|---|---|
| Instance segmentation | crack-seg | box mAP50-95 / mask mAP50-95 | |
| Edge inference | crack-seg, 640 px | PyTorch → ONNX → TensorRT FP16 p50 latency | |
| Visual anomaly detection | MVTec AD (bottle, screw, metal_nut) | image AUROC / pixel AUROC | |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # 27 tests, CPU only, no downloads
pip install -e ".[train]"                   # PyTorch, Ultralytics, transformers
```

See `COLAB.md` for training, export, benchmarks and demos on a free GPU.

## Run on a robot (ROS 2 Jazzy)

```bash
docker build -f docker/Dockerfile.ros2 -t inspection-ros2 .
docker run --net=host -v $PWD/models:/models -v $PWD/configs:/config inspection-ros2 \
  ros2 launch inspection_perception_ros inspection.launch.py model:=/models/best.onnx calibration:=/config/calibration.yaml
```

Topics: subscribes `/camera/color/image_raw`, `/lidar/points`, `/thermal/temperature`; publishes
`inspection/results` (JSON) and `ptz/command`. Calibration format: `configs/calibration.example.yaml`.

On an NVIDIA Jetson, build `docker/Dockerfile.jetson` on the device and build TensorRT engines
there with `scripts/export_and_benchmark.py` (engines are hardware-specific).

## Design decisions and trade-offs

- **ONNX Runtime as the deployment boundary.** Training uses Ultralytics; deployment uses our own
  pre/post-processing on ONNX Runtime, so the robot does not need PyTorch and the same code runs
  with CPU, CUDA or TensorRT execution providers.
- **Robust depth instead of mean depth.** LiDAR points inside a 2D box include background; fusion
  shrinks the box and takes the median of the nearest dense cluster, which keeps depth error low when
  most points in the box hit the wall behind the object.
- **Two-stage tracking association.** Low-confidence detections are used only to continue existing
  tracks, which keeps IDs stable through partial occlusion without creating false tracks.
- **PatchCore for anomalies.** Needs only normal images (defects are rare and varied on real sites),
  gives pixel-level maps for operators, and the threshold is set from held-out normal images to
  control false alarms.
- **PTZ control through intrinsics.** Angles come from pixel offset and focal length at the current
  zoom, so the same gains work at any zoom; zoom changes only once the target is centred.

## Tests and CI

GitHub Actions runs on every push: lint, 27 Python tests (geometry, checkerboard and LiDAR-camera
calibration, fusion, tracking + MOTA, mAP, PatchCore core, thermal, PTZ, events, ONNX decoding with
masks, ROS message conversion, full pipeline), the C++ tests, a `colcon build` of the ROS 2 package
in a ROS 2 Jazzy container, and a Docker build.
