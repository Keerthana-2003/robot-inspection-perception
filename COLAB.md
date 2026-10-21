# Getting real results on free Google Colab

Everything below runs on a free Colab T4 GPU (Runtime → Change runtime type → T4 GPU).
Paste each block into its own cell. Total time is about 1.5 to 2 hours, mostly training.

## 1. Setup
```python
!git clone https://github.com/Keerthana-2003/robot-inspection-perception
%cd robot-inspection-perception
!pip install -q -e ".[train,dev]"
!pytest -q
```

## 2. Detection + instance segmentation (crack-seg, downloads automatically)
```python
!python scripts/train_detector.py --data crack-seg.yaml --model yolo11n-seg.pt --epochs 50
!cat results/detector_metrics.json
```

## 3. Export to ONNX and TensorRT, benchmark speed and accuracy
```python
!pip install -q tensorrt
!python scripts/export_and_benchmark.py --weights runs/segment/crack_seg/weights/best.pt --data crack-seg.yaml
!cat results/benchmark.json
```
If the weights folder is named differently (crack_seg2, ...), use the path printed by step 2.

## 4. Visual anomaly detection (PatchCore on MVTec AD)
Download MVTec AD (from mvtec.com or the Kaggle mirror) and unzip it to `mvtec_ad/`, or upload
just a few categories (bottle, screw, metal_nut) to save time.
```python
!python scripts/train_anomaly.py --root mvtec_ad --categories bottle screw metal_nut
!cat results/anomaly_metrics.json
```

## 5. Video demo (tracking + zone events) and open-vocabulary queries
Upload any short inspection or warehouse video as `demo.mp4`.
```python
!python scripts/run_video.py --video demo.mp4 --weights runs/segment/crack_seg/weights/best.pt
!python scripts/run_video.py --video demo.mp4 --vlm "crack" "rust" "oil leak on floor" --out results/vlm.mp4
```

## 6. Save results
```python
!zip -r results.zip results
from google.colab import files; files.download("results.zip")
```
Send the four JSON files (detector_metrics, benchmark, anomaly_metrics) to update the README and resume.
