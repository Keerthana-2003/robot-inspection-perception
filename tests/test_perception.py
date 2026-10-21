from __future__ import annotations

import numpy as np
import pytest

from perception.anomaly import PatchCoreMemory, greedy_coreset
from perception.boxes import iou_matrix, letterbox_params, nms, unletterbox
from perception.detector import Detections, ONNXDetector, decode_yolo
from perception.events import ZoneMonitor
from perception.fusion import fuse_depth_image, fuse_lidar
from perception.geometry import CameraIntrinsics, make_transform
from perception.metrics import COCO_IOUS, auroc, mean_average_precision, mota, segmentation_iou
from perception.pipeline import InspectionPipeline
from perception.ptz import PTZController
from perception.thermal import find_hotspots
from perception.tracking import ByteTracker

CAM = CameraIntrinsics(fx=800, fy=800, cx=640, cy=360, width=1280, height=720)
T_CAM_LIDAR = make_transform(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float), [0, 0, 0])


# ---------------------------------------------------------------- boxes
def test_iou_and_class_aware_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]], float)
    assert iou_matrix(boxes[:1], boxes[1:2])[0, 0] == pytest.approx(81 / 119)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7, 0.6]), 0.5, classes=np.array([0, 0, 1, 0]))
    assert sorted(keep.tolist()) == [0, 2, 3]  # box 1 suppressed; box 2 survives (other class)


def test_unletterbox_inverts_letterbox():
    src = (480, 640)
    r, px, py = letterbox_params(src, 640)
    original = np.array([[100, 50, 300, 200]], float)
    model_space = original * r + [px, py, px, py]
    np.testing.assert_allclose(unletterbox(model_space, src, 640), original, atol=1e-6)


# ---------------------------------------------------------------- detector decoding
def _yolo_output(boxes_xywh, class_scores, extra=None):
    cols = [np.r_[b, s] if extra is None else np.r_[b, s, extra[i]] for i, (b, s) in enumerate(zip(boxes_xywh, class_scores))]
    return np.array(cols, np.float32).T[None]


def test_decode_yolo_filters_and_suppresses():
    out = _yolo_output([[100, 100, 40, 40], [102, 101, 40, 40], [300, 300, 20, 20], [500, 500, 10, 10]],
                       [[0.9, 0.1], [0.8, 0.1], [0.1, 0.7], [0.05, 0.1]])
    boxes, scores, labels, coeffs = decode_yolo(out, num_classes=2, conf=0.25, iou=0.5)
    assert len(boxes) == 2 and labels.tolist() == [0, 1] and coeffs is None
    np.testing.assert_allclose(boxes[0], [80, 80, 120, 120])


class FakeSession:
    """Stands in for onnxruntime.InferenceSession; returns a fixed YOLO-seg output."""

    class _In:
        name = "images"

    def __init__(self, outputs):
        self.outputs, self.last_input = outputs, None

    def get_inputs(self):
        return [self._In()]

    def run(self, _, feeds):
        self.last_input = feeds["images"]
        return self.outputs


def test_onnx_detector_end_to_end_with_masks():
    # one object at model coords (320, 320, 64x64); single prototype that is 1 everywhere
    out = _yolo_output([[320, 320, 64, 64]], [[0.95]], extra=[[8.0]])
    protos = np.ones((1, 1, 160, 160), np.float32)
    sess = FakeSession([out, protos])
    det = ONNXDetector("unused.onnx", num_classes=1, input_size=640, session=sess)
    image = np.zeros((480, 640, 3), np.uint8)
    res = det(image)
    assert sess.last_input.shape == (1, 3, 640, 640) and sess.last_input.dtype == np.float32
    assert len(res) == 1 and res.masks.shape == (1, 480, 640)
    np.testing.assert_allclose(res.boxes[0], [288, 208, 352, 272], atol=1e-6)  # pad_y = 80 removed
    ys, xs = np.where(res.masks[0])
    assert 285 <= xs.min() <= 291 and 349 <= xs.max() <= 355


# ---------------------------------------------------------------- fusion
def _scene(depth_obj=5.0, depth_wall=20.0):
    rng = np.random.default_rng(1)
    obj = np.c_[np.full(400, depth_obj), rng.uniform(-0.25, 0.25, 400), rng.uniform(-0.5, 0.5, 400)]  # 0.5 x 1.0 m
    wall = np.c_[np.full(3000, depth_wall), rng.uniform(-8, 8, 3000), rng.uniform(-3, 3, 3000)]
    return np.r_[obj, wall]


def test_lidar_fusion_ignores_background_and_measures_object():
    box = np.array([[640 - 40, 360 - 80, 640 + 40, 360 + 80]], float)  # 0.5 x 1.0 m at 5 m = 80 x 160 px
    o = fuse_lidar(box, [0.9], [0], _scene(), T_CAM_LIDAR, CAM)[0]
    assert o.depth_m == pytest.approx(5.0, abs=0.05)
    assert o.size_m[0] == pytest.approx(0.5, abs=0.02) and o.size_m[1] == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(o.center_cam, [0, 0, 5], atol=0.1)


def test_lidar_fusion_reports_missing_support():
    o = fuse_lidar(np.array([[0, 0, 20, 20]], float), [0.9], [0], np.zeros((0, 3)), T_CAM_LIDAR, CAM)[0]
    assert o.depth_m is None and o.support_points == 0


def test_depth_image_fusion():
    depth = np.full((720, 1280), 12.0, np.float32)
    depth[300:420, 600:680] = 3.0
    o = fuse_depth_image(np.array([[600, 300, 680, 420]], float), [0.8], [1], depth, CAM)[0]
    assert o.depth_m == pytest.approx(3.0) and o.label == 1


# ---------------------------------------------------------------- tracking + MOTA
def _moving_objects(frames=30):
    gt = []
    for f in range(frames):
        gt.append({1: np.array([10 + 5 * f, 100, 50 + 5 * f, 180], float),
                   2: np.array([600 - 4 * f, 300, 650 - 4 * f, 360], float)})
    return gt


def test_tracker_keeps_ids_through_low_confidence_occlusion():
    gt = _moving_objects()
    tracker, pred = ByteTracker(min_hits=1), []
    for f, objs in enumerate(gt):
        boxes = np.array(list(objs.values()))
        scores = np.array([0.9, 0.9])
        if 10 <= f < 15:
            scores[0] = 0.2  # partially occluded: low confidence, rescued by the second association stage
        tracks = tracker.update(boxes, scores)
        pred.append({t.track_id: t.box for t in tracks})
    m = mota(gt, pred)
    assert m["IDSW"] == 0 and m["MOTA"] > 0.95
    assert {t.track_id for t in tracker.tracks} == {1, 2}


def test_tracker_coasts_through_missed_frames():
    tracker = ByteTracker(min_hits=1)
    for f in range(10):
        tracker.update(np.array([[10 + 5 * f, 100, 50 + 5 * f, 180]], float), [0.9])
    for _ in range(3):
        tracker.update(np.zeros((0, 4)), np.zeros(0))  # object not detected for 3 frames
    out = tracker.update(np.array([[10 + 5 * 13, 100, 50 + 5 * 13, 180]], float), [0.9])
    assert [t.track_id for t in out] == [1]


# ---------------------------------------------------------------- metrics
def test_map_perfect_and_with_false_positive():
    gts = [{"image_id": 0, "box": [0, 0, 10, 10], "label": 0}, {"image_id": 1, "box": [5, 5, 20, 20], "label": 0}]
    perfect = [dict(g, score=0.9) for g in gts]
    assert mean_average_precision(perfect, gts, COCO_IOUS)["mAP"] == pytest.approx(1.0)
    with_fp = perfect + [{"image_id": 0, "box": [50, 50, 60, 60], "label": 0, "score": 0.95}]
    assert 0.6 < mean_average_precision(with_fp, gts)["mAP"] < 0.75


def test_segmentation_iou_and_auroc():
    gt = np.zeros((4, 4), int)
    gt[:2] = 1
    pred = gt.copy()
    pred[2, 0] = 1
    res = segmentation_iou(pred, gt, 2)
    assert res["per_class"][1] == pytest.approx(8 / 9) and 0.8 < res["mIoU"] < 1
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5


# ---------------------------------------------------------------- anomaly detection (PatchCore core)
def test_patchcore_memory_scores_anomalies_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (2000, 64))
    mem = PatchCoreMemory(coreset_ratio=0.25).fit(normal)
    assert len(mem.bank) == 500
    mem.calibrate_threshold([rng.normal(0, 1, (49, 64)) for _ in range(20)], quantile=1.0)
    good = mem.score_patches(rng.normal(0, 1, (49, 64)), (7, 7))
    bad_patches = rng.normal(0, 1, (49, 64))
    bad_patches[24] += 6.0  # one defective patch in the centre
    bad = mem.score_patches(bad_patches, (7, 7))
    assert bad.score > good.score and bad.is_anomalous
    assert np.unravel_index(bad.patch_map.argmax(), (7, 7)) == (3, 3)  # localised


def test_greedy_coreset_covers_clusters():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 0.1, (500, 8)), rng.normal(10, 0.1, (5, 8))]  # a rare mode
    idx = greedy_coreset(x, ratio=0.01)
    assert (idx >= 500).any()  # k-center greedy keeps the rare mode; random sampling usually would not


# ---------------------------------------------------------------- thermal, PTZ, events
def test_thermal_hotspots_with_severity():
    t = np.full((120, 160), 25.0)
    t[10:20, 10:20] = 40.0   # +15 C warning
    t[60:70, 100:110] = 80.0  # +55 C critical
    t[100, 5] = 90.0          # single noisy pixel ignored
    ambient, spots = find_hotspots(t)
    assert ambient == 25.0 and [s.severity for s in spots] == ["critical", "warning"]
    assert spots[0].box == (100, 60, 109, 69)


def test_ptz_centres_then_zooms():
    ptz = PTZController(CAM)
    right = ptz.command(np.array([1000, 340, 1040, 380]), current_zoom=1.0)
    assert right.pan_deg > 0 and right.tilt_deg == 0 and right.zoom == 1.0 and not right.centered
    centred_small = ptz.command(np.array([630, 350, 650, 370]), current_zoom=1.0)
    assert centred_small.pan_deg == 0 and centred_small.zoom > 10
    up = ptz.command(np.array([620, 50, 660, 90]), current_zoom=2.0)
    assert up.tilt_deg > 0 and abs(up.tilt_deg) < abs(ptz.command(np.array([620, 50, 660, 90]), 1.0).tilt_deg)


def test_zone_events_enter_dwell_exit_and_stationary():
    zone = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], float)
    mon = ZoneMonitor({"restricted": zone}, fps=10, dwell_s=1.0, stationary_s=0.5)
    events = []
    for f in range(1, 31):
        x = 0 if f < 5 else (150 if f < 25 else 500)
        events += mon.update(f, [(7, np.array([x, 150, x + 40, 250], float))])
    types = [e.type for e in events]
    assert types.index("zone_enter") < types.index("dwell") < types.index("zone_exit")
    assert "stationary" in types


# ---------------------------------------------------------------- pipeline
class MovingBoxDetector:
    def __init__(self):
        self.f = 0

    def __call__(self, image):
        self.f += 1
        x = 560 + 2 * self.f
        return Detections(np.array([[x, 200, x + 160, 520]], float), np.array([0.9]), np.array([0]))


def test_pipeline_runs_all_stages():
    pipe = InspectionPipeline(MovingBoxDetector(), CAM, tracker=ByteTracker(min_hits=1), fps=10,
                              zones={"aisle": np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], float)},
                              T_cam_lidar=T_CAM_LIDAR, ptz=PTZController(CAM))
    image = np.zeros((720, 1280, 3), np.uint8)
    thermal = np.full((120, 160), 22.0)
    thermal[50:60, 50:60] = 70.0
    results = [pipe.process(image, lidar_points=_scene(4.0), thermal_c=thermal) for _ in range(3)]
    last = results[-1].to_dict()
    assert last["tracks"][0]["id"] == 1 and last["objects_3d"][0]["depth_m"] == pytest.approx(4.0, abs=0.05)
    assert results[0].events[0].type == "zone_enter" and last["hotspots"][0]["severity"] == "critical"
    assert last["ptz"] is not None and set(last["latency_ms"]) == {"detect", "track", "fusion", "total"}
