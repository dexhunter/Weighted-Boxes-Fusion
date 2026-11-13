"""Time cpu_soft_nms_float on one class of ensemble detections.

Four single-class models, 800 boxes each (3200 boxes). That is the array
soft_nms / nms_method passes in for one label after concatenating models.
Gaussian soft-NMS, library defaults: iou 0.5, sigma 0.5, score floor 0.001.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble_boxes.ensemble_boxes_nms import cpu_soft_nms_float


def _model_boxes(n, rng):
    cx = rng.uniform(0.05, 0.95, n)
    cy = rng.uniform(0.05, 0.95, n)
    w = rng.uniform(0.02, 0.12, n)
    h = rng.uniform(0.02, 0.12, n)
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(np.maximum(cx + w / 2, x1 + 1e-3), 0, 1)
    y2 = np.clip(np.maximum(cy + h / 2, y1 + 1e-3), 0, 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = rng.uniform(0.05, 0.99, n)
    return boxes, scores


def build_workload():
    rng = np.random.default_rng(0)
    boxes = []
    scores = []
    for _ in range(4):
        b, s = _model_boxes(800, rng)
        boxes.append(b)
        scores.append(s)
    return np.concatenate(boxes), np.concatenate(scores)


DETS, SCORES = build_workload()


def workload():
    keep, kept_scores = cpu_soft_nms_float(
        DETS.copy(), SCORES.copy(), 0.5, 0.5, 0.001, 2
    )
    return int(keep.shape[0]), float(kept_scores[0])


def main():
    for _ in range(2):
        workload()
    samples = []
    sink = None
    for _ in range(7):
        gc.collect()
        t0 = time.perf_counter()
        sink = workload()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    median = float(np.median(arr))
    noise = float((arr.max() - arr.min()) / median * 100.0)
    print(json.dumps({
        "function": "ensemble_boxes.ensemble_boxes_nms.cpu_soft_nms_float",
        "median_ms": median,
        "noise_pct": noise,
        "samples_ms": [round(float(x), 4) for x in arr],
        "n_boxes": int(DETS.shape[0]),
        "sink": sink,
    }))


if __name__ == "__main__":
    main()
