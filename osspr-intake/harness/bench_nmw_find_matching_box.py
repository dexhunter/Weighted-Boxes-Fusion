"""Time find_matching_box across one non-maximum-weighted clustering pass.

Five models, 600 boxes each, 8 classes. Boxes are prefiltered once, then each
class is scanned the way non_maximum_weighted does: every box is compared with
the cluster representatives collected so far.
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

from ensemble_boxes.ensemble_boxes_nmw import (
    bb_intersection_over_union,
    find_matching_box,
    prefilter_boxes,
)


def _model_boxes(n, n_cls, rng):
    cx = rng.uniform(0.05, 0.95, n)
    cy = rng.uniform(0.05, 0.95, n)
    w = rng.uniform(0.02, 0.10, n)
    h = rng.uniform(0.02, 0.10, n)
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(np.maximum(cx + w / 2, x1 + 1e-3), 0, 1)
    y2 = np.clip(np.maximum(cy + h / 2, y1 + 1e-3), 0, 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = rng.uniform(0.05, 0.99, n)
    labels = rng.integers(0, n_cls, n)
    return boxes, scores, labels


def build_groups():
    rng = np.random.default_rng(1)
    boxes_list = []
    scores_list = []
    labels_list = []
    for _ in range(5):
        b, s, l = _model_boxes(600, 8, rng)
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(l)
    filtered = prefilter_boxes(boxes_list, scores_list, labels_list, np.ones(5), 0.0)
    return [filtered[k] for k in filtered]


GROUPS = build_groups()

# Compile the scalar IoU kernel before the timed samples.
bb_intersection_over_union(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.5, 0.5]))


def workload():
    acc = 0
    for boxes in GROUPS:
        main_boxes = []
        for j in range(len(boxes)):
            index, _best_iou = find_matching_box(main_boxes, boxes[j], 0.55)
            acc += index
            if index == -1:
                main_boxes.append(boxes[j])
    return acc


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
        "function": "ensemble_boxes.ensemble_boxes_nmw.find_matching_box",
        "median_ms": median,
        "noise_pct": noise,
        "samples_ms": [round(float(x), 4) for x in arr],
        "n_boxes": int(sum(len(g) for g in GROUPS)),
        "sink": int(sink),
    }))


if __name__ == "__main__":
    main()
