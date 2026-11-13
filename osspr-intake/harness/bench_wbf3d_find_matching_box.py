"""Time find_matching_box across one 3D weighted-boxes-fusion clustering pass.

Four models, 400 boxes each, 4 classes. Boxes are prefiltered once. The timed
loop is the per-label scan inside weighted_boxes_fusion_3d: each box is matched
against fused clusters, and a match updates that cluster the way fusion does.
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

from ensemble_boxes.ensemble_boxes_wbf_3d import (
    bb_intersection_over_union_3d,
    find_matching_box,
    get_weighted_box,
    prefilter_boxes,
)


def _model_boxes(n, n_cls, rng):
    centers = rng.uniform(0.08, 0.92, (n, 3))
    size = rng.uniform(0.03, 0.12, (n, 3))
    p1 = np.clip(centers - size / 2, 0, 1)
    p2 = np.clip(np.maximum(centers + size / 2, p1 + 1e-3), 0, 1)
    boxes = np.concatenate([p1, p2], axis=1)
    scores = rng.uniform(0.05, 0.99, n)
    labels = rng.integers(0, n_cls, n)
    return boxes, scores, labels


def build_groups():
    rng = np.random.default_rng(2)
    boxes_list = []
    scores_list = []
    labels_list = []
    for _ in range(4):
        b, s, l = _model_boxes(400, 4, rng)
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(l)
    filtered = prefilter_boxes(boxes_list, scores_list, labels_list, np.ones(4), 0.0)
    return [filtered[k] for k in filtered]


GROUPS = build_groups()

bb_intersection_over_union_3d(
    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5]),
)


def workload():
    """Return seconds spent inside find_matching_box for one fusion pass."""
    elapsed = 0.0
    acc = 0
    for boxes in GROUPS:
        clusters = []
        weighted = []
        for j in range(len(boxes)):
            t0 = time.perf_counter()
            index, _best_iou = find_matching_box(weighted, boxes[j], 0.55)
            elapsed += time.perf_counter() - t0
            acc += index
            if index != -1:
                clusters[index].append(boxes[j])
                weighted[index] = get_weighted_box(clusters[index], "avg")
            else:
                clusters.append([boxes[j].copy()])
                weighted.append(boxes[j].copy())
    return elapsed, acc


def main():
    for _ in range(2):
        workload()
    samples = []
    sink = None
    for _ in range(7):
        gc.collect()
        sink = workload()
        samples.append(sink[0] * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    median = float(np.median(arr))
    noise = float((arr.max() - arr.min()) / median * 100.0)
    print(json.dumps({
        "function": "ensemble_boxes.ensemble_boxes_wbf_3d.find_matching_box",
        "median_ms": median,
        "noise_pct": noise,
        "samples_ms": [round(float(x), 4) for x in arr],
        "n_boxes": int(sum(len(g) for g in GROUPS)),
        "sink": int(sink[1]),
    }))


if __name__ == "__main__":
    main()
