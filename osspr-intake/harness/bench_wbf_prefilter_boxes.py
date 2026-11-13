"""Time prefilter_boxes on a multi-model detection ensemble.

Five models, 3000 boxes each, 8 classes, coordinates already in [0, 1].
This is the validation and per-label grouping step at the start of every
weighted_boxes_fusion call. skip_box_thr is the library default, 0.
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

from ensemble_boxes.ensemble_boxes_wbf import prefilter_boxes


def _model_boxes(n, n_cls, rng):
    cx = rng.uniform(0.05, 0.95, n)
    cy = rng.uniform(0.05, 0.95, n)
    w = rng.uniform(0.02, 0.08, n)
    h = rng.uniform(0.02, 0.08, n)
    x1 = np.clip(cx - w / 2, 0, 1)
    y1 = np.clip(cy - h / 2, 0, 1)
    x2 = np.clip(np.maximum(cx + w / 2, x1 + 1e-3), 0, 1)
    y2 = np.clip(np.maximum(cy + h / 2, y1 + 1e-3), 0, 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = rng.uniform(0.05, 0.99, n)
    labels = rng.integers(0, n_cls, n)
    return boxes, scores, labels


def build_workload():
    rng = np.random.default_rng(3)
    boxes_list = []
    scores_list = []
    labels_list = []
    for _ in range(5):
        b, s, l = _model_boxes(3000, 8, rng)
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(l)
    return boxes_list, scores_list, labels_list


BOXES, SCORES, LABELS = build_workload()
WEIGHTS = np.ones(len(BOXES))


def workload():
    grouped = prefilter_boxes(BOXES, SCORES, LABELS, WEIGHTS, 0.0)
    return len(grouped), int(sum(len(v) for v in grouped.values()))


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
        "function": "ensemble_boxes.ensemble_boxes_wbf.prefilter_boxes",
        "median_ms": median,
        "noise_pct": noise,
        "samples_ms": [round(float(x), 4) for x in arr],
        "n_boxes": int(sum(len(b) for b in BOXES)),
        "sink": sink,
    }))


if __name__ == "__main__":
    main()
