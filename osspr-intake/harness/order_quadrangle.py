# Time order_quadrangle for every vertex-order pass in one quadrangle fusion.
# One RESULT line: milliseconds spent inside that function.

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ensemble_boxes.ensemble_boxes_wbf_quadrangle as quad


def build_image(n_models=4, n_boxes=150, seed=0):
    """Four scene-text models, 150 slightly irregular quads each, one class."""
    rng = np.random.default_rng(seed)
    boxes, scores, labels = [], [], []
    offsets = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for _ in range(n_models):
        cx = rng.uniform(0.1, 0.9, n_boxes)
        cy = rng.uniform(0.1, 0.9, n_boxes)
        w = rng.uniform(0.02, 0.12, n_boxes)
        h = rng.uniform(0.01, 0.05, n_boxes)
        jitter = rng.normal(0.0, 0.002, (8, n_boxes))
        coords = []
        k = 0
        for ox, oy in offsets:
            coords.append(np.clip(cx + ox * w / 2 + jitter[k], 0.0, 1.0))
            k += 1
            coords.append(np.clip(cy + oy * h / 2 + jitter[k], 0.0, 1.0))
            k += 1
        boxes.append(np.stack(coords, axis=1))
        scores.append(rng.uniform(0.05, 0.99, n_boxes))
        labels.append(np.zeros(n_boxes, dtype=int))
    return boxes, scores, labels


def main():
    warnings.filterwarnings("ignore")
    boxes, scores, labels = build_image()
    elapsed = [0.0]
    original = quad.order_quadrangle

    def timed(pts):
        t0 = time.perf_counter()
        out = original(pts)
        elapsed[0] += time.perf_counter() - t0
        return out

    quad.order_quadrangle = timed
    try:
        quad.weighted_boxes_fusion_quadrangle(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
        gc.collect()
        gc.disable()
        elapsed[0] = 0.0
        quad.weighted_boxes_fusion_quadrangle(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
    finally:
        gc.enable()
        quad.order_quadrangle = original
    print("RESULT {:.6f}".format(elapsed[0] * 1000.0))


if __name__ == "__main__":
    main()
