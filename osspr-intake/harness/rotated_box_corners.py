# Time rotated_box_corners for every pair considered while fusing one image.
# One RESULT line: milliseconds spent inside that function.

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ensemble_boxes.ensemble_boxes_wbf_rotated as rot


def build_image(n_models=4, n_boxes=100, seed=0):
    """Four rotated-box models, 100 boxes each, one class (aerial / oriented detection)."""
    rng = np.random.default_rng(seed)
    boxes, scores, labels = [], [], []
    for _ in range(n_models):
        boxes.append(np.stack((
            rng.uniform(0.1, 0.9, n_boxes),
            rng.uniform(0.1, 0.9, n_boxes),
            rng.uniform(0.02, 0.15, n_boxes),
            rng.uniform(0.01, 0.08, n_boxes),
            rng.uniform(-90.0, 90.0, n_boxes),
        ), axis=1))
        scores.append(rng.uniform(0.05, 0.99, n_boxes))
        labels.append(np.zeros(n_boxes, dtype=int))
    return boxes, scores, labels


def main():
    warnings.filterwarnings("ignore")
    boxes, scores, labels = build_image()
    elapsed = [0.0]
    original = rot.rotated_box_corners

    def timed(cx, cy, w, h, angle_deg):
        t0 = time.perf_counter()
        out = original(cx, cy, w, h, angle_deg)
        elapsed[0] += time.perf_counter() - t0
        return out

    rot.rotated_box_corners = timed
    try:
        rot.weighted_boxes_fusion_rotated(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
        gc.collect()
        gc.disable()
        elapsed[0] = 0.0
        rot.weighted_boxes_fusion_rotated(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
    finally:
        gc.enable()
        rot.rotated_box_corners = original
    print("RESULT {:.6f}".format(elapsed[0] * 1000.0))


if __name__ == "__main__":
    main()
