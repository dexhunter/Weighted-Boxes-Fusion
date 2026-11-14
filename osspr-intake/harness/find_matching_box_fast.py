# Time find_matching_box_fast across one dense detection image.
# One RESULT line: milliseconds spent inside that function.

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ensemble_boxes.ensemble_boxes_wbf as wbf


def build_image(n_models=4, n_boxes=800, seed=0):
    """Four models, 800 boxes each, one class. Same shape as a dense ensemble."""
    rng = np.random.default_rng(seed)
    boxes, scores, labels = [], [], []
    for _ in range(n_models):
        centers = rng.uniform(0.05, 0.95, (n_boxes, 2))
        sizes = rng.uniform(0.02, 0.12, (n_boxes, 2))
        boxes.append(np.clip(np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1), 0, 1))
        scores.append(rng.uniform(0.05, 0.99, n_boxes))
        labels.append(np.zeros(n_boxes, dtype=int))
    return boxes, scores, labels


def main():
    warnings.filterwarnings("ignore")
    boxes, scores, labels = build_image()
    elapsed = [0.0]
    original = wbf.find_matching_box_fast

    def timed(boxes_list, new_box, match_iou):
        t0 = time.perf_counter()
        out = original(boxes_list, new_box, match_iou)
        elapsed[0] += time.perf_counter() - t0
        return out

    wbf.find_matching_box_fast = timed
    try:
        # Warm the allocator and the call path, then time one image.
        wbf.weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
        gc.collect()
        gc.disable()
        elapsed[0] = 0.0
        wbf.weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.0)
    finally:
        gc.enable()
        wbf.find_matching_box_fast = original
    print("RESULT {:.6f}".format(elapsed[0] * 1000.0))


if __name__ == "__main__":
    main()
