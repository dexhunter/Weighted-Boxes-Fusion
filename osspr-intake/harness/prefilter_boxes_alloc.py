# Count Python heap blocks allocated while prefilter_boxes builds one image.
# One RESULT line: sum of positive per-line increases of sys.getallocatedblocks
# inside prefilter_boxes, after one warmup call.

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ensemble_boxes.ensemble_boxes_wbf import prefilter_boxes


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


def count_allocations(fn):
    total = 0
    prev = sys.getallocatedblocks()

    def tracer(frame, event, arg):
        nonlocal total, prev
        if frame.f_code.co_name != "prefilter_boxes":
            return tracer
        now = sys.getallocatedblocks()
        if event == "line":
            delta = now - prev
            if delta > 0:
                total += delta
        prev = now
        return tracer

    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(None)
    return total


def main():
    warnings.filterwarnings("ignore")
    boxes, scores, labels = build_image()
    weights = np.ones(len(boxes))
    prefilter_boxes(boxes, scores, labels, weights, 0.0)
    n = count_allocations(lambda: prefilter_boxes(boxes, scores, labels, weights, 0.0))
    print("RESULT {}".format(n))


if __name__ == "__main__":
    main()
