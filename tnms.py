from typing import Tuple
import numpy as np

def IoU(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Intersection over union
    @args:
        - a: a tuple of (start_time, end_time)
        - b: a tuple of (start_time, end_time)
    """
    start_time = max(a[0], b[0])
    end_time = min(a[1], b[1])
    if start_time >= end_time:
        return 0
    return (end_time - start_time) / (a[1] - a[0] + b[1] - b[0] - (end_time - start_time))

def non_maximum_suppression(spans: np.ndarray, scores: np.ndarray, threshold = 0.5):
    """
    Non-maximum suppression
    @args:
        - spans: a list of spans
            - span: a tuple of (start_time, end_time)
        - scores: a list of scores
        - threshold: a threshold for non-maximum suppression
    """
    # Sort by score
    sorted_indices = np.argsort(scores)
    keep = []
    while len(sorted_indices) > 0:
        # Get the last element
        keep.append(sorted_indices[-1])
        sorted_indices = sorted_indices[:-1]
        while len(sorted_indices) > 0:
            if IoU(spans[sorted_indices[-1]], spans[keep[-1]]) < threshold:
                break
            sorted_indices = sorted_indices[:-1]

    return keep

def get_weighted_span(spans, scores):
    """
    Get weighted span
    @args:
        - spans: a list of spans
            - span: a tuple of (start_time, end_time)
        - scores: a list of scores
    """
    total = scores.sum()
    return np.sum(spans * scores / total, axis=1)
