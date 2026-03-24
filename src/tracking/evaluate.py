"""Tracking evaluation using MOTA/IDF1 metrics."""
import numpy as np
import motmetrics as mm


def create_accumulator():
    """Create a fresh MOT accumulator."""
    return mm.MOTAccumulator(auto_id=True)


def update_accumulator(acc, gt_ids, gt_boxes, hyp_ids, hyp_boxes, max_iou=0.5):
    """Add one frame of GT and hypothesis tracks to accumulator."""
    if len(gt_boxes) == 0 or len(hyp_boxes) == 0:
        acc.update(gt_ids, hyp_ids,
                   np.empty((len(gt_ids), len(hyp_ids))))
        return

    gt_ltwh = _xyxy_to_ltwh(np.array(gt_boxes))
    hyp_ltwh = _xyxy_to_ltwh(np.array(hyp_boxes))

    distances = mm.distances.iou_matrix(gt_ltwh, hyp_ltwh, max_iou=max_iou)
    acc.update(gt_ids, hyp_ids, distances)


def compute_metrics(accumulators, names=None):
    """Compute MOTA, IDF1 and other tracking metrics from accumulators."""
    if not isinstance(accumulators, list):
        accumulators = [accumulators]
    if names is None:
        names = [f"seq_{i}" for i in range(len(accumulators))]

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        metrics=["mota", "idf1", "num_switches", "mostly_tracked",
                 "mostly_lost", "num_false_positives", "num_misses",
                 "precision", "recall"],
        names=names,
        generate_overall=True,
    )
    return summary


def _xyxy_to_ltwh(boxes):
    """Convert (N, 4) xyxy boxes to ltwh format."""
    ltwh = np.zeros_like(boxes)
    ltwh[:, 0] = boxes[:, 0]           # left
    ltwh[:, 1] = boxes[:, 1]           # top
    ltwh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    ltwh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return ltwh
