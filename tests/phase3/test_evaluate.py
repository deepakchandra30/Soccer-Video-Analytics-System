"""Tests for tracking evaluation (MOTA/IDF1)."""
import numpy as np
import pytest

from src.tracking.evaluate import (
    create_accumulator, update_accumulator, compute_metrics,
)


class TestTrackingEvaluation:
    def test_accumulator_creation(self):
        acc = create_accumulator()
        assert acc is not None

    def test_perfect_tracking(self):
        """Perfect tracking should give MOTA close to 1.0."""
        acc = create_accumulator()
        gt_boxes = np.array([[10, 10, 50, 100], [100, 10, 150, 100]],
                            dtype=np.float32)
        for _ in range(10):
            update_accumulator(acc, [1, 2], gt_boxes, [1, 2], gt_boxes)

        summary = compute_metrics(acc, names=["test"])
        mota = summary.loc["test", "mota"]
        assert mota > 0.9

    def test_no_detections(self):
        """No detections should give MOTA of 0 or negative (all misses)."""
        acc = create_accumulator()
        gt_boxes = np.array([[10, 10, 50, 100]], dtype=np.float32)
        for _ in range(5):
            update_accumulator(acc, [1], gt_boxes, [], np.empty((0, 4)))

        summary = compute_metrics(acc, names=["test"])
        mota = summary.loc["test", "mota"]
        assert mota <= 0

    def test_metrics_columns(self):
        acc = create_accumulator()
        gt_boxes = np.array([[10, 10, 50, 100]], dtype=np.float32)
        update_accumulator(acc, [1], gt_boxes, [1], gt_boxes)
        summary = compute_metrics(acc)
        assert "mota" in summary.columns
        assert "idf1" in summary.columns
