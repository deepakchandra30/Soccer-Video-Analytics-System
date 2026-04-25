"""Tests for NMS post-processing and prediction output format."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from src.models.temporal.postprocess import (
    nms_detections,
    save_predictions,
    sliding_window_inference,
)
from src.models.temporal.tsm import TSMSpottingHead


class TestNMSDetections:
    def test_format(self):
        # 100 frames, 17 classes, mostly zeros with a few peaks
        scores = np.zeros((100, 17), dtype=np.float32)
        scores[20, 0] = 0.9  # one clear peak
        scores[60, 5] = 0.7
        preds = nms_detections(scores, nms_window=30, confidence_threshold=0.2)
        assert len(preds) >= 2
        for p in preds:
            assert "label" in p
            assert "half" in p
            assert "position" in p
            assert "confidence" in p

    def test_position_in_ms(self):
        scores = np.zeros((100, 17), dtype=np.float32)
        scores[20, 0] = 0.8
        preds = nms_detections(scores, nms_window=30, framerate=2, half=1)
        # nms_detections emits string-typed fields to satisfy the SoccerNet
        # evaluator (see postprocess.py) — cast when comparing numerically.
        # frame 20 at 2fps = 20 * 500 = 10000 ms
        goal_pred = [p for p in preds if int(p["position"]) == 10000]
        assert len(goal_pred) == 1

    def test_suppresses_duplicates(self):
        scores = np.zeros((100, 17), dtype=np.float32)
        scores[20, 0] = 0.9
        scores[23, 0] = 0.7  # 3 frames away, within nms_window=10
        preds = nms_detections(scores, nms_window=10, confidence_threshold=0.2)
        class0_preds = [p for p in preds if float(p["confidence"]) >= 0.7]
        # only the higher-confidence peak should survive
        assert len([p for p in class0_preds
                     if 9000 <= int(p["position"]) <= 12000]) == 1


class TestSavePredictions:
    def test_writes_valid_json(self):
        preds = [
            {"label": "Goal", "half": 1, "position": 10000, "confidence": 0.9},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            save_predictions(preds, path)
            with open(path) as f:
                data = json.load(f)
            assert "predictions" in data
            assert len(data["predictions"]) == 1
        finally:
            os.unlink(path)


class TestSlidingWindowInference:
    def test_output_shape(self):
        model = TSMSpottingHead(feat_dim=512, num_classes=17, hidden_dim=32)
        features = torch.randn(100, 512)
        result = sliding_window_inference(
            model, features, window_size=40, stride=20, device="cpu"
        )
        assert result.shape == (100, 17)
