"""Tests for YOLOv8 player detector."""
import numpy as np
import pytest

from src.tracking.detector import PlayerDetector


@pytest.fixture(scope="module")
def detector():
    """Use the smallest model for test speed."""
    return PlayerDetector(model_name="yolov8n.pt", confidence=0.25)


class TestPlayerDetector:
    def test_detect_returns_correct_keys(self, detector, dummy_frame):
        result = detector.detect(dummy_frame)
        assert "boxes" in result
        assert "confidences" in result
        assert "class_ids" in result

    def test_detect_shapes_match(self, detector, dummy_frame):
        result = detector.detect(dummy_frame)
        n = len(result["confidences"])
        assert result["boxes"].shape == (n, 4)
        assert result["class_ids"].shape == (n,)

    def test_detect_person_only(self, detector, dummy_frame):
        result = detector.detect(dummy_frame)
        if len(result["class_ids"]) > 0:
            # all detections should be person (class 0)
            assert (result["class_ids"] == 0).all()

    def test_detect_empty_frame(self, detector):
        # solid black frame — should detect nothing or very few
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(black)
        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
