"""Tests for ByteTrack tracker with camera cut detection."""
import numpy as np
import pytest

from src.tracking.tracker import PlayerTracker, CameraCutDetector


class TestCameraCutDetector:
    def test_no_cut_on_first_frame(self):
        det = CameraCutDetector(threshold=0.4)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert not det.is_cut(frame)

    def test_no_cut_similar_frames(self):
        det = CameraCutDetector(threshold=0.4)
        frame1 = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[:10, :10] = 200  # small change
        det.is_cut(frame1)
        assert not det.is_cut(frame2)

    def test_detects_cut(self):
        det = CameraCutDetector(threshold=0.3)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        det.is_cut(frame1)
        assert det.is_cut(frame2)


class TestPlayerTracker:
    def test_update_returns_correct_keys(self, dummy_frame, dummy_detections):
        tracker = PlayerTracker()
        result = tracker.update(dummy_frame, dummy_detections)
        assert "boxes" in result
        assert "confidences" in result
        assert "tracker_ids" in result
        assert "is_camera_cut" in result

    def test_tracker_assigns_ids(self, dummy_frame, dummy_detections):
        tracker = PlayerTracker()
        result = tracker.update(dummy_frame, dummy_detections)
        if len(result["tracker_ids"]) > 0:
            # IDs should be positive integers
            assert (result["tracker_ids"] > 0).all()

    def test_consistent_ids_across_frames(self, dummy_detections):
        tracker = PlayerTracker()
        frame = np.random.randint(100, 150, (720, 1280, 3), dtype=np.uint8)
        r1 = tracker.update(frame, dummy_detections)
        # slightly shift boxes
        shifted = dummy_detections.copy()
        shifted["boxes"] = shifted["boxes"] + 2  # 2px shift
        frame2 = frame.copy()
        frame2[:5, :5] = frame2[:5, :5] + 1  # tiny change
        r2 = tracker.update(frame2, shifted)
        if len(r1["tracker_ids"]) > 0 and len(r2["tracker_ids"]) > 0:
            # some IDs should persist
            shared = set(r1["tracker_ids"]) & set(r2["tracker_ids"])
            assert len(shared) > 0

    def test_reset_on_camera_cut(self, dummy_detections):
        tracker = PlayerTracker(cut_threshold=0.3)
        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame2 = np.full((720, 1280, 3), 255, dtype=np.uint8)
        tracker.update(frame1, dummy_detections)
        result = tracker.update(frame2, dummy_detections)
        assert result["is_camera_cut"]

    def test_handles_empty_detections(self, dummy_frame):
        tracker = PlayerTracker()
        empty = {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "confidences": np.empty(0, dtype=np.float32),
            "class_ids": np.empty(0, dtype=np.int32),
        }
        result = tracker.update(dummy_frame, empty)
        assert len(result["boxes"]) == 0
