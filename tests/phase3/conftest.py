"""Shared fixtures for phase 3 tracking tests."""
import numpy as np
import pytest


@pytest.fixture
def dummy_frame():
    """Random 720p BGR frame."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def dummy_detections():
    """Mock detection output with 3 players."""
    return {
        "boxes": np.array([
            [100, 200, 150, 400],
            [300, 180, 360, 410],
            [600, 190, 660, 405],
        ], dtype=np.float32),
        "confidences": np.array([0.92, 0.87, 0.95], dtype=np.float32),
        "class_ids": np.array([0, 0, 0], dtype=np.int32),
    }


@pytest.fixture
def pitch_image_correspondences():
    """4 matching pitch/image keypoint pairs for homography estimation."""
    pitch_pts = np.array([
        [-52.5, -34.0],  # top-left corner
        [52.5, -34.0],   # top-right corner
        [52.5, 34.0],    # bottom-right corner
        [-52.5, 34.0],   # bottom-left corner
    ], dtype=np.float64)
    image_pts = np.array([
        [100, 50],
        [1180, 80],
        [1100, 650],
        [180, 620],
    ], dtype=np.float64)
    return image_pts, pitch_pts
