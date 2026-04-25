"""Shared fixtures for phase 2 temporal model tests."""
import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_features():
    """Batch of random 512-dim features, shape (2, 40, 512)."""
    return torch.randn(2, 40, 512)


@pytest.fixture
def dummy_features_2048():
    """Batch of random 2048-dim features, shape (2, 40, 2048)."""
    return torch.randn(2, 40, 2048)


@pytest.fixture
def dummy_annotations():
    """Mock action annotations in Labels-v3 format."""
    return [
        {"gameTime": "1 - 05:00", "label": "Goal"},
        {"gameTime": "1 - 15:30", "label": "Foul"},
        {"gameTime": "1 - 30:00", "label": "Corner"},
    ]


@pytest.fixture
def mock_match_data(dummy_annotations):
    """List of (features_np, annotations) tuples for ChunkedSoccerNetDataset."""
    feats = np.random.randn(5400, 512).astype(np.float32)
    return [(feats, dummy_annotations)]


@pytest.fixture
def mock_match_data_2048(dummy_annotations):
    """Same as mock_match_data but with 2048-dim features."""
    feats = np.random.randn(5400, 2048).astype(np.float32)
    return [(feats, dummy_annotations)]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
