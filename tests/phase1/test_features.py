"""
Feature extraction and validation tests — Phase 1 Plan 2.
Tests for load_and_validate_features() shape, dtype, and dimension assertions.
"""
import numpy as np
import pytest
from src.features.extract import load_and_validate_features


def test_load_and_validate_features_pca512(tmp_path):
    """Valid PCA-512 features (5400, 512) should load without error."""
    arr = np.zeros((5400, 512), dtype=np.float32)
    fpath = tmp_path / "features.npy"
    np.save(fpath, arr)
    result = load_and_validate_features(str(fpath), expected_dim=512)
    assert result.shape == (5400, 512)
    assert result.dtype == np.float32


def test_load_and_validate_features_wrong_dim(tmp_path):
    """Features with wrong dimension (2048 instead of 512) must raise AssertionError."""
    arr = np.zeros((5400, 2048), dtype=np.float32)
    fpath = tmp_path / "features_2048.npy"
    np.save(fpath, arr)
    with pytest.raises(AssertionError):
        load_and_validate_features(str(fpath), expected_dim=512)


def test_load_and_validate_features_too_few_frames(tmp_path):
    """Features with too few frames (<= 4500) must raise AssertionError."""
    arr = np.zeros((100, 512), dtype=np.float32)
    fpath = tmp_path / "features_short.npy"
    np.save(fpath, arr)
    with pytest.raises(AssertionError):
        load_and_validate_features(str(fpath), expected_dim=512)


def test_load_and_validate_features_custom_resnet(tmp_path):
    """Custom ResNet-50 features (5400, 2048) should be valid with expected_dim=2048."""
    arr = np.zeros((5400, 2048), dtype=np.float32)
    fpath = tmp_path / "features_resnet.npy"
    np.save(fpath, arr)
    result = load_and_validate_features(str(fpath), expected_dim=2048)
    assert result.shape == (5400, 2048)
