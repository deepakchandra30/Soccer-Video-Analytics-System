"""Tests for TSM temporal shift model and chunked dataset."""
import numpy as np
import pytest
import torch

from src.models.temporal.tsm import TemporalShift, TSMSpottingHead
from src.models.temporal.losses import get_class_weights
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from config.tsm_config import TSM_CONFIG


class TestTemporalShift:
    def test_shape_preserved(self, dummy_features):
        shift = TemporalShift(n_div=8)
        out = shift(dummy_features)
        assert out.shape == dummy_features.shape

    def test_channels_shifted(self, dummy_features):
        shift = TemporalShift(n_div=8)
        out = shift(dummy_features)
        fold = 512 // 8  # 64 channels
        # first fold should differ (shifted forward in time)
        assert not torch.allclose(out[:, :, :fold], dummy_features[:, :, :fold])


class TestTSMSpottingHead:
    def test_forward_shape_512(self, dummy_features):
        model = TSMSpottingHead(feat_dim=512, num_classes=17)
        out = model(dummy_features)
        assert out.shape == (2, 40, 18)  # 17 classes + 1 background

    def test_forward_shape_2048(self, dummy_features_2048):
        model = TSMSpottingHead(feat_dim=2048, num_classes=17)
        out = model(dummy_features_2048)
        assert out.shape == (2, 40, 18)

    def test_feature_dim_mismatch(self):
        model = TSMSpottingHead(feat_dim=512)
        bad_input = torch.randn(1, 10, 256)
        with pytest.raises(Exception):
            model(bad_input)


class TestChunkedDataset:
    def test_item_shape(self, mock_match_data):
        ds = ChunkedSoccerNetDataset(mock_match_data, chunk_size=40, feat_dim=512)
        item = ds[0]
        assert item["features"].shape == (40, 512)
        assert item["targets"].shape == (40,)

    def test_feat_dim_2048(self, mock_match_data_2048):
        ds = ChunkedSoccerNetDataset(mock_match_data_2048, chunk_size=40, feat_dim=2048)
        item = ds[0]
        assert item["features"].shape == (40, 2048)

    def test_targets_range(self, mock_match_data):
        ds = ChunkedSoccerNetDataset(mock_match_data, chunk_size=40, feat_dim=512)
        item = ds[0]
        # targets should be in [0, 17]: 0=background, 1-17=event classes
        assert item["targets"].min() >= 0
        assert item["targets"].max() <= 17


class TestLosses:
    def test_class_weighted_loss(self):
        weights = get_class_weights(num_classes=17, bg_weight=0.05)
        assert weights.shape == (18,)
        assert weights[0].item() == pytest.approx(0.05)
        assert weights[1].item() == pytest.approx(1.0)
        # verify the loss computes without error
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        logits = torch.randn(10, 18)
        targets = torch.randint(0, 18, (10,))
        loss = criterion(logits, targets)
        assert loss.item() > 0


class TestConfig:
    def test_config_values(self):
        required_keys = [
            "feat_dim", "num_classes", "hidden_dim", "n_shifts", "n_div",
            "chunk_size", "event_ratio", "batch_size", "lr", "weight_decay",
            "epochs", "patience", "bg_weight", "nms_window",
            "confidence_threshold", "framerate", "window_size", "stride",
        ]
        for key in required_keys:
            assert key in TSM_CONFIG, f"Missing config key: {key}"
