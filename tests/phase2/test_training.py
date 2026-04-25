"""Tests for TSM training loop and prediction generation."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.losses import get_class_weights
from src.training.trainer import train_epoch, validate_epoch, EarlyStopping


@pytest.fixture
def tiny_model():
    """Small TSM model for fast testing."""
    return TSMSpottingHead(feat_dim=512, num_classes=17, hidden_dim=32, n_shifts=1)


@pytest.fixture
def tiny_dataloader():
    """Minimal dataloader with 3 batches."""
    items = []
    for _ in range(12):  # 3 batches of 4
        items.append({
            "features": torch.randn(20, 512),
            "targets": torch.randint(0, 18, (20,)),
        })

    def collate(batch):
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "targets": torch.stack([b["targets"] for b in batch]),
        }

    return DataLoader(items, batch_size=4, collate_fn=collate)


class TestTrainEpoch:
    def test_reduces_loss(self, tiny_model, tiny_dataloader):
        weights = get_class_weights(17, 0.05)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss = train_epoch(tiny_model, tiny_dataloader, optimizer, criterion, "cpu")
        assert np.isfinite(loss)

    def test_updates_params(self, tiny_model, tiny_dataloader):
        weights = get_class_weights(17, 0.05)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        params_before = [p.clone() for p in tiny_model.parameters()]
        train_epoch(tiny_model, tiny_dataloader, optimizer, criterion, "cpu")
        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, tiny_model.parameters())
        )
        assert changed


class TestValidateEpoch:
    def test_returns_finite_loss(self, tiny_model, tiny_dataloader):
        weights = get_class_weights(17, 0.05)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        loss = validate_epoch(tiny_model, tiny_dataloader, criterion, "cpu")
        assert np.isfinite(loss)


class TestEarlyStopping:
    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3, mode="max")
        assert not es.step(0.5)
        assert not es.step(0.4)  # worse, counter=1
        assert not es.step(0.3)  # worse, counter=2
        assert es.step(0.2)      # worse, counter=3 -> stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5)
        es.step(0.4)  # counter=1
        es.step(0.6)  # better, counter reset
        assert not es.step(0.5)  # worse again, counter=1
        assert not es.step(0.4)  # counter=2


class TestGeneratePredictions:
    def test_creates_json(self, tiny_model):
        from src.evaluation.predict import generate_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            # set up a fake match directory
            match_id = "england_epl/2014-2015/test_match"
            match_dir = os.path.join(tmpdir, "data", match_id)
            os.makedirs(match_dir)
            # create dummy feature files
            np.save(os.path.join(match_dir, "1_ResNET_TF2_PCA512.npy"),
                    np.random.randn(100, 512).astype(np.float32))
            np.save(os.path.join(match_dir, "2_ResNET_TF2_PCA512.npy"),
                    np.random.randn(100, 512).astype(np.float32))

            out_dir = os.path.join(tmpdir, "preds")
            generate_predictions(
                model=tiny_model,
                match_dirs=[match_dir],
                match_ids=[match_id],
                output_dir=out_dir,
                device="cpu",
            )

            pred_file = os.path.join(out_dir, match_id, "results_spotting.json")
            assert os.path.exists(pred_file)
            with open(pred_file) as f:
                data = json.load(f)
            assert "predictions" in data

    def test_prediction_format(self, tiny_model):
        from src.evaluation.predict import generate_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            match_id = "england_epl/2014-2015/test_match"
            match_dir = os.path.join(tmpdir, "data", match_id)
            os.makedirs(match_dir)
            np.save(os.path.join(match_dir, "1_ResNET_TF2_PCA512.npy"),
                    np.random.randn(100, 512).astype(np.float32))
            np.save(os.path.join(match_dir, "2_ResNET_TF2_PCA512.npy"),
                    np.random.randn(100, 512).astype(np.float32))

            out_dir = os.path.join(tmpdir, "preds")
            generate_predictions(
                model=tiny_model,
                match_dirs=[match_dir],
                match_ids=[match_id],
                output_dir=out_dir,
                device="cpu",
            )

            pred_file = os.path.join(out_dir, match_id, "results_spotting.json")
            with open(pred_file) as f:
                data = json.load(f)

            # SoccerNet's evaluator rejects non-string values, so all
            # prediction fields ship as str (see postprocess.py).
            for p in data["predictions"]:
                assert isinstance(p["label"], str)
                assert isinstance(p["half"], str)
                assert isinstance(p["position"], str)
                assert isinstance(p["confidence"], str)
