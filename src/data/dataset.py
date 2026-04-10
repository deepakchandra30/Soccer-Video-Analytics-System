"""PyTorch Dataset for SoccerNet match features and labels."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SoccerNetDataset(Dataset):
    """Loads pre-extracted SoccerNet features (.npy) and v3 annotations.

    Expects directory layout:
        data/{match_id}/1_ResNET_TF2_PCA512.npy  (or 1_resnet50_2048.npy)
        data/{match_id}/2_ResNET_TF2_PCA512.npy  (or 2_resnet50_2048.npy)
        data/{match_id}/Labels-v3.json
    """

    FEATURE_FILES = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_resnet50_2048.npy", "2_resnet50_2048.npy"),
    }
    FEATURE_DIMS = {"pca512": 512, "resnet50": 2048}

    def __init__(self, data_dir, split, feature_type="pca512"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.feature_dim = self.FEATURE_DIMS[feature_type]

        # only keep directories that actually have labels
        self.matches = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / "Labels-v3.json").exists()
        ])

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match_dir = self.matches[idx]
        f1, f2 = self.FEATURE_FILES[self.feature_type]

        half1 = np.load(match_dir / f1)
        half2 = np.load(match_dir / f2)
        features = np.concatenate([half1, half2], axis=0)

        assert features.shape[1] == self.feature_dim, (
            f"Expected dim {self.feature_dim}, got {features.shape[1]} in {match_dir}"
        )

        # NB: v3 uses "actions" key, NOT "annotations" — confirmed from actual data
        with open(match_dir / "Labels-v3.json") as f:
            labels = json.load(f)

        return {
            "features": torch.FloatTensor(features),
            "annotations": labels.get("actions", []),
            "match_dir": str(match_dir),
        }
