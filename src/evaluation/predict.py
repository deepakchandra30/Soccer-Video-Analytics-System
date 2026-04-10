"""Generate results_spotting.json predictions from a trained model."""
import os
from pathlib import Path

import numpy as np
import torch

from src.models.temporal.postprocess import (
    nms_detections,
    save_predictions,
    sliding_window_inference,
)


def generate_predictions(model, match_dirs, match_ids, output_dir,
                         feature_files=("1_ResNET_TF2_PCA512.npy",
                                        "2_ResNET_TF2_PCA512.npy"),
                         window_size=40, stride=20, nms_window=30,
                         confidence_threshold=0.2, framerate=2, device="cpu"):
    """Run inference on all matches and write results_spotting.json per match.

    Args:
        model: trained temporal model
        match_dirs: list of paths to match data directories
        match_ids: list of match identifiers (must match SoccerNet naming)
        output_dir: root directory for prediction output
        feature_files: tuple of (half1_file, half2_file) names
        window_size: sliding window size in frames
        stride: sliding window stride
        nms_window: NMS suppression window in frames
        confidence_threshold: minimum detection confidence
        framerate: feature fps
        device: torch device
    """
    model.eval()
    f1_name, f2_name = feature_files

    for match_dir, match_id in zip(match_dirs, match_ids):
        match_dir = Path(match_dir)

        all_preds = []
        for half_idx, fname in enumerate([f1_name, f2_name], start=1):
            fpath = match_dir / fname
            features = np.load(str(fpath))
            features_t = torch.FloatTensor(features)

            scores = sliding_window_inference(
                model, features_t,
                window_size=window_size,
                stride=stride,
                device=device,
            )

            half_preds = nms_detections(
                scores,
                nms_window=nms_window,
                confidence_threshold=confidence_threshold,
                framerate=framerate,
                half=half_idx,
            )
            all_preds.extend(half_preds)

        # write predictions
        out_path = Path(output_dir) / match_id
        out_path.mkdir(parents=True, exist_ok=True)
        save_predictions(all_preds, str(out_path / "results_spotting.json"))

    return str(output_dir)
