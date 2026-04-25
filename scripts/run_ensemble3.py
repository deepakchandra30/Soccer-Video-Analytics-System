#!/usr/bin/env python
"""3-way ensemble: TSM + SlowFast + NetVLAD++ (additive — does not
modify scripts/run_ensemble.py).

Drop-in structural cousin of scripts/run_ensemble.py with one extra
checkpoint. Weighted-average fusion; each model's contribution is
controlled by --tsm-weight, --slowfast-weight, --netvlad-weight.
Weights are auto-normalised to sum to 1.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds import set_seeds
from config.tsm_config import TSM_CONFIG
from config.slowfast_config import SLOWFAST_CONFIG
from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.slowfast import SlowFastSpotting
from src.models.temporal.netvlad import NetVLADSpottingHead
from src.models.temporal.postprocess import (
    multi_scale_inference, nms_detections, save_predictions,
)
from src.training.trainer import load_checkpoint
from src.training.train_tsm import load_matches
from src.evaluation.evaluate import run_evaluation


FEAT_MAP = {
    "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
    "resnet50": ("1_ResNET_TF2.npy", "2_ResNET_TF2.npy"),
    "baidu": ("1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"),
}
FEAT_DIMS = {"pca512": 512, "resnet50": 2048, "baidu": 8576}


def main():
    parser = argparse.ArgumentParser(description="3-way TSM+SF+NetVLAD ensemble")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--tsm-checkpoint", required=True)
    parser.add_argument("--slowfast-checkpoint", required=True)
    parser.add_argument("--netvlad-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/ensemble3")
    parser.add_argument("--split", default="test",
                        choices=["valid", "test", "challenge"])
    parser.add_argument("--feature-type", default="pca512",
                        choices=list(FEAT_MAP))
    parser.add_argument("--feat-dim", type=int, default=None)
    parser.add_argument("--device", default=None)
    # Fusion weights; auto-normalised, so absolute scale doesn't matter.
    # The defaults match the best 2-way sweep (SF-dominant) with a small
    # NetVLAD contribution to start from a known-good baseline.
    parser.add_argument("--tsm-weight", type=float, default=0.15)
    parser.add_argument("--slowfast-weight", type=float, default=0.60)
    parser.add_argument("--netvlad-weight", type=float, default=0.25)
    parser.add_argument("--scales", default="40:20,80:40")
    parser.add_argument("--nms-mode", default="hard", choices=["hard", "soft"])
    parser.add_argument("--nms-window", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--netvlad-hidden-dim", type=int, default=256)
    parser.add_argument("--netvlad-num-clusters", type=int, default=32)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.feat_dim is None:
        args.feat_dim = FEAT_DIMS[args.feature_type]

    scales = tuple(tuple(int(x) for x in p.split(":"))
                   for p in args.scales.split(","))

    # Framerate must match the feature extraction rate: Baidu=1fps, PCA=2fps.
    from src.training.train_tsm import FEATURE_FRAMERATE
    _framerate = FEATURE_FRAMERATE.get(args.feature_type, 2)

    # Normalise weights so the user can pass any scale.
    w = np.array([args.tsm_weight, args.slowfast_weight, args.netvlad_weight],
                 dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"Weights must be non-negative and sum > 0, got {w}")
    w /= w.sum()
    w_tsm, w_sf, w_nv = float(w[0]), float(w[1]), float(w[2])

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.feature_type} features ({args.feat_dim}-dim) "
          f"from {args.data_dir} for split={args.split}...")
    matches, match_dirs, match_ids = load_matches(
        args.data_dir, args.split, args.feature_type,
    )
    print(f"  Loaded {len(matches)} matches.")

    # Build all three models.
    tsm = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=TSM_CONFIG["hidden_dim"], n_shifts=TSM_CONFIG["n_shifts"],
    ).to(args.device)
    load_checkpoint(args.tsm_checkpoint, tsm)
    tsm.train(False)

    slowfast = SlowFastSpotting(
        feat_dim=args.feat_dim, num_classes=SLOWFAST_CONFIG["num_classes"],
        slow_stride=SLOWFAST_CONFIG["slow_stride"],
        hidden_dim=SLOWFAST_CONFIG["hidden_dim"],
    ).to(args.device)
    load_checkpoint(args.slowfast_checkpoint, slowfast)
    slowfast.train(False)

    netvlad = NetVLADSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=args.netvlad_hidden_dim,
        num_clusters=args.netvlad_num_clusters,
    ).to(args.device)
    load_checkpoint(args.netvlad_checkpoint, netvlad)
    netvlad.train(False)

    pred_root = Path(args.output_dir) / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    for match, match_dir, match_id in zip(matches, match_dirs, match_ids):
        features, _annotations, half1_len = match
        half1 = torch.FloatTensor(features[:half1_len])
        half2 = torch.FloatTensor(features[half1_len:])

        h1_preds = []
        h2_preds = []
        for half_idx, half in enumerate([half1, half2], start=1):
            tsm_s = multi_scale_inference(tsm, half, scales=scales,
                                           device=args.device)
            sf_s = multi_scale_inference(slowfast, half, scales=scales,
                                          device=args.device)
            nv_s = multi_scale_inference(netvlad, half, scales=scales,
                                          device=args.device)
            ens = w_tsm * tsm_s + w_sf * sf_s + w_nv * nv_s
            preds = nms_detections(
                ens,
                nms_window=args.nms_window,
                confidence_threshold=args.confidence_threshold,
                framerate=_framerate,
                half=half_idx,
                nms_mode=args.nms_mode,
            )
            (h1_preds if half_idx == 1 else h2_preds).extend(preds)

        out_dir = pred_root / match_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(h1_preds + h2_preds,
                         str(out_dir / "results_spotting.json"),
                         url_local=match_id)

    print("Running SoccerNet evaluator on 3-way ensemble predictions...")
    results = run_evaluation(args.data_dir, str(pred_root), split=args.split)
    avg_map = float(results.get("a_mAP", 0.0)) * 100.0

    print("\n3-Way Ensemble Results")
    print("----------------------")
    print(f"  Feature type:     {args.feature_type} ({args.feat_dim}-dim)")
    print(f"  Split:            {args.split}")
    print(f"  Weights (norm):   TSM={w_tsm:.3f}  SF={w_sf:.3f}  NetVLAD={w_nv:.3f}")
    print(f"  Multi-scale TTA:  {scales}")
    print(f"  NMS mode:         {args.nms_mode} (window={args.nms_window})")
    print(f"  Conf threshold:   {args.confidence_threshold}")
    print(f"  avg-mAP tight:    {avg_map:.4f}%")


if __name__ == "__main__":
    main()
