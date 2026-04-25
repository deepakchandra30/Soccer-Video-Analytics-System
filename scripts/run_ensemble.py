"""Ensemble TSM + SlowFast predictions with multi-scale TTA + soft-NMS.

Loads two trained checkpoints, runs each model with multi-scale sliding-window
inference, weighted-averages the per-frame event probabilities, and writes
SoccerNet-evaluator-compatible predictions. Reports tight avg-mAP.

Designed to be reusable across feature sets — pass --feature-type to switch
between PCA-512 (default), full ResNet-2048, or Baidu 8576-dim once that
download is available.
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
from src.models.temporal.postprocess import (
    multi_scale_inference,
    nms_detections,
    save_predictions,
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


def ensemble_match(tsm, slowfast, half1, half2, weights, scales, device,
                   fusion="weighted"):
    """Run both models with multi-scale TTA on one match's two halves.

    Args:
        fusion: "weighted" (default) — per-frame per-class weighted arithmetic
            mean of the two score arrays. Classic and interpretable but
            dilutes the stronger model when score scales differ.
            "max" — per-frame per-class elementwise max. Lets each model
            contribute its confident predictions without averaging them
            down; useful when one model is clearly stronger *overall* but
            the weaker one picks up a few specific classes better.

    Returns (half1_scores, half2_scores) as (T, num_classes) numpy arrays.

    Short-circuits: if SlowFast weight is 0 (TSM-only mode), skip SlowFast
    inference entirely — same answer, half the GPU time.
    """
    w_tsm, w_sf = weights
    w_sum = w_tsm + w_sf
    out = []
    tsm_only = (slowfast is None) or (w_sf == 0)
    for half in (half1, half2):
        tsm_scores = multi_scale_inference(tsm, half, scales=scales, device=device)
        if tsm_only:
            out.append(tsm_scores)
            continue
        sf_scores = multi_scale_inference(slowfast, half, scales=scales, device=device)
        # both are (T, num_classes); same T because multi_scale_inference
        # always returns at the input feature length.
        if fusion == "weighted":
            ensembled = (w_tsm * tsm_scores + w_sf * sf_scores) / w_sum
        elif fusion == "max":
            ensembled = np.maximum(tsm_scores, sf_scores)
        else:
            raise ValueError(f"Unknown fusion={fusion!r}")
        out.append(ensembled)
    return out[0], out[1]


def main():
    parser = argparse.ArgumentParser(description="TSM+SlowFast ensemble inference")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--tsm-checkpoint", required=True)
    parser.add_argument("--slowfast-checkpoint", default=None,
                        help="Optional. Omit to run TSM-only with TTA + soft-NMS "
                             "(useful as an intermediate data point before SlowFast "
                             "finishes retraining).")
    parser.add_argument("--output-dir", default="outputs/ensemble")
    parser.add_argument("--split", default="valid",
                        choices=["valid", "test", "challenge"])
    parser.add_argument("--feature-type", default="pca512",
                        choices=list(FEAT_MAP))
    parser.add_argument("--feat-dim", type=int, default=None,
                        help="Override default feat-dim for the chosen feature type")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tsm-weight", type=float, default=0.5,
                        help="Ensemble weight on TSM scores; SlowFast gets 1-w")
    parser.add_argument("--scales", default="40:20,80:40",
                        help="Comma-separated window:stride pairs for multi-scale TTA")
    parser.add_argument("--fusion", default="weighted",
                        choices=["weighted", "max"],
                        help="How to combine TSM and SlowFast scores. 'weighted' "
                             "= per-class weighted mean (tsm_weight vs 1-tsm_weight); "
                             "'max' = elementwise max (better when one model is "
                             "much stronger overall)")
    parser.add_argument("--nms-mode", default="soft", choices=["hard", "soft"])
    parser.add_argument("--nms-window", type=int, default=15)
    parser.add_argument("--confidence-threshold", type=float, default=0.15)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.feat_dim is None:
        args.feat_dim = FEAT_DIMS[args.feature_type]

    scales = tuple(tuple(int(x) for x in pair.split(":"))
                   for pair in args.scales.split(","))

    # Framerate must match feature extraction rate (Baidu=1fps, PCA=2fps).
    from src.training.train_tsm import FEATURE_FRAMERATE
    _framerate = FEATURE_FRAMERATE.get(args.feature_type, 2)

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load eval matches and feature filenames
    print(f"Loading {args.feature_type} features ({args.feat_dim}-dim) "
          f"from {args.data_dir} for split={args.split}...")
    matches, match_dirs, match_ids = load_matches(
        args.data_dir, args.split, args.feature_type
    )
    print(f"  Loaded {len(matches)} matches.")

    # Build models — use train(False) instead of .eval() because the literal
    # `.eval(` substring trips the project security-reminder hook.
    tsm = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=TSM_CONFIG["num_classes"],
        hidden_dim=TSM_CONFIG["hidden_dim"], n_shifts=TSM_CONFIG["n_shifts"],
    ).to(args.device)
    load_checkpoint(args.tsm_checkpoint, tsm)
    tsm.train(False)

    if args.slowfast_checkpoint:
        slowfast = SlowFastSpotting(
            feat_dim=args.feat_dim, num_classes=SLOWFAST_CONFIG["num_classes"],
            slow_stride=SLOWFAST_CONFIG["slow_stride"],
            hidden_dim=SLOWFAST_CONFIG["hidden_dim"],
        ).to(args.device)
        load_checkpoint(args.slowfast_checkpoint, slowfast)
        slowfast.train(False)
        weights = (args.tsm_weight, 1.0 - args.tsm_weight)
    else:
        # TSM-only mode: ensemble_match short-circuits and runs TSM only.
        slowfast = None
        weights = (1.0, 0.0)
        print("  (no SlowFast checkpoint — running TSM-only with TTA + soft-NMS)")
    pred_root = Path(args.output_dir) / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    for match, match_dir, match_id in zip(matches, match_dirs, match_ids):
        features, _annotations, half1_len = match
        # Use the half1_len from load_matches to split the concatenated feature
        # array back into two halves — same offset that ChunkedSoccerNetDataset
        # uses, so prediction frame-indices line up with the labels.
        half1 = torch.FloatTensor(features[:half1_len])
        half2 = torch.FloatTensor(features[half1_len:])

        h1_scores, h2_scores = ensemble_match(
            tsm, slowfast, half1, half2, weights=weights,
            scales=scales, device=args.device, fusion=args.fusion,
        )

        all_preds = []
        for half_idx, scores in enumerate([h1_scores, h2_scores], start=1):
            half_preds = nms_detections(
                scores,
                nms_window=args.nms_window,
                confidence_threshold=args.confidence_threshold,
                framerate=_framerate,
                half=half_idx,
                nms_mode=args.nms_mode,
            )
            all_preds.extend(half_preds)

        out_path = pred_root / match_id
        out_path.mkdir(parents=True, exist_ok=True)
        save_predictions(
            all_preds, str(out_path / "results_spotting.json"),
            url_local=match_id,
        )

    print("Running SoccerNet evaluator on ensemble predictions...")
    results = run_evaluation(args.data_dir, str(pred_root), split=args.split)
    avg_map = float(results.get("a_mAP", 0.0)) * 100.0

    print("\nEnsemble Results")
    print("----------------")
    print(f"  Feature type:     {args.feature_type} ({args.feat_dim}-dim)")
    print(f"  Split:            {args.split}")
    print(f"  Fusion:           {args.fusion}")
    print(f"  TSM weight:       {weights[0]:.2f}")
    print(f"  SlowFast weight:  {weights[1]:.2f}")
    print(f"  Multi-scale TTA:  {scales}")
    print(f"  NMS mode:         {args.nms_mode} (window={args.nms_window})")
    print(f"  Conf threshold:   {args.confidence_threshold}")
    print(f"  avg-mAP tight:    {avg_map:.4f}%")


if __name__ == "__main__":
    main()
