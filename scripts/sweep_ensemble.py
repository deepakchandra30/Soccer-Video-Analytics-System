#!/usr/bin/env python
"""Caching grid-search over ensemble post-processing hyperparameters.

Why this script exists:
    ``scripts/run_ensemble.py`` runs model inference on every call, so
    sweeping N post-processing configs costs N * ~5 min. Inference output
    is deterministic given (checkpoint, features, scales) — the rest of
    the pipeline is just weighting and NMS. So: run inference **once**,
    cache per-match (TSM, SlowFast) score arrays to disk, and iterate
    cheaply over post-processing.

    Cost: ~5 min one-time inference + ~15-30 sec per swept config.

Design (additive — no edits to existing files):
    - Imports ``multi_scale_inference`` and ``nms_detections`` from
      ``src.models.temporal.postprocess`` (unchanged).
    - Reuses ``load_matches`` from ``src.training.train_tsm`` (unchanged).
    - Reuses ``run_evaluation`` from ``src.evaluation.evaluate`` (unchanged).
    - Only adds ``scripts/sweep_ensemble.py`` (this file). No new
      dependencies, no modifications to models, configs, or trainers.

Output:
    - Per-match cached scores in ``outputs/sweep_cache/{match_id}/*.npy``.
    - Sweep CSV log at ``outputs/sweep_cache/results.csv``.
    - Best-config predictions in ``outputs/sweep_best/predictions/``.
    - Prints best config + mAP so you can copy-paste it into
      ``scripts/run_full_pipeline.py`` as the new HEADLINE_* constants.
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import time
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
    "pca512": 512,
    "resnet50": 2048,
    "baidu": 8576,
}


def parse_scales(s):
    return tuple(tuple(int(x) for x in pair.split(":")) for pair in s.split(","))


def compute_and_cache_scores(tsm, slowfast, matches, match_ids, scales, device,
                             cache_dir):
    """Run each model once per match and cache per-half score arrays to disk.

    Cache layout:
        cache_dir/{match_id}/tsm_h{1,2}.npy       - (T, C) arrays
        cache_dir/{match_id}/slowfast_h{1,2}.npy  - (T, C) arrays
        cache_dir/{match_id}/half_lens.json       - {"h1": T1, "h2": T2}

    Skips inference for matches whose cache already exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (match, match_id) in enumerate(zip(matches, match_ids)):
        match_cache = cache_dir / match_id
        tsm_h1_p = match_cache / "tsm_h1.npy"
        tsm_h2_p = match_cache / "tsm_h2.npy"
        sf_h1_p = match_cache / "slowfast_h1.npy"
        sf_h2_p = match_cache / "slowfast_h2.npy"
        lens_p = match_cache / "half_lens.json"
        if all(p.exists() for p in (tsm_h1_p, tsm_h2_p, sf_h1_p, sf_h2_p, lens_p)):
            continue
        match_cache.mkdir(parents=True, exist_ok=True)
        features, _annotations, half1_len = match
        half1 = torch.FloatTensor(features[:half1_len])
        half2 = torch.FloatTensor(features[half1_len:])

        tsm_h1 = multi_scale_inference(tsm, half1, scales=scales, device=device)
        tsm_h2 = multi_scale_inference(tsm, half2, scales=scales, device=device)
        sf_h1 = multi_scale_inference(slowfast, half1, scales=scales, device=device)
        sf_h2 = multi_scale_inference(slowfast, half2, scales=scales, device=device)

        np.save(tsm_h1_p, tsm_h1)
        np.save(tsm_h2_p, tsm_h2)
        np.save(sf_h1_p, sf_h1)
        np.save(sf_h2_p, sf_h2)
        with open(lens_p, "w") as f:
            json.dump({"h1": int(half1.shape[0]), "h2": int(half2.shape[0])}, f)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(matches) - i - 1) / (i + 1)
            print(f"  [cache] {i+1}/{len(matches)} matches done, "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    print(f"  [cache] all {len(matches)} matches cached in "
          f"{time.time()-t0:.0f}s.", flush=True)


def load_cached_scores(match_ids, cache_dir):
    """Return list[(tsm_h1, tsm_h2, sf_h1, sf_h2)] parallel to match_ids."""
    cache_dir = Path(cache_dir)
    out = []
    for mid in match_ids:
        d = cache_dir / mid
        out.append((
            np.load(d / "tsm_h1.npy"),
            np.load(d / "tsm_h2.npy"),
            np.load(d / "slowfast_h1.npy"),
            np.load(d / "slowfast_h2.npy"),
        ))
    return out


def eval_one_config(cached_scores, match_ids, match_dirs, data_dir, split,
                    framerate, tsm_weight, conf, nms_window, nms_mode,
                    pred_root):
    """Apply one post-processing config to every match's cached scores,
    write a predictions/ tree, run the official evaluator, return mAP.

    Uses weighted-mean fusion: (w_tsm * tsm + w_sf * sf) / (w_tsm + w_sf).
    """
    pred_root = Path(pred_root)
    # Clean previous predictions so stale matches don't poison the eval.
    # Match ids are nested paths like league/season/match, so rmtree is
    # the only correct cleanup — the naive single-level iterdir/unlink
    # approach crashes on the intermediate season directories.
    if pred_root.exists():
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)

    w_sf = 1.0 - tsm_weight
    w_sum = tsm_weight + w_sf  # normalise in case user passes >1

    for (tsm_h1, tsm_h2, sf_h1, sf_h2), mid in zip(cached_scores, match_ids):
        ens_h1 = (tsm_weight * tsm_h1 + w_sf * sf_h1) / w_sum
        ens_h2 = (tsm_weight * tsm_h2 + w_sf * sf_h2) / w_sum
        all_preds = []
        for half_idx, scores in enumerate([ens_h1, ens_h2], start=1):
            half_preds = nms_detections(
                scores,
                nms_window=nms_window,
                confidence_threshold=conf,
                framerate=framerate,
                half=half_idx,
                nms_mode=nms_mode,
            )
            all_preds.extend(half_preds)
        out_dir = pred_root / mid
        out_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(
            all_preds, str(out_dir / "results_spotting.json"), url_local=mid,
        )
    results = run_evaluation(data_dir, str(pred_root), split=split)
    return float(results.get("a_mAP", 0.0)) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Ensemble hyperparameter sweep")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--tsm-checkpoint", default="outputs/tsm_fixed/best.pt")
    parser.add_argument("--slowfast-checkpoint",
                        default="outputs/slowfast_fixed/best.pt")
    parser.add_argument("--split", default="valid",
                        choices=["valid", "test", "challenge"])
    parser.add_argument("--feature-type", default="pca512",
                        choices=list(FEAT_MAP))
    parser.add_argument("--feat-dim", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scales", default="40:20,80:40")
    parser.add_argument("--cache-dir", default="outputs/sweep_cache")
    parser.add_argument("--best-output-dir", default="outputs/sweep_best")
    # Grid (defaults: small Phase-A grid around the known 42.37% setting).
    parser.add_argument("--tsm-weights", default="0.1,0.15,0.2,0.25,0.3,0.35")
    parser.add_argument("--conf-thresholds", default="0.1,0.15,0.2,0.25")
    parser.add_argument("--nms-windows", default="7,10,15,20")
    parser.add_argument("--nms-modes", default="hard,soft")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max configs to sweep (safety cap).")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.feat_dim is None:
        args.feat_dim = FEAT_MAP[args.feature_type]

    scales = parse_scales(args.scales)
    set_seeds(42)

    # Parse grids.
    tsm_weights = [float(x) for x in args.tsm_weights.split(",")]
    confs = [float(x) for x in args.conf_thresholds.split(",")]
    nms_windows = [int(x) for x in args.nms_windows.split(",")]
    nms_modes = [x.strip() for x in args.nms_modes.split(",")]
    grid = list(itertools.product(tsm_weights, confs, nms_windows, nms_modes))
    if args.limit is not None:
        grid = grid[: args.limit]
    print(f"Sweep grid: {len(grid)} configs "
          f"(tsm_weights={tsm_weights}, confs={confs}, "
          f"nms_windows={nms_windows}, nms_modes={nms_modes})", flush=True)

    # Load matches and features.
    print(f"Loading {args.feature_type} features from {args.data_dir} "
          f"for split={args.split}...", flush=True)
    matches, match_dirs, match_ids = load_matches(
        args.data_dir, args.split, args.feature_type,
    )
    print(f"  {len(matches)} matches.", flush=True)

    # Build models.
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

    # Cache model scores once.
    print("Caching per-match score arrays (one-time cost)...", flush=True)
    compute_and_cache_scores(tsm, slowfast, matches, match_ids, scales,
                             args.device, args.cache_dir)
    cached_scores = load_cached_scores(match_ids, args.cache_dir)

    # Sweep.
    results_log = Path(args.cache_dir) / "results.csv"
    if not results_log.exists():
        results_log.write_text(
            "tsm_weight,conf_threshold,nms_window,nms_mode,avg_mAP_tight\n"
        )

    best = {"mAP": -1.0, "config": None}
    framerate = TSM_CONFIG["framerate"]
    pred_root_tmp = Path(args.cache_dir) / "_tmp_predictions"
    t_sweep_start = time.time()
    for i, (w, conf, nw, nm) in enumerate(grid):
        t0 = time.time()
        mAP = eval_one_config(
            cached_scores, match_ids, match_dirs, args.data_dir, args.split,
            framerate, tsm_weight=w, conf=conf, nms_window=nw, nms_mode=nm,
            pred_root=pred_root_tmp,
        )
        with open(results_log, "a") as f:
            f.write(f"{w},{conf},{nw},{nm},{mAP:.4f}\n")
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_sweep_start
        eta = (total_elapsed / (i + 1)) * (len(grid) - i - 1)
        marker = ""
        if mAP > best["mAP"]:
            best = {"mAP": mAP, "config": (w, conf, nw, nm)}
            marker = "  << best so far"
        print(f"  [{i+1:3d}/{len(grid)}] w={w:.2f} conf={conf:.2f} "
              f"nms_win={nw:2d} mode={nm:4s}  mAP={mAP:6.3f}%  "
              f"({elapsed:.1f}s, eta={eta:.0f}s){marker}", flush=True)

    # Re-run best config into the final best-output-dir (so the client can
    # use it without touching the cache).
    w, conf, nw, nm = best["config"]
    print(f"\nBest config: tsm_weight={w} conf={conf} "
          f"nms_window={nw} nms_mode={nm}  avg-mAP tight={best['mAP']:.4f}%",
          flush=True)
    best_dir = Path(args.best_output_dir) / "predictions"
    eval_one_config(
        cached_scores, match_ids, match_dirs, args.data_dir, args.split,
        framerate, tsm_weight=w, conf=conf, nms_window=nw, nms_mode=nm,
        pred_root=best_dir,
    )
    print(f"Best-config predictions saved to: {best_dir}", flush=True)
    print(f"Full sweep log: {results_log}", flush=True)


if __name__ == "__main__":
    main()
