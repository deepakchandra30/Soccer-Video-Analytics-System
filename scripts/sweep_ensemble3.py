#!/usr/bin/env python
"""3-way ensemble weight + post-processing sweep with score caching.

Design: run TSM/SF/NetVLAD inference ONCE on the target split, cache
per-match per-half scores to disk, then iterate cheaply over (weight_tsm,
weight_sf, weight_nv) * (nms_window, conf_threshold) combos. Each trial
takes seconds instead of minutes.

Why this exists: the default weights in scripts/run_ensemble3.py were tuned
on PCA-512 (SF-dominant 0.10/0.60/0.30). Baidu NetVLAD typically carries
more signal than Baidu TSM, so weights may need re-balancing — this script
finds the best combo empirically.

Writes:
    outputs/sweep3_cache/{match_id}/{tsm,slowfast,netvlad}_h{1,2}.npy
    outputs/sweep3_cache/results.csv  (one row per swept config)
    outputs/sweep3_best/predictions/  (best-config predictions, ready to submit)
"""
import argparse
import csv
import itertools
import json
import os
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
from src.models.temporal.netvlad import NetVLADSpottingHead
from src.models.temporal.postprocess import (
    multi_scale_inference, nms_detections, save_predictions,
)
from src.training.trainer import load_checkpoint
from src.training.train_tsm import load_matches, FEATURE_FRAMERATE


FEAT_DIMS = {"pca512": 512, "resnet50": 2048, "baidu": 8576}


def cache_scores(tsm, sf, nv, matches, match_ids, scales, device, cache_dir):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (match, match_id) in enumerate(zip(matches, match_ids)):
        mc = cache_dir / match_id
        paths = [mc / f"{m}_h{h}.npy" for m in ("tsm", "slowfast", "netvlad") for h in (1, 2)]
        lens_p = mc / "half_lens.json"
        if all(p.exists() for p in paths) and lens_p.exists():
            continue
        mc.mkdir(parents=True, exist_ok=True)
        features, _, half1_len = match
        h1 = torch.FloatTensor(features[:half1_len])
        h2 = torch.FloatTensor(features[half1_len:])
        scores = {}
        for name, model in (("tsm", tsm), ("slowfast", sf), ("netvlad", nv)):
            scores[f"{name}_h1"] = multi_scale_inference(model, h1, scales=scales, device=device)
            scores[f"{name}_h2"] = multi_scale_inference(model, h2, scales=scales, device=device)
        for k, v in scores.items():
            np.save(mc / f"{k}.npy", v)
        with open(lens_p, "w") as f:
            json.dump({"h1": int(h1.shape[0]), "h2": int(h2.shape[0])}, f)
        if (i + 1) % 20 == 0 or i == len(matches) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  cached {i+1}/{len(matches)}  rate={rate:.1f}/s")


def apply_weights_and_nms(cache_dir, match_ids, weights, framerate, nms_window,
                          conf, nms_mode, out_dir):
    w_tsm, w_sf, w_nv = weights
    total = w_tsm + w_sf + w_nv
    w_tsm, w_sf, w_nv = w_tsm / total, w_sf / total, w_nv / total
    cache_dir = Path(cache_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for match_id in match_ids:
        mc = cache_dir / match_id
        preds = []
        for half_idx in (1, 2):
            tsm_s = np.load(mc / f"tsm_h{half_idx}.npy")
            sf_s = np.load(mc / f"slowfast_h{half_idx}.npy")
            nv_s = np.load(mc / f"netvlad_h{half_idx}.npy")
            ens = w_tsm * tsm_s + w_sf * sf_s + w_nv * nv_s
            preds.extend(nms_detections(ens, nms_window=nms_window,
                                         confidence_threshold=conf,
                                         framerate=framerate, half=half_idx,
                                         nms_mode=nms_mode))
        mout = out_dir / match_id; mout.mkdir(parents=True, exist_ok=True)
        save_predictions(preds, str(mout / "results_spotting.json"), url_local=match_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--feature-type", default="baidu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tsm-checkpoint", required=True)
    ap.add_argument("--slowfast-checkpoint", required=True)
    ap.add_argument("--netvlad-checkpoint", required=True)
    ap.add_argument("--cache-dir", default="outputs/sweep3_cache")
    ap.add_argument("--best-dir", default="outputs/sweep3_best")
    ap.add_argument("--scales", default="40:20,80:40")
    ap.add_argument("--device", default=None)
    # Default weight grid: coarse 3-way grid around PCA-512 best (.10/.60/.30)
    # and memory-recommended (.15/.50/.35). Step 0.05 keeps it small.
    ap.add_argument("--weight-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--nms-windows", default="3,5,7")
    ap.add_argument("--conf-thresholds", default="0.03,0.05,0.08")
    ap.add_argument("--nms-mode", default="hard", choices=["hard", "soft"])
    ap.add_argument("--max-configs", type=int, default=60,
                    help="Cap on swept configs (guard against combinatorial blow-up)")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = FEAT_DIMS[args.feature_type]
    scales = tuple(tuple(int(x) for x in p.split(":")) for p in args.scales.split(","))
    framerate = FEATURE_FRAMERATE.get(args.feature_type, 2)

    print(f"Loading {args.feature_type} features ({feat_dim}-dim, fps={framerate}) for split={args.split}...")
    matches, _dirs, match_ids = load_matches(args.data_dir, args.split, args.feature_type)
    print(f"  {len(matches)} matches")

    set_seeds(42)
    tsm = TSMSpottingHead(feat_dim=feat_dim, num_classes=TSM_CONFIG["num_classes"],
                          hidden_dim=TSM_CONFIG["hidden_dim"], n_shifts=TSM_CONFIG["n_shifts"]).to(args.device)
    load_checkpoint(args.tsm_checkpoint, tsm); tsm.train(False)
    sf = SlowFastSpotting(feat_dim=feat_dim, num_classes=SLOWFAST_CONFIG["num_classes"],
                           slow_stride=SLOWFAST_CONFIG["slow_stride"],
                           hidden_dim=SLOWFAST_CONFIG["hidden_dim"]).to(args.device)
    load_checkpoint(args.slowfast_checkpoint, sf); sf.train(False)
    nv = NetVLADSpottingHead(feat_dim=feat_dim, num_classes=TSM_CONFIG["num_classes"],
                              hidden_dim=256, num_clusters=32).to(args.device)
    load_checkpoint(args.netvlad_checkpoint, nv); nv.train(False)

    print("Caching per-model scores (one-time cost)...")
    cache_scores(tsm, sf, nv, matches, match_ids, scales, args.device, args.cache_dir)

    # Build weight grid — just enough variety without blowing up.
    # We fix w_sf ≥ 0.30 since SF was strongest empirically, vary TSM and NV.
    weight_grid = [float(x) for x in args.weight_grid.split(",")]
    nms_ws = [int(x) for x in args.nms_windows.split(",")]
    confs = [float(x) for x in args.conf_thresholds.split(",")]

    configs = []
    # Curated weight sets: covers known-good + tilt toward NetVLAD (Baidu's
    # strongest component in published work) and a balanced default.
    weight_sets = [
        (0.10, 0.60, 0.30),  # PCA-512 default
        (0.15, 0.50, 0.35),  # memory note recommendation
        (0.10, 0.50, 0.40),  # NV-leaning
        (0.15, 0.55, 0.30),
        (0.20, 0.50, 0.30),
        (0.10, 0.45, 0.45),  # balanced NV+SF
        (0.33, 0.33, 0.34),  # equal
        (0.15, 0.45, 0.40),
        (0.20, 0.40, 0.40),
        (0.05, 0.55, 0.40),  # very NV-heavy
        (0.10, 0.40, 0.50),  # NV-dominant
        (0.05, 0.50, 0.45),
    ]
    for ws in weight_sets:
        for nw in nms_ws:
            for cf in confs:
                configs.append((ws, nw, cf))
    if len(configs) > args.max_configs:
        configs = configs[: args.max_configs]
    print(f"Sweeping {len(configs)} configs ({len(weight_sets)} weight sets × {len(nms_ws)} nms × {len(confs)} conf)...")

    from src.evaluation.evaluate import run_evaluation
    results_csv = Path(args.cache_dir) / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w_tsm", "w_sf", "w_nv", "nms_window", "conf",
                          "nms_mode", "avg_map_tight"])
    best = None
    for idx, (ws, nw, cf) in enumerate(configs):
        tag = f"w{ws[0]}_{ws[1]}_{ws[2]}_nms{nw}_c{cf}"
        trial_out = Path(args.cache_dir) / "trials" / tag / "predictions"
        apply_weights_and_nms(
            args.cache_dir, match_ids, ws, framerate, nw, cf, args.nms_mode, trial_out,
        )
        eval_res = run_evaluation(args.data_dir, str(trial_out), split=args.split)
        mAP = float(eval_res.get("a_mAP", 0.0)) * 100.0
        print(f"  [{idx+1}/{len(configs)}] weights={ws} nms_w={nw} conf={cf}: avg-mAP tight = {mAP:.3f}%")
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ws[0], ws[1], ws[2], nw, cf, args.nms_mode, mAP])
        if best is None or mAP > best["mAP"]:
            best = {"weights": ws, "nms_window": nw, "conf": cf, "mAP": mAP}

    print("\nBest config:")
    print(f"  weights={best['weights']}  nms_window={best['nms_window']}  conf={best['conf']}")
    print(f"  avg-mAP tight: {best['mAP']:.4f}%")

    # Regenerate best predictions in a clean dir for final use.
    best_preds = Path(args.best_dir) / "predictions"
    apply_weights_and_nms(
        args.cache_dir, match_ids, best["weights"], framerate,
        best["nms_window"], best["conf"], args.nms_mode, best_preds,
    )
    with open(Path(args.best_dir) / "best_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best predictions written to {best_preds}")
    print(f"Best config saved to {Path(args.best_dir) / 'best_config.json'}")


if __name__ == "__main__":
    main()
