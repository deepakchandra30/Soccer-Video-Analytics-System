#!/usr/bin/env python
"""Architectural ablation from cached 3-way scores.

Reuses the per-model multi-scale scores cached by ``sweep_ensemble3.py``
under ``outputs/sweep3_cache/`` to generate TSM-only, SF-only,
NetVLAD-only, 2-way, and 3-way predictions — no new inference, just a
weighted combination + NMS + evaluation per mode.

Writes:
    outputs/ablation_cached/<mode>/predictions/<match>/results_spotting.json
    results/ablation_architecture.json   (consolidated per-mode metrics)
    results/per_class_ap_ablation_<mode>.svg
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sweep_ensemble3 import apply_weights_and_nms
from scripts.generate_full_metrics import (
    evaluate_with_metric, extract_per_class,
    compute_per_tolerance_tight, SOCCERNET_V2_CLASSES,
    plot_per_class_bar,
)
from src.training.train_tsm import load_matches, FEATURE_FRAMERATE


# The modes the supervisor's rubric asks about (TSM / SF / Two-stage ~= TSM+SF,
# 3-way including NetVLAD). We include all three singles + both 2-way partials
# + 3-way for a complete matrix.
MODES = {
    "tsm_only":     (1.0, 0.0, 0.0),
    "slowfast_only": (0.0, 1.0, 0.0),
    "netvlad_only": (0.0, 0.0, 1.0),
    "tsm_slowfast": (0.14, 0.86, 0.0),   # TSM+SF 2-way (finalize watchdog default)
    "slowfast_netvlad": (0.0, 0.67, 0.33),
    "three_way":    (0.10, 0.60, 0.30),   # sweep-best
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="outputs/sweep3_cache")
    ap.add_argument("--out-root", default="outputs/ablation_cached")
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--feature-type", default="baidu")
    ap.add_argument("--nms-window", type=int, default=3)
    ap.add_argument("--conf-threshold", type=float, default=0.03)
    ap.add_argument("--nms-mode", default="hard")
    ap.add_argument("--summary-path", default="results/ablation_architecture.json")
    args = ap.parse_args()

    framerate = FEATURE_FRAMERATE.get(args.feature_type, 2)
    print(f"Loading match IDs for {args.feature_type}/{args.split} (framerate={framerate})...")
    _matches, _dirs, match_ids = load_matches(args.data_dir, args.split, args.feature_type)
    print(f"  {len(match_ids)} matches.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": args.split,
        "feature_type": args.feature_type,
        "framerate": framerate,
        "nms_window": args.nms_window,
        "conf_threshold": args.conf_threshold,
        "nms_mode": args.nms_mode,
        "cache_dir": args.cache_dir,
        "modes": {},
    }

    for mode, weights in MODES.items():
        print(f"\n[{mode}] weights={weights}")
        mode_out = out_root / mode
        preds_dir = mode_out / "predictions"
        apply_weights_and_nms(
            args.cache_dir, match_ids, weights, framerate,
            args.nms_window, args.conf_threshold, args.nms_mode, str(preds_dir),
        )
        # Evaluate tight + loose + per-tolerance tight
        tight = evaluate_with_metric(args.data_dir, str(preds_dir), args.split, "tight")
        loose = evaluate_with_metric(args.data_dir, str(preds_dir), args.split, "loose")
        per_tol = compute_per_tolerance_tight(args.data_dir, str(preds_dir), args.split)
        per_class_tight = extract_per_class(tight, num_classes=17)
        per_class_loose = extract_per_class(loose, num_classes=17)

        mode_result = {
            "weights": list(weights),
            "tight_avg_mAP_pct": float(tight["a_mAP"]) * 100.0,
            "loose_avg_mAP_pct": float(loose["a_mAP"]) * 100.0,
            "tight_at_1s_pct": per_tol.get(1, 0.0) * 100.0,
            "tight_at_2s_pct": per_tol.get(2, 0.0) * 100.0,
            "tight_at_5s_pct": per_tol.get(5, 0.0) * 100.0,
            "tight_visible_pct": float(tight.get("a_mAP_visible") or 0.0) * 100.0,
            "tight_unshown_pct": float(tight.get("a_mAP_unshown") or 0.0) * 100.0,
            "per_class_tight_pct": {
                SOCCERNET_V2_CLASSES[i]: float(per_class_tight[i]) * 100.0
                for i in range(len(SOCCERNET_V2_CLASSES))
            },
            "per_class_loose_pct": {
                SOCCERNET_V2_CLASSES[i]: float(per_class_loose[i]) * 100.0
                for i in range(len(SOCCERNET_V2_CLASSES))
            },
            "predictions_dir": str(preds_dir),
        }
        summary["modes"][mode] = mode_result
        print(f"  tight={mode_result['tight_avg_mAP_pct']:.3f}%  "
              f"loose={mode_result['loose_avg_mAP_pct']:.3f}%  "
              f"@1s={mode_result['tight_at_1s_pct']:.3f}%")

        # Per-class SVG for this mode (small, zero extra inference)
        svg_path = Path("results") / f"per_class_ap_ablation_{mode}.svg"
        plot_per_class_bar(
            per_class_tight, SOCCERNET_V2_CLASSES, str(svg_path),
            title=f"Per-class AP (tight) — ablation/{mode}, "
                  f"avg-mAP={mode_result['tight_avg_mAP_pct']:.2f}%",
        )

    # Emit summary JSON
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary: {summary_path}")

    # Pretty table
    print("\n" + "=" * 72)
    print(f"{'Mode':<18}{'Weights':<22}{'Tight':>9}{'Loose':>9}{'@1s':>9}")
    print("-" * 72)
    for mode, r in summary["modes"].items():
        w = r["weights"]; ws = f"({w[0]},{w[1]},{w[2]})"
        print(f"{mode:<18}{ws:<22}{r['tight_avg_mAP_pct']:>8.3f}%"
              f"{r['loose_avg_mAP_pct']:>8.3f}%"
              f"{r['tight_at_1s_pct']:>8.3f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
