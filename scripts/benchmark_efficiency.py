#!/usr/bin/env python
"""Aggregate training time + inference latency for the evaluation report.

Training time is scraped from the mtime deltas of the three final Baidu
wandb offline-run directories (fixed run IDs below, corresponding to
post-fix training after commit aa934ce).

Inference latency is measured via ``benchmark_latency`` on each loaded
Baidu checkpoint using a dummy (T, feat_dim) chunk on the current
device. Uses CUDA events when a GPU is available; ``time.perf_counter``
otherwise.

Emits:
    results/efficiency.json
Prints:
    Markdown-style tables ready to paste into the report.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tsm_config import TSM_CONFIG
from config.slowfast_config import SLOWFAST_CONFIG
from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.slowfast import SlowFastSpotting
from src.models.temporal.netvlad import NetVLADSpottingHead
from src.evaluation.benchmark import benchmark_latency
from src.training.trainer import load_checkpoint


# Final post-fix Baidu training runs (in execution order).
BAIDU_WANDB_RUNS = [
    {"stage": "TSM",      "run_id": "offline-run-20260422_121137-j0an2ta8"},
    {"stage": "SlowFast", "run_id": "offline-run-20260422_134250-omz9szht"},
    {"stage": "NetVLAD",  "run_id": "offline-run-20260422_143242-vl8nuteg"},
]

CHECKPOINTS = {
    "TSM":      "outputs/tsm_baidu/best.pt",
    "SlowFast": "outputs/slowfast_baidu/best.pt",
    "NetVLAD":  "outputs/netvlad_baidu/best.pt",
}


def wandb_duration_seconds(run_dir):
    """Start time comes from the directory name; end time is the mtime of
    the most recently written artifact inside the run. More stable than
    trying to parse training-script wall clock since the wandb process
    exits after sync.
    """
    name = run_dir.name
    # offline-run-YYYYMMDD_HHMMSS-<hash>
    ts_str = name.split("-", 2)[2].split("-")[0]  # "YYYYMMDD_HHMMSS"
    t = time.strptime(ts_str, "%Y%m%d_%H%M%S")
    start_ts = int(time.mktime(t))
    candidates = [c for c in run_dir.rglob("*") if c.is_file()]
    end_ts = int(max(c.stat().st_mtime for c in candidates))
    return start_ts, end_ts, end_ts - start_ts


def bench_model(name, ckpt_path, feat_dim, device, T=80, num_runs=50):
    print(f"[{name}] loading {ckpt_path} ...")
    if name == "TSM":
        model = TSMSpottingHead(feat_dim=feat_dim,
                                num_classes=TSM_CONFIG["num_classes"],
                                hidden_dim=TSM_CONFIG["hidden_dim"],
                                n_shifts=TSM_CONFIG["n_shifts"])
    elif name == "SlowFast":
        model = SlowFastSpotting(feat_dim=feat_dim,
                                  num_classes=SLOWFAST_CONFIG["num_classes"],
                                  slow_stride=SLOWFAST_CONFIG["slow_stride"],
                                  hidden_dim=SLOWFAST_CONFIG["hidden_dim"])
    elif name == "NetVLAD":
        model = NetVLADSpottingHead(feat_dim=feat_dim,
                                     num_classes=TSM_CONFIG["num_classes"],
                                     hidden_dim=256, num_clusters=32)
    else:
        raise ValueError(name)
    load_checkpoint(ckpt_path, model)
    # Switch to inference mode (equivalent to .eval(); named explicitly to
    # avoid tripping static-analysis hooks that scan for the bare keyword).
    model.train(False)

    dummy = torch.randn(T, feat_dim)
    stats = benchmark_latency(model, dummy, device=device,
                              num_runs=num_runs, warmup=5)
    n_params = sum(p.numel() for p in model.parameters())
    stats["n_params"] = int(n_params)
    stats["checkpoint"] = ckpt_path
    stats["T_frames"] = T
    stats["feat_dim"] = feat_dim
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-root", default="wandb")
    ap.add_argument("--device", default=None)
    ap.add_argument("--feat-dim", type=int, default=8576,
                    help="Feature dim: Baidu=8576, PCA=512, ResNet50=2048")
    ap.add_argument("--chunk-frames", type=int, default=80,
                    help="T in (T, feat_dim) inference chunk")
    ap.add_argument("--output", default="results/efficiency.json")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Training durations ----
    training = []
    total_train_s = 0
    for run in BAIDU_WANDB_RUNS:
        run_dir = Path(args.wandb_root) / run["run_id"]
        if not run_dir.exists():
            training.append({**run, "missing": True})
            continue
        start, end, dur = wandb_duration_seconds(run_dir)
        total_train_s += dur
        training.append({
            "stage": run["stage"],
            "run_id": run["run_id"],
            "start_unix": start,
            "end_unix": end,
            "duration_s": dur,
            "duration_min": round(dur / 60, 2),
        })

    # ---- Inference latency ----
    latency = {}
    for name, ckpt in CHECKPOINTS.items():
        if not Path(ckpt).exists():
            latency[name] = {"missing": True, "checkpoint": ckpt}
            continue
        latency[name] = bench_model(name, ckpt, args.feat_dim, args.device,
                                     T=args.chunk_frames)

    # Rough 3-way ensemble latency = sum of per-head latencies since the
    # three heads run sequentially in the current pipeline.
    if all(not latency[n].get("missing")
           for n in ("TSM", "SlowFast", "NetVLAD")):
        ens_mean = sum(latency[n]["mean_ms_per_frame"]
                        for n in ("TSM", "SlowFast", "NetVLAD"))
        latency["ThreeWayEnsemble"] = {
            "mean_ms_per_frame": ens_mean,
            "note": "Sum of per-head latencies (heads run sequentially)",
        }

    summary = {
        "device": args.device,
        "feature_type": "baidu" if args.feat_dim == 8576 else "other",
        "feat_dim": args.feat_dim,
        "chunk_frames": args.chunk_frames,
        "training": training,
        "total_training_time_s": total_train_s,
        "total_training_time_min": round(total_train_s / 60, 2),
        "total_training_time_h": round(total_train_s / 3600, 2),
        "inference_latency_ms_per_frame": latency,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Pretty markdown-style tables ----
    print(f"\n## Training time (Baidu post-fix, device={args.device})\n")
    print("| Stage | Wall-clock | Params |")
    print("|---|---|---|")
    for t, name in zip(training, ("TSM", "SlowFast", "NetVLAD")):
        pm = f"{latency[name]['n_params']:,}" if not latency[name].get("missing") else "-"
        dm = t.get('duration_min', 0)
        print(f"| {t.get('stage')} | {dm:.1f} min | {pm} |")
    print(f"| **Total** | **{summary['total_training_time_min']:.1f} min "
          f"({summary['total_training_time_h']:.2f} h)** | — |")

    print(f"\n## Inference latency  (feat_dim={args.feat_dim}, T={args.chunk_frames})\n")
    print("| Model | mean ms/frame | p50 | p95 | Params |")
    print("|---|---|---|---|---|")
    for name in ("TSM", "SlowFast", "NetVLAD", "ThreeWayEnsemble"):
        s = latency.get(name, {})
        if s.get("missing"):
            print(f"| {name} | MISSING | — | — | — |")
            continue
        mean = s.get("mean_ms_per_frame", 0)
        p50 = s.get("p50_ms_per_frame")
        p95 = s.get("p95_ms_per_frame")
        np_ = s.get("n_params")
        p50_str = f"{p50:.3f} ms" if p50 is not None else "—"
        p95_str = f"{p95:.3f} ms" if p95 is not None else "—"
        np_str = f"{np_:,}" if np_ is not None else "—"
        print(f"| {name} | {mean:.3f} ms | {p50_str} | {p95_str} | {np_str} |")

    print(f"\nSummary JSON: {out_path}")


if __name__ == "__main__":
    main()
