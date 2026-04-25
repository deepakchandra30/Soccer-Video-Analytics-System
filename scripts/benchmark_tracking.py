#!/usr/bin/env python
"""Benchmark YOLOv8 + ByteTrack throughput.

Runs in two modes:
  * ``--video PATH`` — real footage (match FPS).  Preferred for the
    final report, but requires having video on disk.  Deepak runs this
    on the remote box with actual match .mkv files.
  * default — synthetic 30-second "video" of random frames.  Detection
    FPS here is representative for the GPU's YOLOv8-m throughput at the
    configured resolution (input-size-dominated), but tracking cost is
    not representative because detection count will be ~0 on noise.
    We supplement with a separate ByteTrack-only microbench using
    synthetic 15-bbox detections to estimate the per-frame association
    overhead.

Emits:
    results/tracking_fps.json
    markdown table on stdout
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.detector import PlayerDetector
from src.tracking.tracker import PlayerTracker


def synthetic_frame(h=720, w=1280, rng=None):
    rng = rng or np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def bench_detection_synthetic(detector, n_frames, h, w):
    rng = np.random.default_rng(42)
    # Warmup
    for _ in range(5):
        detector.detect(synthetic_frame(h, w, rng))
    times = []
    n_dets_total = 0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        frame = synthetic_frame(h, w, rng)
        t1 = time.perf_counter()
        d = detector.detect(frame)
        times.append(time.perf_counter() - t1)
        n_dets_total += d["boxes"].shape[0]
    wall = time.perf_counter() - t0
    return {
        "n_frames": n_frames,
        "frame_hw": (h, w),
        "wall_clock_s": wall,
        "fps": n_frames / wall,
        "per_frame_mean_ms": float(np.mean(times) * 1000),
        "per_frame_p50_ms": float(np.percentile(times, 50) * 1000),
        "per_frame_p95_ms": float(np.percentile(times, 95) * 1000),
        "mean_detections_per_frame": n_dets_total / n_frames,
    }


def bench_tracker_synthetic(tracker, n_frames, n_detections_per_frame=15, h=720, w=1280):
    """ByteTrack microbench on synthetic detections — realistic count but
    random positions, which gives a pessimistic association cost since
    no consistent tracks can form (real footage would be faster on average
    because stable tracks reduce re-association work).
    """
    rng = np.random.default_rng(7)
    frame = synthetic_frame(h, w, rng)  # used only for cut-detector histogram
    # Warmup
    for _ in range(5):
        boxes = np.column_stack([
            rng.integers(0, w - 50, n_detections_per_frame),
            rng.integers(0, h - 50, n_detections_per_frame),
            rng.integers(50, 200, n_detections_per_frame),
            rng.integers(50, 200, n_detections_per_frame),
        ]).astype(np.float32)
        boxes[:, 2] += boxes[:, 0]; boxes[:, 3] += boxes[:, 1]
        tracker.update(frame, {
            "boxes": boxes,
            "confidences": rng.uniform(0.3, 0.95, n_detections_per_frame).astype(np.float32),
            "class_ids": np.zeros(n_detections_per_frame, dtype=np.int32),
        })
    times = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        boxes = np.column_stack([
            rng.integers(0, w - 50, n_detections_per_frame),
            rng.integers(0, h - 50, n_detections_per_frame),
            rng.integers(50, 200, n_detections_per_frame),
            rng.integers(50, 200, n_detections_per_frame),
        ]).astype(np.float32)
        boxes[:, 2] += boxes[:, 0]; boxes[:, 3] += boxes[:, 1]
        t1 = time.perf_counter()
        tracker.update(frame, {
            "boxes": boxes,
            "confidences": rng.uniform(0.3, 0.95, n_detections_per_frame).astype(np.float32),
            "class_ids": np.zeros(n_detections_per_frame, dtype=np.int32),
        })
        times.append(time.perf_counter() - t1)
    wall = time.perf_counter() - t0
    return {
        "n_frames": n_frames,
        "dets_per_frame": n_detections_per_frame,
        "wall_clock_s": wall,
        "fps": n_frames / wall,
        "per_frame_mean_ms": float(np.mean(times) * 1000),
    }


def bench_real_video(detector, tracker, video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"cannot open {video_path}"
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(total, max_frames) if max_frames else total
    det_times, trk_times = [], []
    t0 = time.perf_counter()
    read = 0
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        t1 = time.perf_counter(); d = detector.detect(frame); det_times.append(time.perf_counter() - t1)
        t2 = time.perf_counter(); tracker.update(frame, d); trk_times.append(time.perf_counter() - t2)
    wall = time.perf_counter() - t0
    cap.release()
    return {
        "video_path": str(video_path),
        "video_fps": video_fps,
        "n_frames": read,
        "wall_clock_s": wall,
        "throughput_fps": read / wall,
        "detect_mean_ms": float(np.mean(det_times) * 1000),
        "track_mean_ms": float(np.mean(trk_times) * 1000),
        "realtime_multiplier": (read / wall) / max(video_fps, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None,
                    help="Path to real video (optional; falls back to synthetic)")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=750,
                    help="30s @ 25fps by default")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--output", default="results/tracking_fps.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}  resolution={args.height}x{args.width}")

    detector = PlayerDetector(model_name=args.model, device=device)
    tracker = PlayerTracker()

    summary = {"device": device, "model": args.model,
               "resolution": [args.height, args.width]}

    if args.video and Path(args.video).exists():
        print(f"Benchmarking real video: {args.video}")
        summary["real_video"] = bench_real_video(
            detector, tracker, args.video, args.max_frames,
        )
    else:
        print(f"No video supplied — running synthetic benchmarks on "
              f"{args.max_frames} frames at {args.height}x{args.width}...")
        summary["synthetic_detector"] = bench_detection_synthetic(
            detector, args.max_frames, args.height, args.width,
        )
        # Reset tracker before association benchmark.
        tracker = PlayerTracker()
        summary["synthetic_tracker"] = bench_tracker_synthetic(
            tracker, args.max_frames,
        )
        summary["note"] = (
            "Synthetic benchmarks: detection FPS is representative "
            "(YOLOv8 cost is input-size dominated). Tracker FPS is "
            "pessimistic (random-position dets prevent stable tracks). "
            "Re-run with --video on real match footage for headline number."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown summary
    print("\n## YOLOv8 + ByteTrack throughput\n")
    if "real_video" in summary:
        r = summary["real_video"]
        print(f"| Source | FPS | Detect ms | Track ms | RT multiplier |")
        print(f"|---|---|---|---|---|")
        print(f"| {Path(r['video_path']).name} | "
              f"{r['throughput_fps']:.1f} | {r['detect_mean_ms']:.2f} | "
              f"{r['track_mean_ms']:.2f} | "
              f"{r['realtime_multiplier']:.2f}× |")
    if "synthetic_detector" in summary:
        d = summary["synthetic_detector"]; t = summary["synthetic_tracker"]
        print(f"| Component | FPS | per-frame (mean ms) | p95 ms | Notes |")
        print(f"|---|---|---|---|---|")
        print(f"| Detector (synthetic) | {d['fps']:.1f} | "
              f"{d['per_frame_mean_ms']:.2f} | {d.get('per_frame_p95_ms', 0):.2f} | "
              f"YOLOv8m on {d['frame_hw']} random frames |")
        print(f"| Tracker (synthetic)  | {t['fps']:.1f} | "
              f"{t['per_frame_mean_ms']:.2f} | — | "
              f"ByteTrack on {t['dets_per_frame']} random bboxes/frame |")
    print(f"\nSummary JSON: {args.output}")


if __name__ == "__main__":
    main()
