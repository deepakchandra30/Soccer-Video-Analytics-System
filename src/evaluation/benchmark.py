"""Latency benchmarking for temporal models and the two-stage pipeline."""
import time

import numpy as np
import torch

from src.models.temporal.postprocess import sliding_window_inference
from src.models.temporal.pipeline import TwoStagePipeline


def benchmark_latency(model, features, device="cpu", num_runs=100, warmup=10):
    """Measure per-frame inference latency.

    Uses CUDA events for GPU timing, time.perf_counter for CPU.
    Returns dict with mean/std/p50/p95 ms per frame.
    """
    model = model.to(device).eval()
    T = features.shape[0]
    use_cuda = device != "cpu" and torch.cuda.is_available()

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = features.unsqueeze(0).to(device)
            _ = model(x)
    if use_cuda:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        x = features.unsqueeze(0).to(device)

        if use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            with torch.no_grad():
                _ = model(x)
            end_ev.record()
            torch.cuda.synchronize()
            elapsed_ms = start_ev.elapsed_time(end_ev)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed_ms / T)

    latencies = np.array(latencies)
    return {
        "mean_ms_per_frame": float(np.mean(latencies)),
        "std_ms_per_frame": float(np.std(latencies)),
        "p50_ms_per_frame": float(np.percentile(latencies, 50)),
        "p95_ms_per_frame": float(np.percentile(latencies, 95)),
    }


def benchmark_pipeline(coarse_model, fine_model, features, pipeline_config,
                        *, framerate, device="cpu", num_runs=10, warmup=2):
    """Compare single-stage vs two-stage latency.

    ``framerate`` is required (keyword-only) for the same reason it is
    required on TwoStagePipeline: PCA-512 / ResNet-50 are 2fps, Baidu is 1fps,
    and choosing the wrong one writes prediction positions at the wrong time
    so tight-mAP collapses to ~1%. Pass 2 for PCA / ResNet, 1 for Baidu.

    Returns dict with single_stage_ms, two_stage_ms, speedup_factor,
    and candidate_ratio.
    """
    T = features.shape[0]

    # single-stage: TSM on full match
    single = benchmark_latency(coarse_model, features, device, num_runs, warmup)

    # two-stage: run pipeline to measure actual candidate ratio
    pipeline = TwoStagePipeline(coarse_model, fine_model, pipeline_config,
                                framerate=framerate, device=device)
    half_len = T // 2
    half1 = features[:half_len]
    half2 = features[half_len:]
    pipeline.run(half1, half2)

    candidate_count = pipeline.get_candidate_frame_count()
    candidate_ratio = candidate_count / max(T, 1)

    # two-stage timing: coarse on full + fine on candidates
    coarse_time = single["mean_ms_per_frame"] * T

    # estimate fine stage time
    if candidate_count > 0:
        fine_features = features[:min(candidate_count, T)]
        fine = benchmark_latency(fine_model, fine_features, device,
                                 num_runs, warmup)
        fine_time = fine["mean_ms_per_frame"] * candidate_count
    else:
        fine_time = 0.0

    two_stage_total = coarse_time + fine_time
    single_total = single["mean_ms_per_frame"] * T

    speedup = single_total / max(two_stage_total, 1e-6)

    return {
        "single_stage_ms": single_total,
        "two_stage_ms": two_stage_total,
        "speedup_factor": speedup,
        "candidate_ratio": candidate_ratio,
    }