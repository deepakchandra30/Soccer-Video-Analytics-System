"""Tests for latency benchmarking utilities."""
import pytest
import torch

from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.slowfast import SlowFastSpotting
from src.evaluation.benchmark import benchmark_latency, benchmark_pipeline
from config.pipeline_config import PIPELINE_CONFIG


@pytest.fixture
def tiny_tsm():
    return TSMSpottingHead(feat_dim=512, num_classes=17, hidden_dim=32, n_shifts=1)


@pytest.fixture
def tiny_slowfast():
    return SlowFastSpotting(feat_dim=512, num_classes=17, hidden_dim=32)


class TestBenchmarkLatency:
    def test_returns_correct_keys(self, tiny_tsm):
        features = torch.randn(40, 512)
        result = benchmark_latency(tiny_tsm, features, device="cpu",
                                   num_runs=3, warmup=1)
        assert "mean_ms_per_frame" in result
        assert "std_ms_per_frame" in result
        assert "p50_ms_per_frame" in result
        assert "p95_ms_per_frame" in result

    def test_positive_values(self, tiny_tsm):
        features = torch.randn(40, 512)
        result = benchmark_latency(tiny_tsm, features, device="cpu",
                                   num_runs=3, warmup=1)
        assert result["mean_ms_per_frame"] > 0
        assert result["p50_ms_per_frame"] > 0
        assert result["p95_ms_per_frame"] > 0


class TestBenchmarkPipeline:
    def test_comparison_format(self, tiny_tsm, tiny_slowfast):
        features = torch.randn(200, 512)
        result = benchmark_pipeline(
            tiny_tsm, tiny_slowfast, features, PIPELINE_CONFIG,
            framerate=2, device="cpu", num_runs=2, warmup=1,
        )
        assert "single_stage_ms" in result
        assert "two_stage_ms" in result
        assert "speedup_factor" in result
        assert "candidate_ratio" in result
