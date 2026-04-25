"""Tests for visualization utilities."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.evaluation.ablation import AblationResult
from src.evaluation.visualization import plot_accuracy_vs_speed, plot_ablation_heatmap


@pytest.fixture(autouse=True)
def cleanup_plots():
    yield
    plt.close("all")


@pytest.fixture
def mock_results():
    return [
        AblationResult(name="tsm_512", config={"model": "tsm"},
                       avg_map=50.0, map_at_1s=40.0, map_at_2s=48.0,
                       map_at_5s=55.0, latency_ms=0.5, num_params=100000),
        AblationResult(name="slowfast", config={"model": "slowfast"},
                       avg_map=53.0, map_at_1s=43.0, map_at_2s=51.0,
                       map_at_5s=58.0, latency_ms=1.2, num_params=200000),
        AblationResult(name="two_stage", config={"model": "two_stage"},
                       avg_map=52.0, map_at_1s=42.0, map_at_2s=50.0,
                       map_at_5s=56.0, latency_ms=0.8, num_params=300000),
    ]


class TestAccuracyVsSpeed:
    def test_creates_file(self, mock_results, tmp_path):
        path = str(tmp_path / "test_scatter.png")
        plot_accuracy_vs_speed(mock_results, output_path=path)
        assert os.path.exists(path)

    def test_handles_empty_data(self, tmp_path):
        path = str(tmp_path / "empty_scatter.png")
        plot_accuracy_vs_speed([], output_path=path)
        assert os.path.exists(path)


class TestAblationHeatmap:
    def test_creates_file(self, mock_results, tmp_path):
        path = str(tmp_path / "test_heatmap.png")
        plot_ablation_heatmap(mock_results, output_path=path)
        assert os.path.exists(path)

    def test_handles_empty_data(self, tmp_path):
        path = str(tmp_path / "empty_heatmap.png")
        plot_ablation_heatmap([], output_path=path)
        assert os.path.exists(path)
