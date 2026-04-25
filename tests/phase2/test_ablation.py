"""Tests for ablation runner, comparison table, and config."""
import json
import os
import tempfile

import pytest

from src.evaluation.ablation import AblationRunner, AblationResult
from src.evaluation.comparison import generate_comparison_table, load_baseline_results
from config.ablation_config import ABLATION_EXPERIMENTS


class TestAblationResult:
    def test_dataclass_fields(self):
        r = AblationResult(
            name="test", config={"model": "tsm"},
            avg_map=50.0, map_at_1s=40.0, map_at_2s=48.0, map_at_5s=55.0,
            latency_ms=1.5, num_params=100000,
        )
        assert r.name == "test"
        assert r.avg_map == 50.0
        assert r.latency_ms == 1.5
        assert r.num_params == 100000


class TestAblationRunner:
    def test_registers_experiments(self):
        runner = AblationRunner()
        runner.add_experiment("exp1", {"model": "tsm", "feat_dim": 512})
        runner.add_experiment("exp2", {"model": "slowfast", "feat_dim": 512})
        assert len(runner.experiments) == 2

    def test_run_returns_results(self):
        runner = AblationRunner(device="cpu")
        runner.add_experiment("tiny", {"model": "tsm", "feat_dim": 512,
                                       "hidden_dim": 32, "chunk_size": 20})
        results = runner.run()
        assert len(results) == 1
        assert isinstance(results[0], AblationResult)
        assert results[0].name == "tiny"
        assert results[0].num_params > 0

    def test_saves_and_loads_json(self):
        runner = AblationRunner(device="cpu")
        runner.add_experiment("tiny", {"model": "tsm", "feat_dim": 512,
                                       "hidden_dim": 32, "chunk_size": 20})
        runner.run()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            runner.save_results(path)
            loaded = AblationRunner.load_results(path)
            assert len(loaded) == 1
            assert loaded[0].name == "tiny"
            assert loaded[0].num_params == runner.results[0].num_params
        finally:
            os.unlink(path)


class TestAblationConfig:
    def test_has_backbone_experiments(self):
        dims = [e["feat_dim"] for e in ABLATION_EXPERIMENTS]
        assert 512 in dims, "Must have 512-dim (PCA) experiment"
        assert 2048 in dims, "Must have 2048-dim (ResNet-50) experiment"

    def test_has_at_least_11_experiments(self):
        assert len(ABLATION_EXPERIMENTS) >= 11


class TestComparison:
    def test_load_baselines(self):
        baselines = load_baseline_results()
        assert len(baselines) >= 5  # should have multiple published results
        for b in baselines:
            assert "model" in b
            assert "year" in b
            assert "avg_map" in b
            assert "features" in b

    def test_table_format(self):
        results = [AblationResult(name="test_tsm", config={"feature_type": "pca512"},
                                  avg_map=55.0, latency_ms=1.0, num_params=1000)]
        table = generate_comparison_table(results)
        assert "|" in table
        assert "Model" in table

    def test_includes_baselines(self):
        results = [AblationResult(name="test_tsm", config={"feature_type": "pca512"},
                                  avg_map=55.0)]
        table = generate_comparison_table(results)
        assert "NetVLAD" in table
        assert "E2E-Spot" in table

    def test_includes_project_row(self):
        results = [AblationResult(name="test_tsm", config={"feature_type": "pca512"},
                                  avg_map=55.0)]
        table = generate_comparison_table(results)
        assert "test_tsm" in table
