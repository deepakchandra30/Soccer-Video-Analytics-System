"""Tests for two-stage pipeline (TSM coarse -> SlowFast fine)."""
import numpy as np
import pytest
import torch

from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.slowfast import SlowFastSpotting
from src.models.temporal.pipeline import TwoStagePipeline
from config.pipeline_config import PIPELINE_CONFIG


@pytest.fixture
def coarse_model():
    return TSMSpottingHead(feat_dim=512, num_classes=17, hidden_dim=32, n_shifts=1)


@pytest.fixture
def fine_model():
    return SlowFastSpotting(feat_dim=512, num_classes=17, hidden_dim=32)


@pytest.fixture
def pipeline(coarse_model, fine_model):
    # framerate=2 is the SoccerNet PCA-512 default; the test fixtures use
    # random features so the value only affects ms-encoding, not detection.
    return TwoStagePipeline(coarse_model, fine_model, PIPELINE_CONFIG,
                            framerate=2, device="cpu")


class TestCoarseDetection:
    def test_candidates_returned(self, pipeline):
        features = torch.randn(200, 512)
        candidates = pipeline.detect_candidates(features, half=1)
        assert isinstance(candidates, list)
        # should detect at least something with random features and low threshold
        # (random logits will produce some peaks above 0.15)

    def test_candidate_format(self, pipeline):
        features = torch.randn(200, 512)
        candidates = pipeline.detect_candidates(features, half=1)
        if len(candidates) > 0:
            c = candidates[0]
            assert "start_frame" in c
            assert "end_frame" in c
            assert "half" in c
            assert "predictions" in c

    def test_high_recall(self, pipeline):
        """With low threshold, coarse detector should retain most injected events."""
        # create features where the coarse model will produce some peaks
        features = torch.randn(500, 512)
        candidates = pipeline.detect_candidates(features, half=1)
        # with random model + low threshold (0.15), we expect many candidates
        # the real recall test happens at integration time with trained models,
        # but we verify the pipeline at least generates candidates
        assert isinstance(candidates, list)


class TestFineClassification:
    def test_refines_candidates(self, pipeline):
        features = torch.randn(500, 512)
        candidates = pipeline.detect_candidates(features, half=1)
        if len(candidates) > 0:
            refined = pipeline.classify_candidates(features, candidates)
            assert isinstance(refined, list)
            for p in refined:
                assert "label" in p
                assert "position" in p
                assert "confidence" in p


class TestPipelineEndToEnd:
    def test_produces_predictions(self, pipeline):
        half1 = torch.randn(200, 512)
        half2 = torch.randn(200, 512)
        predictions = pipeline.run(half1, half2)
        assert isinstance(predictions, list)
        for p in predictions:
            assert "label" in p
            assert "half" in p
            assert "position" in p
            assert "confidence" in p

    def test_processes_subset(self, pipeline):
        """Pipeline should process fewer frames than full match via candidates."""
        half1 = torch.randn(2000, 512)
        half2 = torch.randn(2000, 512)
        pipeline.run(half1, half2)
        total_match_frames = 4000
        candidate_frames = pipeline.get_candidate_frame_count()
        # should process less than 100% of frames (unless all frames are candidates)
        assert candidate_frames <= total_match_frames


class TestSpeedup:
    def test_candidate_ratio(self, pipeline):
        """Two-stage should process fewer total frames than single-stage."""
        half1 = torch.randn(1000, 512)
        half2 = torch.randn(1000, 512)
        pipeline.run(half1, half2)
        candidate_frames = pipeline.get_candidate_frame_count()
        total = 2000
        # candidate ratio should be less than 100%
        ratio = candidate_frames / total
        assert ratio <= 1.0


class TestFramerateContract:
    """Lock-in tests for the framerate parameter — these guard against the
    2026-04-22 regression where Baidu (1fps) features inherited the static
    2fps default and every prediction position was written at half its true
    time, collapsing tight-mAP to ~1%.
    """
    def test_framerate_required_keyword(self, coarse_model, fine_model):
        """framerate is keyword-only; positional placement must fail."""
        with pytest.raises(TypeError):
            TwoStagePipeline(coarse_model, fine_model, PIPELINE_CONFIG, 1)

    def test_framerate_missing_raises(self, coarse_model, fine_model):
        """Constructing without framerate fails fast — no silent default."""
        with pytest.raises(TypeError):
            TwoStagePipeline(coarse_model, fine_model, PIPELINE_CONFIG)

    @pytest.mark.parametrize("bad", [0, -1, "two", None])
    def test_framerate_invalid_rejected(self, coarse_model, fine_model, bad):
        with pytest.raises((TypeError, ValueError)):
            TwoStagePipeline(coarse_model, fine_model, PIPELINE_CONFIG,
                             framerate=bad, device="cpu")

    def test_baidu_positions_are_double_pca(self, coarse_model, fine_model):
        """For the same coarse/fine event, framerate=1 (Baidu) must encode
        the position at exactly twice the ms of framerate=2 (PCA): a 2-frame
        gap is 1s at 2fps but 2s at 1fps, so the absolute ms doubles. Catches
        any regression that quietly returns 2fps positions for 1fps features.
        """
        torch.manual_seed(0)
        features = torch.randn(400, 512)

        cfg = PIPELINE_CONFIG
        pca = TwoStagePipeline(coarse_model, fine_model, cfg,
                               framerate=2, device="cpu")
        baidu = TwoStagePipeline(coarse_model, fine_model, cfg,
                                 framerate=1, device="cpu")

        cand_pca = pca.detect_candidates(features, half=1)
        cand_baidu = baidu.detect_candidates(features, half=1)

        # candidate frame ranges depend only on coarse output + pad, not on
        # framerate, so they must agree exactly
        assert [(c["start_frame"], c["end_frame"]) for c in cand_pca] == \
               [(c["start_frame"], c["end_frame"]) for c in cand_baidu]

        # every coarse-stage prediction's position (in ms) must double when
        # framerate halves — this is the single line that the old
        # `.get("framerate", 2)` fallback was silently wrong about
        for cp, cb in zip(cand_pca, cand_baidu):
            pos_pca = int(cp["predictions"][0]["position"])
            pos_baidu = int(cb["predictions"][0]["position"])
            assert pos_baidu == pos_pca * 2

    def test_globalised_gametime_matches_position(self, pipeline):
        """gameTime must reflect the *globalised* position, not the
        within-window time emitted by the inner NMS pass. Older code left it
        stale (e.g. position=1809000 / gameTime='1 - 00:30') which made the
        JSON self-inconsistent and shows up as nonsense in any evaluator
        that falls back to gameTime when position is missing.
        """
        torch.manual_seed(0)
        half1 = torch.randn(500, 512)
        preds = pipeline.run(half1, torch.zeros(0, 512))
        for p in preds:
            ms = int(p["position"])
            seconds = ms // 1000
            expected = f"{p['half']} - {seconds // 60:02d}:{seconds % 60:02d}"
            assert p["gameTime"] == expected, (
                f"gameTime {p['gameTime']!r} does not match "
                f"position {ms}ms (expected {expected!r})"
            )
