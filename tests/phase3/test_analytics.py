"""Tests for per-player analytics."""
import numpy as np
import pytest

from src.tracking.analytics import compute_player_stats, compute_heatmap


@pytest.fixture
def sample_tracks():
    """Minimal tracks with 2 players across 3 frames."""
    return [
        {"frame_idx": 0, "players": [
            {"track_id": 1, "bbox": [0,0,50,100], "confidence": 0.9,
             "pitch_xy": [10.0, 5.0]},
            {"track_id": 2, "bbox": [100,0,150,100], "confidence": 0.8,
             "pitch_xy": [-20.0, 10.0]},
        ]},
        {"frame_idx": 1, "players": [
            {"track_id": 1, "bbox": [5,0,55,100], "confidence": 0.9,
             "pitch_xy": [11.0, 5.5]},
        ]},
        {"frame_idx": 2, "players": [
            {"track_id": 1, "bbox": [10,0,60,100], "confidence": 0.9,
             "pitch_xy": [12.0, 6.0]},
            {"track_id": 2, "bbox": [105,0,155,100], "confidence": 0.85,
             "pitch_xy": [-19.0, 10.5]},
        ]},
    ]


class TestPlayerStats:
    def test_screen_time(self, sample_tracks):
        stats = compute_player_stats(sample_tracks, fps=25)
        assert stats[1]["screen_time_frames"] == 3
        assert stats[2]["screen_time_frames"] == 2

    def test_mean_position(self, sample_tracks):
        stats = compute_player_stats(sample_tracks, fps=25)
        # player 1 mean x should be ~11
        assert 10 < stats[1]["mean_position"][0] < 12


class TestHeatmap:
    def test_shape(self, sample_tracks):
        hm = compute_heatmap(sample_tracks, resolution=(105, 68), sigma=0)
        assert hm.shape == (68, 105)

    def test_nonzero(self, sample_tracks):
        hm = compute_heatmap(sample_tracks, resolution=(105, 68), sigma=0)
        assert hm.sum() > 0

    def test_empty_tracks(self):
        hm = compute_heatmap([], resolution=(105, 68), sigma=0)
        assert hm.sum() == 0