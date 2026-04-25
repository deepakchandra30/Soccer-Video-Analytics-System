"""Tests for match analytics computation."""
import json
import os
import tempfile

import pytest

from src.analytics.match_stats import (
    compute_match_analytics,
    save_match_analytics,
    load_match_analytics,
)


class TestMatchAnalytics:
    def test_computes_all_fields(self, sample_events, sample_tracks):
        result = compute_match_analytics(sample_events, sample_tracks)
        assert "player_stats" in result
        assert "attributed_events" in result
        assert "event_involvement" in result
        assert "num_events" in result
        assert result["num_events"] == 3

    def test_save_and_load(self, sample_events, sample_tracks):
        analytics = compute_match_analytics(sample_events, sample_tracks)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_match_analytics(analytics, path)
            loaded = load_match_analytics(path)
            assert "player_stats" in loaded
            assert "attributed_events" in loaded
        finally:
            os.unlink(path)
