"""Tests for tracking pipeline."""
import json
import os
import tempfile

import numpy as np
import pytest

from src.tracking.pipeline import TrackingPipeline


class TestTrackingPipeline:
    def test_save_and_load_tracks(self):
        tracks = [
            {"frame_idx": 0, "timestamp_ms": 0, "is_camera_cut": False,
             "num_players": 1,
             "players": [{"track_id": 1, "bbox": [100, 200, 150, 400],
                          "confidence": 0.9, "pitch_xy": [10.0, 5.0]}]},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline = TrackingPipeline()
            pipeline.save_tracks(tracks, path)
            loaded = TrackingPipeline.load_tracks(path)
            assert len(loaded) == 1
            assert loaded[0]["players"][0]["track_id"] == 1
        finally:
            os.unlink(path)

    def test_track_record_format(self):
        """Verify the expected track record structure."""
        record = {
            "frame_idx": 0,
            "timestamp_ms": 0,
            "is_camera_cut": False,
            "num_players": 2,
            "players": [
                {"track_id": 1, "bbox": [0, 0, 50, 100],
                 "confidence": 0.9, "pitch_xy": [0, 0]},
                {"track_id": 2, "bbox": [100, 0, 150, 100],
                 "confidence": 0.85, "pitch_xy": [10, 5]},
            ],
        }
        assert record["num_players"] == len(record["players"])
        for p in record["players"]:
            assert "track_id" in p
            assert "bbox" in p
            assert len(p["bbox"]) == 4
            assert "confidence" in p
            assert "pitch_xy" in p
