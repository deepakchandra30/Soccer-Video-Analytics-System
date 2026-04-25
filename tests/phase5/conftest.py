"""Test fixtures for FastAPI tests.

Creates a temporary match directory with synthetic Labels-v3.json,
tracks.json, and feature .npy files, then points the API's DATA_DIR at
the temp path so the tests do not depend on a real SoccerNet download.
"""
import json

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def setup_test_data(tmp_path, monkeypatch):
    import src.api.app as app_module

    match_dir = tmp_path / "england_epl" / "2014-2015" / "match_001"
    match_dir.mkdir(parents=True)

    labels = {
        "actions": {
            "0": {
                "imageMetadata": {
                    "label": "Goal",
                    "half": 1,
                    "position": 120000,
                    "gameTime": "1 - 02:00",
                    "visibility": "visible",
                    "team": "home",
                }
            },
            "1": {
                "imageMetadata": {
                    "label": "Foul",
                    "half": 1,
                    "position": 300000,
                    "gameTime": "1 - 05:00",
                    "visibility": "visible",
                    "team": "away",
                }
            },
            "2": {
                "imageMetadata": {
                    "label": "Corner",
                    "half": 2,
                    "position": 600000,
                    "gameTime": "2 - 10:00",
                    "visibility": "visible",
                    "team": "home",
                }
            },
        }
    }
    (match_dir / "Labels-v3.json").write_text(json.dumps(labels))

    tracks = []
    for i in range(100):
        players = [
            {
                "track_id": pid,
                "bbox": [100.0 + pid * 50, 200.0, 150.0 + pid * 50, 300.0],
                "confidence": 0.9,
            }
            for pid in range(5)
        ]
        tracks.append({"frame_idx": i, "players": players})
    (match_dir / "tracks.json").write_text(json.dumps(tracks))

    feat = np.random.default_rng(42).standard_normal((50, 512)).astype(np.float32)
    np.save(str(match_dir / "1_ResNET_TF2_PCA512.npy"), feat)
    np.save(str(match_dir / "2_ResNET_TF2_PCA512.npy"), feat)

    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path.resolve())
    yield
