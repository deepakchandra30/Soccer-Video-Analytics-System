"""Shared fixtures for phase 4 analytics/narrative tests."""
import pytest


@pytest.fixture
def sample_events():
    """Mock event predictions."""
    return [
        {"label": "Goal", "half": 1, "position": 1423000, "confidence": 0.92},
        {"label": "Foul", "half": 1, "position": 930000, "confidence": 0.78},
        {"label": "Corner", "half": 2, "position": 2100000, "confidence": 0.65},
    ]


@pytest.fixture
def sample_tracks():
    """Minimal tracks with 2 players across 5 frames."""
    tracks = []
    for i in range(5):
        tracks.append({
            "frame_idx": i,
            "timestamp_ms": i * 40,
            "is_camera_cut": False,
            "num_players": 2,
            "players": [
                {"track_id": 7, "bbox": [100, 200, 150, 400],
                 "confidence": 0.9, "pitch_xy": [10.0 + i, 5.0]},
                {"track_id": 11, "bbox": [300, 180, 360, 410],
                 "confidence": 0.88, "pitch_xy": [-20.0, 10.0 + i]},
            ],
        })
    return tracks


@pytest.fixture
def sample_analytics(sample_events, sample_tracks):
    """Pre-computed analytics dict."""
    return {
        "player_stats": {
            7: {"track_id": 7, "screen_time_frames": 5,
                "screen_time_seconds": 0.2, "num_positions": 5,
                "mean_position": [12.0, 5.0]},
            11: {"track_id": 11, "screen_time_frames": 5,
                 "screen_time_seconds": 0.2, "num_positions": 5,
                 "mean_position": [-20.0, 12.0]},
        },
        "attributed_events": [
            {"label": "Goal", "half": 1, "position": 1423000,
             "confidence": 0.92, "attributed_player": 7,
             "attribution_distance": 2.1},
            {"label": "Foul", "half": 1, "position": 930000,
             "confidence": 0.78, "attributed_player": 11,
             "attribution_distance": 3.5},
            {"label": "Corner", "half": 2, "position": 2100000,
             "confidence": 0.65, "attributed_player": None,
             "attribution_distance": None},
        ],
        "event_involvement": {
            7: {"Goal": 1, "total": 1},
            11: {"Foul": 1, "total": 1},
        },
        "num_events": 3,
        "num_attributed": 2,
        "num_players_tracked": 2,
    }
