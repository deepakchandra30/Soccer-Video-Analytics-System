"""Per-match analytics aggregation."""
import json
import os

import numpy as np

from src.tracking.analytics import compute_player_stats, compute_heatmap
from src.analytics.attribution import attribute_events_to_players, compute_event_involvement


def compute_match_analytics(events, tracks, fps=25):
    """Compute full match analytics from events and tracks.

    Args:
        events: list of event prediction dicts
        tracks: list of frame records from TrackingPipeline
        fps: video frame rate

    Returns dict with:
        - player_stats: per-player screen-time and positions
        - event_involvement: per-player event counts
        - attributed_events: events with player attribution
        - heatmap_shape: tuple for the heatmap dimensions
    """
    # per-player stats from tracks
    player_stats = compute_player_stats(tracks, fps=fps)

    # attribute events to players
    attributed = attribute_events_to_players(events, tracks)

    # event involvement counts
    involvement = compute_event_involvement(attributed)

    # merge involvement into player stats
    for tid, inv in involvement.items():
        if tid in player_stats:
            player_stats[tid]["event_involvement"] = inv
        else:
            player_stats[tid] = {
                "track_id": tid,
                "screen_time_frames": 0,
                "screen_time_seconds": 0,
                "num_positions": 0,
                "mean_position": None,
                "event_involvement": inv,
            }

    # team heatmap
    heatmap = compute_heatmap(tracks)

    return {
        "player_stats": player_stats,
        "attributed_events": attributed,
        "event_involvement": involvement,
        "heatmap_shape": heatmap.shape,
        "num_events": len(events),
        "num_attributed": sum(1 for e in attributed
                              if e.get("attributed_player") is not None),
        "num_players_tracked": len(player_stats),
    }


def save_match_analytics(analytics, output_path):
    """Save analytics to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # make serializable
    serializable = {
        "player_stats": {str(k): v for k, v in analytics["player_stats"].items()},
        "attributed_events": analytics["attributed_events"],
        "event_involvement": {str(k): v for k, v in analytics["event_involvement"].items()},
        "summary": {
            "num_events": analytics["num_events"],
            "num_attributed": analytics["num_attributed"],
            "num_players_tracked": analytics["num_players_tracked"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_match_analytics(path):
    """Load analytics from JSON."""
    with open(path) as f:
        return json.load(f)
