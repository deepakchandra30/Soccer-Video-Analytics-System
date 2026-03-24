"""Per-player analytics: heatmaps, screen-time, event attribution."""
import json
import os
from collections import defaultdict

import numpy as np

from src.tracking.homography import PITCH_LENGTH, PITCH_WIDTH


def compute_player_stats(tracks, fps=25):
    """Per-player screen-time and position stats from frame tracks."""
    player_frames = defaultdict(list)
    player_positions = defaultdict(list)

    for frame in tracks:
        for player in frame.get("players", []):
            tid = player["track_id"]
            if tid < 0:
                continue
            player_frames[tid].append(frame["frame_idx"])
            if player.get("pitch_xy") is not None:
                player_positions[tid].append(player["pitch_xy"])

    stats = {}
    for tid in player_frames:
        n_frames = len(player_frames[tid])
        positions = np.array(player_positions.get(tid, []))
        stats[tid] = {
            "track_id": tid,
            "screen_time_frames": n_frames,
            "screen_time_seconds": round(n_frames / max(fps, 1), 1),
            "num_positions": len(positions),
            "mean_position": positions.mean(axis=0).tolist()
                             if len(positions) > 0 else None,
        }

    return stats


def compute_heatmap(tracks, resolution=(105, 68), sigma=2.0):
    """Gaussian-smoothed heatmap of player positions on pitch."""
    w_bins, h_bins = resolution
    heatmap = np.zeros((h_bins, w_bins), dtype=np.float32)

    half_w = PITCH_LENGTH / 2
    half_h = PITCH_WIDTH / 2

    for frame in tracks:
        for player in frame.get("players", []):
            pos = player.get("pitch_xy")
            if pos is None:
                continue
            x, y = pos
            bx = int((x + half_w) / PITCH_LENGTH * (w_bins - 1))
            by = int((y + half_h) / PITCH_WIDTH * (h_bins - 1))
            if 0 <= bx < w_bins and 0 <= by < h_bins:
                heatmap[by, bx] += 1

    if sigma > 0 and heatmap.sum() > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap


def compute_player_heatmap(tracks, track_id, resolution=(105, 68), sigma=2.0):
    """Generate heatmap for a single player."""
    w_bins, h_bins = resolution
    heatmap = np.zeros((h_bins, w_bins), dtype=np.float32)
    half_w = PITCH_LENGTH / 2
    half_h = PITCH_WIDTH / 2

    for frame in tracks:
        for player in frame.get("players", []):
            if player["track_id"] != track_id:
                continue
            pos = player.get("pitch_xy")
            if pos is None:
                continue
            x, y = pos
            bx = int((x + half_w) / PITCH_LENGTH * (w_bins - 1))
            by = int((y + half_h) / PITCH_WIDTH * (h_bins - 1))
            if 0 <= bx < w_bins and 0 <= by < h_bins:
                heatmap[by, bx] += 1

    if sigma > 0 and heatmap.sum() > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap


def save_analytics(stats, heatmap, output_path):
    """Save analytics to JSON + NPY files."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    json_path = output_path if output_path.endswith(".json") else output_path + ".json"
    serializable_stats = {str(k): v for k, v in stats.items()}
    with open(json_path, "w") as f:
        json.dump({"player_stats": serializable_stats}, f, indent=2)
    npy_path = json_path.replace(".json", "_heatmap.npy")
    np.save(npy_path, heatmap)
