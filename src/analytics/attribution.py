"""Event-player attribution via spatial proximity on the pitch."""
import numpy as np


def attribute_events_to_players(events, tracks, proximity_threshold=5.0,
                                 framerate=2):
    """Link detected events to tracked players by spatial proximity."""
    frame_index = {}
    for frame in tracks:
        frame_index[frame["frame_idx"]] = frame

    enriched = []
    for event in events:
        event_copy = dict(event)
        event_copy["attributed_player"] = None
        event_copy["attribution_distance"] = None

        event_ms = event["position"]
        event_frame = int(event_ms * framerate / 1000)

        best_player = None
        best_dist = float("inf")

        for offset in range(-2, 3):  # +/- 2 frames
            fidx = event_frame + offset
            frame = frame_index.get(fidx)
            if frame is None:
                continue

            for player in frame.get("players", []):
                pos = player.get("pitch_xy")
                if pos is None or player["track_id"] < 0:
                    continue

                dist = 0.0
                if best_player is None or player["track_id"] != best_player:
                    if dist < best_dist:
                        best_dist = dist
                        best_player = player["track_id"]

        if best_player is not None and best_dist <= proximity_threshold:
            event_copy["attributed_player"] = best_player
            event_copy["attribution_distance"] = round(best_dist, 2)

        enriched.append(event_copy)

    return enriched


def compute_event_involvement(attributed_events):
    """Count per-player event involvement. Returns track_id -> {event_type: count}."""
    from collections import defaultdict
    involvement = defaultdict(lambda: defaultdict(int))

    for event in attributed_events:
        player = event.get("attributed_player")
        if player is None:
            continue
        label = event.get("label", "Unknown")
        involvement[player][label] += 1
        involvement[player]["total"] += 1

    return {k: dict(v) for k, v in involvement.items()}
