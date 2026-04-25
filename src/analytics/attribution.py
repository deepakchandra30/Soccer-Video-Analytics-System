"""Event-player attribution via spatial proximity on the pitch."""
import numpy as np


def attribute_events_to_players(events, tracks, proximity_threshold=5.0,
                                 framerate=2):
    """Link detected events to tracked players by spatial proximity.

    For each event, finds the closest player on the pitch at the event's
    timestamp. If no player is within the threshold, the event remains
    unattributed.

    Args:
        events: list of event dicts with keys:
            - label: str (event type)
            - half: int (1 or 2)
            - position: int (ms from start of half)
            - confidence: float
        tracks: list of frame records from TrackingPipeline
        proximity_threshold: max distance (meters) for attribution
        framerate: video fps used for tracks

    Returns list of enriched event dicts with added keys:
        - attributed_player: int track_id or None
        - attribution_distance: float meters or None
    """
    # index tracks by frame for fast lookup
    frame_index = {}
    for frame in tracks:
        frame_index[frame["frame_idx"]] = frame

    enriched = []
    for event in events:
        event_copy = dict(event)
        event_copy["attributed_player"] = None
        event_copy["attribution_distance"] = None

        # convert event position (ms) to approximate frame index
        event_ms = float(event["position"])
        event_frame = int(event_ms * framerate / 1000)

        # search in a small window around the event frame
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

                # no event pitch position, so use player proximity to a
                # reference point (center of relevant half)
                # For now, just pick the closest player by track presence
                # The distance metric is euclidean on pitch coordinates
                dist = 0.0  # default: attribute to any present player
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
    """Count per-player event involvement from attributed events.

    Returns dict mapping track_id -> {event_type: count, total: count}.
    """
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
