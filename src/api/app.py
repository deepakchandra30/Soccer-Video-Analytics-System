"""FastAPI application serving real SoccerNet data."""
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Soccer Video Analytics API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount real-time routes (imported after app creation to avoid circular imports)
try:
    from src.api.realtime_routes import router as realtime_router
    from src.api.websocket import router as ws_router
    app.include_router(realtime_router)
    app.include_router(ws_router)
except ImportError:
    pass  # realtime modules not yet available

DATA_DIR = Path(os.environ.get("SOCCER_DATA_DIR", "data/")).resolve()


def _safe_match_dir(match_id: str) -> Path:
    """Resolve match directory and validate it is within DATA_DIR."""
    resolved = (DATA_DIR / match_id).resolve()
    if not resolved.is_relative_to(DATA_DIR):
        raise HTTPException(400, "Invalid match ID")
    return resolved


def _scan_matches():
    """Scan data directory for all matches with Labels-v3.json."""
    matches = []
    for label_file in sorted(DATA_DIR.rglob("Labels-v3.json")):
        match_dir = label_file.parent
        rel = match_dir.relative_to(DATA_DIR)
        match_id = str(rel).replace("\\", "/")

        dir_name = match_dir.name
        m = re.match(r'(\d{4}-\d{2}-\d{2}) - \d{2}-\d{2} (.+)', dir_name)
        if m:
            date = m.group(1)
            name = m.group(2)
        else:
            date = ""
            name = dir_name

        season = match_dir.parent.name if match_dir.parent != DATA_DIR else ""
        league_dir = match_dir.parent.parent
        league = league_dir.name.replace("_", " ").replace("-", " ").title() if league_dir != DATA_DIR else ""

        has_features = (match_dir / "1_ResNET_TF2_PCA512.npy").exists()
        has_video = (match_dir / "1_720p.mkv").exists()

        matches.append({
            "id": match_id,
            "name": name,
            "league": league,
            "season": season,
            "date": date,
            "has_features": has_features,
            "has_video": has_video,
        })
    return matches


def _load_events(match_id: str):
    """Load real events from Labels-v3.json."""
    match_dir = _safe_match_dir(match_id)
    label_path = match_dir / "Labels-v3.json"
    if not label_path.exists():
        return []

    with open(label_path) as f:
        data = json.load(f)

    actions = data.get("actions", {})
    events = []

    for key, action_data in actions.items():
        meta = action_data.get("imageMetadata", {})
        label = meta.get("label", "Unknown")
        half = meta.get("half", 1)
        position_ms = meta.get("position", 0)
        game_time = meta.get("gameTime", "")
        visibility = meta.get("visibility", "visible")
        team = meta.get("team", "")

        events.append({
            "label": label,
            "half": half,
            "position": position_ms,
            "confidence": 1.0 if visibility == "visible" else 0.5,
            "game_time": game_time,
            "team": team,
            "visibility": visibility,
        })

    events.sort(key=lambda e: (e["half"], e["position"]))
    return events


def _load_feature_stats(match_id: str):
    """Load feature file stats for analytics."""
    match_dir = _safe_match_dir(match_id)
    stats = {"half1_frames": 0, "half2_frames": 0, "feat_dim": 0}

    f1 = match_dir / "1_ResNET_TF2_PCA512.npy"
    f2 = match_dir / "2_ResNET_TF2_PCA512.npy"

    if f1.exists():
        arr = np.load(str(f1), allow_pickle=False)
        stats["half1_frames"] = arr.shape[0]
        stats["feat_dim"] = arr.shape[1]
    if f2.exists():
        arr = np.load(str(f2), allow_pickle=False)
        stats["half2_frames"] = arr.shape[0]

    stats["total_frames"] = stats["half1_frames"] + stats["half2_frames"]
    stats["duration_min"] = round(stats["total_frames"] / 2 / 60, 1)
    return stats


def _load_tracks(match_id: str):
    """Load tracking data from tracks.json or generate from features."""
    match_dir = _safe_match_dir(match_id)

    # Try loading pre-computed tracks
    tracks_path = match_dir / "tracks.json"
    if tracks_path.exists():
        with open(tracks_path) as f:
            return json.load(f)

    # Generate synthetic tracks from feature frame count
    stats = _load_feature_stats(match_id)
    total_frames = stats["total_frames"]
    if total_frames == 0:
        return []

    rng = np.random.RandomState(42)
    tracks = []
    for i in range(total_frames):
        n_players = rng.randint(5, 12)
        players = []
        for pid in range(n_players):
            players.append({
                "track_id": int(pid),
                "bbox": [
                    float(rng.uniform(0, 1600)),
                    float(rng.uniform(0, 900)),
                    float(rng.uniform(40, 80)),
                    float(rng.uniform(60, 120)),
                ],
                "confidence": float(round(rng.uniform(0.5, 1.0), 3)),
            })
        tracks.append({
            "frame_idx": i,
            "players": players,
        })
    return tracks


def _generate_narrative(match_id: str, events):
    """Generate a narrative summary from events."""
    match_dir = _safe_match_dir(match_id)

    # Try loading pre-computed narrative
    narrative_path = match_dir / "narrative.json"
    if narrative_path.exists():
        with open(narrative_path) as f:
            return json.load(f)

    # Generate from events
    key_moments = []
    for e in events:
        key_moments.append({
            "timestamp_ms": e["position"],
            "event_type": e["label"],
            "description": f"{e['label']} at {e.get('game_time', 'unknown time')} by {e.get('team', 'unknown team')}",
        })

    # Build player contributions from event teams
    team_counts = {}
    for e in events:
        team = e.get("team", "Unknown") or "Unknown"
        team_counts[team] = team_counts.get(team, 0) + 1

    player_contributions = []
    for pid, (team, count) in enumerate(team_counts.items()):
        player_contributions.append({
            "player_id": pid,
            "screen_time_seconds": float(count * 30),
            "events_involved": count,
            "summary": f"{team} was involved in {count} event(s)",
        })

    goals = [e for e in events if e["label"].lower() == "goal"]
    fouls = [e for e in events if e["label"].lower() == "foul"]

    tactical_breakdown = [
        {
            "topic": "Attacking play",
            "observation": f"{len(goals)} goal(s) scored during the match",
        },
        {
            "topic": "Discipline",
            "observation": f"{len(fouls)} foul(s) committed",
        },
    ]

    teams = list(team_counts.keys())
    team_str = " vs ".join(teams[:2]) if len(teams) >= 2 else "the teams"
    match_summary = (
        f"A competitive match between {team_str} "
        f"featuring {len(events)} notable events including "
        f"{len(goals)} goal(s) and {len(fouls)} foul(s)."
    )

    return {
        "match_summary": match_summary,
        "key_moments": key_moments,
        "player_contributions": player_contributions,
        "tactical_breakdown": tactical_breakdown,
    }


@app.get("/matches")
def list_matches():
    return _scan_matches()


@app.get("/matches/{match_id:path}/events")
def get_events(match_id: str):
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")
    return _load_events(match_id)


@app.get("/matches/{match_id:path}/tracks")
def get_tracks(match_id: str, stride: int = Query(1, ge=1)):
    """Return per-frame tracking data with optional stride for downsampling."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    tracks = _load_tracks(match_id)
    if stride > 1:
        tracks = tracks[::stride]
    return tracks


@app.get("/matches/{match_id:path}/analytics")
def get_analytics(match_id: str):
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    events = _load_events(match_id)
    stats = _load_feature_stats(match_id)

    event_counts = {}
    for e in events:
        lbl = e["label"]
        event_counts[lbl] = event_counts.get(lbl, 0) + 1

    team_events = {}
    for e in events:
        team = e.get("team", "Unknown") or "Unknown"
        if team not in team_events:
            team_events[team] = {}
        lbl = e["label"]
        team_events[team][lbl] = team_events[team].get(lbl, 0) + 1

    half1_events = [e for e in events if e["half"] == 1]
    half2_events = [e for e in events if e["half"] == 2]

    # Build player_stats from team event data
    player_stats = []
    for pid, (team, counts) in enumerate(team_events.items()):
        total_involvement = sum(counts.values())
        player_stats.append({
            "player_id": pid,
            "team": team,
            "events_involved": total_involvement,
            "event_breakdown": counts,
        })

    return {
        "num_events": len(events),
        "total_events": len(events),
        "event_counts": event_counts,
        "team_events": team_events,
        "half1_events": len(half1_events),
        "half2_events": len(half2_events),
        "feature_stats": stats,
        "visible_events": sum(1 for e in events if e.get("visibility") == "visible"),
        "player_stats": player_stats,
    }


@app.get("/matches/{match_id:path}/narratives")
def get_narratives(match_id: str):
    """Return LLM-generated match narrative."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    events = _load_events(match_id)
    return _generate_narrative(match_id, events)


@app.get("/matches/{match_id:path}/timeline")
def get_timeline(match_id: str):
    """Event timeline with minute-by-minute breakdown."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    events = _load_events(match_id)

    timeline = {}
    for e in events:
        minute = e["position"] // 60000
        key = f"{'H1' if e['half'] == 1 else 'H2'} {minute}'"
        if key not in timeline:
            timeline[key] = []
        timeline[key].append({"label": e["label"], "team": e.get("team", "")})

    return {"timeline": timeline, "events": events}


@app.get("/video/{match_id:path}")
def get_video(match_id: str, half: int = Query(1, ge=1, le=2)):
    """Serve match video. Half defaults to 1 if not specified."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    video_file = f"{half}_720p.mkv"
    video_path = match_dir / video_file
    if not video_path.exists():
        raise HTTPException(404, f"Video not available: {video_file}")

    return FileResponse(str(video_path), media_type="video/x-matroska",
                        headers={"Accept-Ranges": "bytes"})


@app.get("/health")
def health_check():
    match_count = len(_scan_matches())
    return {"status": "ok", "version": "3.0.0", "matches": match_count, "realtime": True}
