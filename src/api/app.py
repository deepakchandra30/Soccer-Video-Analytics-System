"""FastAPI application serving real SoccerNet data."""
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

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
        # Guard against partial downloads: SoccerNet downloader writes
        # directly to the destination path, so an in-progress 50 MB file
        # would otherwise enable the Analyze button and crash the WebSocket
        # processor when opencv tries to read past EOF. A complete 90-min
        # half is ~1 GB; 200 MB is a safe lower bound.
        first_half = match_dir / "1_720p.mkv"
        second_half = match_dir / "2_720p.mkv"
        has_video = (
            first_half.exists() and second_half.exists()
            and first_half.stat().st_size > 200 * 1024 * 1024
            and second_half.stat().st_size > 200 * 1024 * 1024
        )

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
    """Load tracking data from tracks.json. Returns [] when not yet computed.

    Bug-fix 2026-04-10: previously generated synthetic random-jittered
    tracking dicts when tracks.json was missing, masking ML pipeline
    failures behind realistic-looking fake data. Now returns an empty
    list so the absence of real tracks is visible to clients.
    """
    match_dir = _safe_match_dir(match_id)
    tracks_path = match_dir / "tracks.json"
    if not tracks_path.exists():
        return []
    with open(tracks_path) as f:
        return json.load(f)


def _generate_narrative(match_id: str, events):
    """Return a real LLM-generated narrative or 503 (no fake fallback).

    Resolution order:
      1. Pre-computed narrative.json on disk (offline cache).
      2. Live OpenAI call when OPENAI_API_KEY is set.
      3. HTTP 503 — explicitly NOT a hard-coded summary. The deterministic
         "competitive match between..." string the client flagged on
         2026-04-10 produced text that read like an LLM narrative without
         any real grounding, which is worse than failing loudly.
    """
    match_dir = _safe_match_dir(match_id)

    narrative_path = match_dir / "narrative.json"
    if narrative_path.exists():
        with open(narrative_path) as f:
            return json.load(f)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "narrative_service_unavailable",
                "reason": "OPENAI_API_KEY is not configured and no narrative.json is cached.",
                "hint": "Set OPENAI_API_KEY in the environment, or place a narrative.json next to Labels-v3.json.",
            },
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "Summarise this football match. Every claim must come from the events list. "
            "Return JSON with keys: match_summary, key_moments, player_contributions, tactical_breakdown. "
            f"Events: {json.dumps(events)}"
        )
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a football analyst. Reply with JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "narrative_service_unavailable",
                "reason": f"LLM call failed: {type(exc).__name__}",
            },
        )


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
async def get_tracks(match_id: str, stride: int = Query(1, ge=1)):
    """Return per-frame tracking data with optional stride for downsampling."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    tracks = await asyncio.to_thread(_load_tracks, match_id)
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
async def get_narratives(match_id: str):
    """Return real LLM-generated match narrative or 503 (never a fake heuristic)."""
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    events = await asyncio.to_thread(_load_events, match_id)
    return await asyncio.to_thread(_generate_narrative, match_id, events)


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
async def get_video(match_id: str, request: Request, half: int = Query(1, ge=1, le=2)):
    """Serve match video with HTTP Range support for seeking/scrubbing.

    Bug-fix 2026-04-10: previously returned a plain FileResponse with
    just an Accept-Ranges header attached, which never produces 206
    Partial Content responses. The browser's <video> element needs proper
    206 + Content-Range to seek through the stream. This handler now
    parses the Range header per RFC 7233 and streams the requested byte
    slice back as 206; absence of Range falls back to full-file 200.
    """
    match_dir = _safe_match_dir(match_id)
    if not match_dir.exists():
        raise HTTPException(404, f"Match not found: {match_id}")

    video_file = f"{half}_720p.mkv"
    video_path = match_dir / video_file
    if not video_path.exists():
        raise HTTPException(404, f"Video not available: {video_file}")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            str(video_path),
            media_type="video/x-matroska",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    range_spec = range_header.strip().replace("bytes=", "")
    if range_spec.startswith("-"):
        suffix_length = int(range_spec[1:])
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    else:
        parts = range_spec.split("-", 1)
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

    if start >= file_size or start < 0 or end < start:
        raise HTTPException(
            status_code=416,
            detail="Range Not Satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    def iter_range():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                read_size = min(65536, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type="video/x-matroska",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
    )


@app.get("/health")
async def health_check():
    """Cheap health check — never triggers a filesystem scan.

    Bug-fix 2026-04-10: previously called _scan_matches() which rglob'd the
    entire DATA_DIR on every hit. Synchronous + heavy = blocked the event
    loop for every other client. Now returns a constant payload; the
    /matches endpoint exists for the (more expensive) full scan.
    """
    return {"status": "ok", "version": app.version, "realtime": True}
