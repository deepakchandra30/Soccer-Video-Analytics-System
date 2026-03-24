import os
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.realtime.session import SessionManager

router = APIRouter(prefix="/realtime", tags=["realtime"])

DATA_DIR = Path(os.environ.get("SOCCER_DATA_DIR", "data/")).resolve()


@router.get("/status")
def list_sessions():
    """List all active processing sessions."""
    manager = SessionManager()
    sessions = []
    for match_id in manager.list_sessions():
        session = manager.get_session(match_id)
        if session is None:
            continue
        stats = session.get_running_stats()
        sessions.append({
            "match_id": match_id,
            "frames_processed": stats.get("frames_processed", 0),
        })
    return {"sessions": sessions}


@router.post("/start/{match_id:path}")
def start_session(match_id: str):
    """Start a processing session for a match."""
    match_id = urllib.parse.unquote(match_id)
    match_dir = DATA_DIR / match_id
    if not match_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Match directory not found: {match_id}")

    video_path = match_dir / "1_720p.mkv"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found for match: {match_id}")

    session = SessionManager().get_or_create(match_id, video_path)
    video_info = session.processor.get_video_info()

    return {"status": "started", "match_id": match_id, "video_info": video_info}


@router.post("/stop/{match_id:path}")
def stop_session(match_id: str):
    """Stop and close a processing session."""
    match_id = urllib.parse.unquote(match_id)
    if match_id not in SessionManager().list_sessions():
        raise HTTPException(status_code=404, detail=f"No active session for match: {match_id}")

    SessionManager().close_session(match_id)
    return {"status": "stopped", "match_id": match_id}


@router.get("/stats/{match_id:path}")
def get_stats(match_id: str):
    """Get running stats for a session."""
    match_id = urllib.parse.unquote(match_id)
    session = SessionManager().get_session(match_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active session for match: {match_id}")

    return session.get_running_stats()


@router.get("/frame/{match_id:path}")
def process_frame(
    match_id: str,
    timestamp_ms: Optional[int] = Query(None),
    frame_idx: Optional[int] = Query(None),
):
    """Process a single frame by timestamp or frame index."""
    match_id = urllib.parse.unquote(match_id)
    session = SessionManager().get_session(match_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active session for match: {match_id}")

    if timestamp_ms is None and frame_idx is None:
        raise HTTPException(status_code=400, detail="Provide either 'timestamp_ms' or 'frame_idx' query parameter")

    try:
        if timestamp_ms is not None:
            return session.process_at_time(timestamp_ms)
        return session.process_at_frame(frame_idx)
    except (IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Frame out of range: {exc}",
        ) from exc


@router.post("/process-range/{match_id:path}")
def process_range(
    match_id: str,
    start_ms: int = Query(...),
    end_ms: int = Query(...),
    stride: int = Query(1000),
):
    """Process a range of frames between start_ms and end_ms.

    The stride is in milliseconds (default 1000 = one frame per second).
    """
    match_id = urllib.parse.unquote(match_id)
    session = SessionManager().get_session(match_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active session for match: {match_id}")

    if start_ms >= end_ms:
        raise HTTPException(status_code=400, detail="'start_ms' must be less than 'end_ms'")

    if stride < 1:
        raise HTTPException(status_code=400, detail="'stride' must be at least 1")

    results = []
    current_ms = start_ms
    while current_ms <= end_ms:
        try:
            result = session.process_at_time(current_ms)
            results.append(result)
        except (IndexError, ValueError) as exc:
            results.append({"timestamp_ms": current_ms, "error": str(exc)})
        current_ms += stride

    return {"processed": len(results), "results": results}
