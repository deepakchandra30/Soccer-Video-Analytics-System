import json
import asyncio
import os
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.realtime.session import SessionManager

router = APIRouter()

DATA_DIR = Path(os.environ.get("SOCCER_DATA_DIR", "data/")).resolve()

# Maximum time (in seconds) to allow a single frame-processing call to run
# before treating it as hung. YOLOv8 on CPU can take 5+ s per frame, so give
# generous headroom.
FRAME_PROCESS_TIMEOUT = 30.0


@router.websocket("/ws/realtime/{match_id:path}")
async def realtime_analytics(websocket: WebSocket, match_id: str, half: int = 1):
    await websocket.accept()

    video_path = DATA_DIR / match_id / f"{half}_720p.mkv"
    if not video_path.exists():
        await websocket.send_json(
            {"type": "error", "message": f"Video not found: {video_path}"}
        )
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    device = os.environ.get("SOCCER_DEVICE", "cpu")

    try:
        mgr = SessionManager()
        session = await asyncio.wait_for(
            loop.run_in_executor(
                None, mgr.get_or_create, match_id, str(video_path), device
            ),
            timeout=FRAME_PROCESS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        await websocket.send_json(
            {"type": "error", "message": "Timed out creating session"}
        )
        await websocket.close()
        return
    except Exception as e:
        await websocket.send_json(
            {"type": "error", "message": f"Failed to create session: {e}"}
        )
        await websocket.close()
        return

    try:
        video_info = await asyncio.wait_for(
            loop.run_in_executor(None, session.processor.get_video_info),
            timeout=FRAME_PROCESS_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception) as e:
        await websocket.send_json(
            {"type": "error", "message": f"Failed to get video info: {e}"}
        )
        await websocket.close()
        return

    await websocket.send_json({"type": "connected", "video_info": video_info})

    play_task = None

    async def _stream_frames(start_ms: int, speed: float):
        """Stream frame results at the video's fps rate, adjusted by speed."""
        fps = video_info.get("fps", 30.0)
        duration_ms = video_info.get("duration_ms", 0)
        frame_interval = 1.0 / (fps * speed) if speed > 0 else 1.0 / fps
        frame_duration_ms = 1000.0 / fps
        current_ms = start_ms

        try:
            while current_ms <= duration_ms:
                try:
                    data = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, session.process_at_time, current_ms
                        ),
                        timeout=FRAME_PROCESS_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Frame processing timed out at {current_ms} ms",
                        }
                    )
                    return
                except Exception as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Frame processing error at {current_ms} ms: {e}",
                        }
                    )
                    return

                await websocket.send_json({"type": "frame_result", "data": data})
                current_ms += frame_duration_ms
                await asyncio.sleep(frame_interval)
        except asyncio.CancelledError:
            # Playback was cancelled (pause / seek / new play / disconnect).
            # Let the cancellation propagate so the task finalises cleanly.
            raise

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "seek":
                # Cancel any ongoing playback
                if play_task and not play_task.done():
                    play_task.cancel()

                timestamp_ms = msg.get("timestamp_ms", 0)
                try:
                    data = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, session.process_at_time, timestamp_ms
                        ),
                        timeout=FRAME_PROCESS_TIMEOUT,
                    )
                    await websocket.send_json({"type": "frame_result", "data": data})
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Frame processing timed out at {timestamp_ms} ms",
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Seek error at {timestamp_ms} ms: {e}",
                        }
                    )

            elif msg_type == "play":
                # Cancel any ongoing playback before starting a new one
                if play_task and not play_task.done():
                    play_task.cancel()

                start_ms = msg.get("start_ms", 0)
                speed = msg.get("speed", 1.0)
                play_task = asyncio.create_task(
                    _stream_frames(start_ms, speed)
                )

            elif msg_type == "pause":
                if play_task and not play_task.done():
                    play_task.cancel()

            elif msg_type == "get_stats":
                try:
                    stats = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, session.get_running_stats
                        ),
                        timeout=FRAME_PROCESS_TIMEOUT,
                    )
                    await websocket.send_json({"type": "stats", "data": stats})
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {"type": "error", "message": "Timed out fetching stats"}
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": f"Stats error: {e}"}
                    )

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        # Do NOT close the session; the client may reconnect.
        if play_task and not play_task.done():
            play_task.cancel()
