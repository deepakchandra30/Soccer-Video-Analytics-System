"""Real-time processing session management for match video analysis."""

import time
import threading
from typing import Dict, List, Optional

import cv2

from src.realtime.processor import RealtimeProcessor


class RealtimeSession:
    """Manages a real-time processing session for a single match.

    Wraps a RealtimeProcessor and maintains running statistics across
    all processed frames without re-scanning cached results.
    """

    def __init__(self, match_id: str, video_path: str, device: Optional[str] = None):
        self._match_id = match_id
        self._video_path = video_path
        self._processor = RealtimeProcessor(video_path, device=device)

        # Open the video to read fps for timestamp-to-frame conversion.
        cap = cv2.VideoCapture(str(video_path))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Lock protecting the mutable stat counters below.  _update_stats can
        # be called from multiple threads (e.g. WebSocket handler running in
        # an executor), so every read/write of the counters goes through this.
        self._stats_lock = threading.Lock()

        # Running aggregate counters (avoid re-scanning all cached frames).
        self._frames_processed: int = 0
        self._total_players_detected: int = 0
        self._total_confidence_sum: float = 0.0
        self._total_detections_with_conf: int = 0
        self._camera_cuts_detected: int = 0
        self._unique_track_ids: set = set()
        self._total_processing_time: float = 0.0

        # Per-player screen time: track_id -> number of frames seen.
        self._player_frame_counts: Dict[int, int] = {}

    @property
    def match_id(self) -> str:
        """Return the match identifier for this session."""
        return self._match_id

    @property
    def processor(self) -> RealtimeProcessor:
        """Return the underlying processor."""
        return self._processor

    def process_at_time(self, timestamp_ms: int) -> dict:
        """Process the frame at a given timestamp in milliseconds.

        Converts the timestamp to a frame index based on the video fps,
        clamps it to the valid range, then delegates to process_at_frame.
        """
        frame_idx = int(timestamp_ms * self._fps / 1000.0)
        return self.process_at_frame(frame_idx)

    def process_at_frame(self, frame_idx: int) -> dict:
        """Process a specific frame and update running statistics.

        The *frame_idx* is clamped to ``[0, total_frames - 1]`` so callers
        never trigger an ``IndexError`` from out-of-range values.

        Returns the frame result dict from the processor.
        """
        # Clamp to valid range to avoid IndexError.
        if self._total_frames > 0:
            frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        else:
            frame_idx = max(0, frame_idx)

        start = time.perf_counter()
        result = self._processor.process_frame(frame_idx)
        elapsed = time.perf_counter() - start

        self._update_stats(result, elapsed)
        return result

    def _update_stats(self, result: dict, elapsed: float) -> None:
        """Incrementally update running aggregates from a single frame result.

        This method is safe to call from multiple threads; all counter
        mutations are serialised through ``self._stats_lock``.
        """
        with self._stats_lock:
            self._frames_processed += 1
            self._total_processing_time += elapsed

            players = result.get("players", [])
            num_players = len(players)
            self._total_players_detected += num_players

            for player in players:
                conf = player.get("confidence", 0.0)
                self._total_confidence_sum += conf
                self._total_detections_with_conf += 1

                track_id = player.get("track_id")
                if track_id is not None and track_id != -1:
                    self._unique_track_ids.add(track_id)
                    self._player_frame_counts[track_id] = (
                        self._player_frame_counts.get(track_id, 0) + 1
                    )

            if result.get("is_camera_cut", False):
                self._camera_cuts_detected += 1

    def get_running_stats(self) -> dict:
        """Return aggregated statistics from all processed frames so far.

        Returns:
            Dict with keys: frames_processed, total_players_detected,
            avg_players_per_frame, avg_confidence, camera_cuts_detected,
            unique_track_ids, processing_fps.
        """
        with self._stats_lock:
            frames = self._frames_processed
            return {
                "frames_processed": frames,
                "total_players_detected": self._total_players_detected,
                "avg_players_per_frame": (
                    self._total_players_detected / frames if frames > 0 else 0.0
                ),
                "avg_confidence": (
                    self._total_confidence_sum / self._total_detections_with_conf
                    if self._total_detections_with_conf > 0
                    else 0.0
                ),
                "camera_cuts_detected": self._camera_cuts_detected,
                "unique_track_ids": len(self._unique_track_ids),
                "processing_fps": (
                    frames / self._total_processing_time
                    if self._total_processing_time > 0
                    else 0.0
                ),
            }

    def get_player_stats(self) -> Dict[int, dict]:
        """Return per-player screen time statistics.

        Returns:
            Dict mapping track_id to a dict with:
                - frames_seen: number of frames this player appeared in
                - screen_time_ratio: fraction of processed frames containing
                  this player
        """
        with self._stats_lock:
            frames = self._frames_processed
            return {
                track_id: {
                    "frames_seen": count,
                    "screen_time_ratio": count / frames if frames > 0 else 0.0,
                }
                for track_id, count in self._player_frame_counts.items()
            }

    def close(self) -> None:
        """Release resources held by this session."""
        self._processor.close()


class SessionManager:
    """Manages multiple RealtimeSession instances (singleton-like).

    Provides get-or-create semantics so callers can retrieve an existing
    session for a match without tracking lifecycle themselves.

    The singleton is implemented via ``__new__`` + ``__init__`` with a
    ``_initialized`` guard so that ``_sessions`` is always set exactly once
    and the instance is safe to pickle and to patch in tests.
    """

    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SessionManager":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        # Guard so repeated __init__ calls (which Python makes every time
        # the constructor is called) do not reset the sessions dict.
        if self._initialized:
            return
        self._sessions: Dict[str, RealtimeSession] = {}
        self._initialized = True

    def get_or_create(
        self, match_id: str, video_path: str, device: Optional[str] = None
    ) -> RealtimeSession:
        """Return an existing session for match_id, or create a new one.

        Thread-safe: the lookup-and-create is protected by ``_lock`` so
        concurrent requests for the same *match_id* will not create
        duplicate sessions.
        """
        with self._lock:
            if match_id not in self._sessions:
                self._sessions[match_id] = RealtimeSession(
                    match_id, video_path, device=device
                )
            return self._sessions[match_id]

    def get_session(self, match_id: str) -> Optional[RealtimeSession]:
        """Return an existing session for match_id, or None if not found."""
        return self._sessions.get(match_id)

    def close_session(self, match_id: str) -> None:
        """Close and remove the session for the given match_id."""
        session = self._sessions.pop(match_id, None)
        if session is not None:
            session.close()

    def list_sessions(self) -> List[str]:
        """Return a list of active session match_ids."""
        return list(self._sessions.keys())

    def close_all(self) -> None:
        """Close all active sessions and clear the registry."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
