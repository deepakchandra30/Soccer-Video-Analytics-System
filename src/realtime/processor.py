"""Real-time video frame processor for on-demand detection and tracking."""
import collections
import threading

import cv2
import numpy as np


class RealtimeProcessor:
    """Processes individual video frames on demand with detection and tracking.

    Opens a video file and lazily initializes YOLOv8 detection and ByteTrack
    tracking so that frames can be processed individually rather than in batch.
    Results are cached (up to 500 frames) so repeated access is instant.
    """

    _CACHE_LIMIT = 500

    def __init__(self, video_path, device=None):
        """Open a video file and read its metadata.

        Args:
            video_path: Path to the video file.
            device: Torch device string for YOLO inference (e.g. "cuda:0").
                    None lets the detector choose automatically.
        """
        self._video_path = str(video_path)
        self._device = device

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video file: {self._video_path}"
            )

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Lazy-init placeholders
        self._detector = None
        self._tracker = None

        # Track the last processed frame index so we can detect backward seeks
        # and reset the tracker to avoid stale track IDs.
        self._last_frame_idx = -1

        # Lock for thread-safe frame processing (the WebSocket endpoint may
        # call process_frame from a thread pool).
        self._lock = threading.Lock()

        # LRU-style frame cache: OrderedDict keeps insertion order so we can
        # efficiently evict the oldest entries when the limit is reached.
        self._cache = collections.OrderedDict()

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_models(self):
        """Initialise detector and tracker on first use."""
        if self._detector is None:
            from src.tracking.detector import PlayerDetector
            self._detector = PlayerDetector(device=self._device)

        if self._tracker is None:
            from src.tracking.tracker import PlayerTracker
            self._tracker = PlayerTracker(frame_rate=int(round(self.fps)))

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def _read_frame(self, frame_idx):
        """Seek to *frame_idx* and return the decoded BGR frame.

        Uses ``CAP_PROP_POS_FRAMES`` to seek first.  If the read fails (which
        can happen with certain codecs), the method falls back to reading
        sequentially from the current position until the target frame is
        reached.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(
                f"frame_idx {frame_idx} out of range "
                f"[0, {self.total_frames})"
            )

        # Primary path: seek directly.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if ret:
            return frame

        # Fallback: sequential read from the current position.  Re-open the
        # capture if the current position is already past the target.
        cur_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur_pos > frame_idx:
            self._cap.release()
            self._cap = cv2.VideoCapture(self._video_path)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to reopen video file: {self._video_path}"
                )
            cur_pos = 0

        frames_to_skip = frame_idx - cur_pos
        for _ in range(frames_to_skip):
            ret, _ = self._cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return frame

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_put(self, frame_idx, result):
        """Store a result in the cache, evicting the oldest if necessary."""
        if frame_idx in self._cache:
            # Move to end so it is treated as most-recently-used.
            self._cache.move_to_end(frame_idx)
            return
        self._cache[frame_idx] = result
        while len(self._cache) > self._CACHE_LIMIT:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_idx):
        """Detect and track players in a single frame.

        Args:
            frame_idx: Zero-based frame index to process.

        Returns:
            dict matching the TrackingPipeline per-frame format::

                {
                    "frame_idx": int,
                    "timestamp_ms": int,
                    "is_camera_cut": bool,
                    "num_players": int,
                    "players": [
                        {
                            "track_id": int,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float,
                            "pitch_xy": None,
                        },
                        ...
                    ],
                }
        """
        # Return cached result immediately if available.
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]

        with self._lock:
            # Double-check after acquiring the lock: another thread may have
            # populated the cache while we were waiting.
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]

            self._ensure_models()

            # Detect backward seeks and reset the tracker so that stale track
            # IDs from future frames do not leak into earlier ones.
            if frame_idx < self._last_frame_idx:
                from src.tracking.tracker import PlayerTracker
                self._tracker = PlayerTracker(
                    frame_rate=int(round(self.fps))
                )

            self._last_frame_idx = frame_idx

            frame = self._read_frame(frame_idx)
            detections = self._detector.detect(frame)
            tracked = self._tracker.update(frame, detections)

            timestamp_ms = (
                int(round(frame_idx / self.fps * 1000)) if self.fps else 0
            )

            boxes = tracked["boxes"]
            confidences = tracked["confidences"]
            tracker_ids = tracked["tracker_ids"]
            is_camera_cut = bool(tracked["is_camera_cut"])

            players = []
            for i in range(len(boxes)):
                # ByteTrack can return fewer tracker_ids than boxes; guard
                # against an IndexError by falling back to -1.
                tid = (
                    int(tracker_ids[i])
                    if i < len(tracker_ids)
                    else -1
                )
                players.append({
                    "track_id": tid,
                    "bbox": [float(v) for v in boxes[i]],
                    "confidence": float(confidences[i]),
                    "pitch_xy": None,
                })

            result = {
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "is_camera_cut": is_camera_cut,
                "num_players": len(players),
                "players": players,
            }

            self._cache_put(frame_idx, result)
            return result

    def process_range(self, start_frame, end_frame, stride=1):
        """Process a contiguous range of frames.

        Args:
            start_frame: First frame index (inclusive).
            end_frame: Last frame index (exclusive).
            stride: Step between frames (default 1).

        Returns:
            List of result dicts, one per processed frame.
        """
        results = []
        for idx in range(start_frame, end_frame, stride):
            results.append(self.process_frame(idx))
        return results

    def get_video_info(self):
        """Return a dict summarising the open video's properties.

        Returns:
            dict with keys fps, total_frames, width, height, duration_ms.
        """
        duration_ms = int(round(self.total_frames / self.fps * 1000)) if self.fps else 0
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_ms": duration_ms,
        }

    def close(self):
        """Release the video capture and clear the cache."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cache.clear()

    def __del__(self):
        """Release resources if the user forgot to call close()."""
        try:
            self.close()
        except Exception:
            # Suppress errors during interpreter shutdown when globals
            # may have already been torn down.
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
