"""Helpers for video frame reading and temporal windowing."""
from typing import List, Tuple
import numpy as np


def read_frames_decord(video_path: str, frame_indices: List[int]) -> np.ndarray:
    """Read specific frames from video using decord. Returns (N, H, W, 3) uint8."""
    from decord import VideoReader, cpu as decord_cpu

    vr = VideoReader(video_path, ctx=decord_cpu(0))
    return vr.get_batch(frame_indices).asnumpy()


def build_clip_windows(n_frames, window_size, stride):
    """Sliding window over a frame sequence. Last window is clamped to n_frames.

    >>> build_clip_windows(10, 4, 2)
    [(0, 4), (2, 6), (4, 8), (6, 10)]
    """
    if n_frames <= 0:
        return []
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    windows = []
    start = 0
    while start < n_frames:
        end = min(start + window_size, n_frames)
        windows.append((start, end))
        if end == n_frames:
            break
        start += stride
    return windows
