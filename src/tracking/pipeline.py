"""Full tracking pipeline: detect -> track -> project to pitch."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.tracking.detector import PlayerDetector
from src.tracking.tracker import PlayerTracker
from src.tracking.homography import PitchHomography


class TrackingPipeline:
    """Processes match video to produce per-frame player tracks with pitch coords.

    Output: list of per-frame track records, each containing player positions
    in both pixel and pitch coordinates.
    """
    def __init__(self, detector=None, tracker=None, homography=None,
                 device=None):
        self.detector = detector or PlayerDetector(device=device)
        self.tracker = tracker or PlayerTracker()
        self.homography = homography or PitchHomography()

    def process_video(self, video_path, homography_points=None,
                      max_frames=None, progress=True):
        """Run tracking on a video file.

        Args:
            video_path: path to .mkv or .mp4 video
            homography_points: tuple of (image_pts, pitch_pts) for projection
            max_frames: optional frame limit for testing
            progress: show tqdm progress bar

        Returns list of frame records:
            [{"frame_idx": int, "players": [{"track_id": int, "bbox": [4],
              "confidence": float, "pitch_xy": [2] or None}]}]
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        # estimate homography if points provided
        has_homography = False
        if homography_points is not None:
            img_pts, pitch_pts = homography_points
            has_homography = self.homography.estimate(img_pts, pitch_pts)

        self.tracker.reset()
        tracks = []
        iterator = range(total)
        if progress:
            iterator = tqdm(iterator, desc="Tracking")

        for frame_idx in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            # detect players
            dets = self.detector.detect(frame)

            # track with camera cut handling
            tracked = self.tracker.update(frame, dets)

            # build frame record
            players = []
            for i in range(len(tracked["boxes"])):
                bbox = tracked["boxes"][i].tolist()
                player = {
                    "track_id": int(tracked["tracker_ids"][i])
                                if i < len(tracked["tracker_ids"]) else -1,
                    "bbox": bbox,
                    "confidence": float(tracked["confidences"][i]),
                    "pitch_xy": None,
                }

                # project foot position to pitch if homography available
                if has_homography:
                    foot = PitchHomography.bbox_foot_position(
                        tracked["boxes"][i:i+1]
                    )
                    pitch_pos = self.homography.project_to_pitch(foot)
                    player["pitch_xy"] = pitch_pos[0].tolist()

                players.append(player)

            tracks.append({
                "frame_idx": frame_idx,
                "timestamp_ms": int(frame_idx * 1000 / fps) if fps > 0 else 0,
                "is_camera_cut": tracked["is_camera_cut"],
                "num_players": len(players),
                "players": players,
            })

        cap.release()
        return tracks

    def save_tracks(self, tracks, output_path):
        """Save tracks to JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"tracks": tracks, "num_frames": len(tracks)}, f)

    @staticmethod
    def load_tracks(path):
        """Load tracks from JSON."""
        with open(path) as f:
            data = json.load(f)
        return data["tracks"]
