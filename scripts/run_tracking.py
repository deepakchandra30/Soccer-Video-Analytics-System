"""Run player tracking pipeline on match videos."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.pipeline import TrackingPipeline
from src.tracking.analytics import compute_player_stats, compute_heatmap, save_analytics


def main():
    parser = argparse.ArgumentParser(description="Run player tracking on video")
    parser.add_argument("--video", required=True, help="Path to match video")
    parser.add_argument("--output-dir", default="outputs/tracking")
    parser.add_argument("--model", default="yolov8m.pt",
                        help="YOLOv8 model name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only first N frames (for testing)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Video FPS for screen-time calculation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = TrackingPipeline(device=args.device)
    print(f"Processing {args.video}...")
    tracks = pipeline.process_video(args.video, max_frames=args.max_frames)

    tracks_path = os.path.join(args.output_dir, "tracks.json")
    pipeline.save_tracks(tracks, tracks_path)
    print(f"Tracks saved to {tracks_path} ({len(tracks)} frames)")

    stats = compute_player_stats(tracks, fps=args.fps)
    heatmap = compute_heatmap(tracks)
    analytics_path = os.path.join(args.output_dir, "analytics.json")
    save_analytics(stats, heatmap, analytics_path)
    print(f"Analytics saved ({len(stats)} players tracked)")

    print(f"\nTracking Summary:")
    print(f"  Total frames: {len(tracks)}")
    print(f"  Unique players: {len(stats)}")
    for tid, s in sorted(stats.items(), key=lambda x: x[1]["screen_time_frames"],
                          reverse=True)[:10]:
        print(f"  Player #{tid}: {s['screen_time_seconds']}s screen time, "
              f"{s['num_positions']} positions")


if __name__ == "__main__":
    main()
