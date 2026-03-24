#!/usr/bin/env python
"""Extract ResNet features from a match video and save as .npy."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet features from video")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not Path(args.video_path).exists():
        print(f"Video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    from config.seeds import set_seeds
    set_seeds(42)

    from src.features.extract import extract_features
    print(f"Extracting {args.model} @ {args.fps}fps from {args.video_path}")

    features = extract_features(
        video_path=args.video_path,
        output_path=args.output_path,
        model_name=args.model,
        fps=args.fps,
        batch_size=args.batch_size,
    )
    print(f"Saved {features.shape} to {args.output_path}")


if __name__ == "__main__":
    main()
