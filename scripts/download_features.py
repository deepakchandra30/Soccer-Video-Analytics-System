#!/usr/bin/env python
"""Download SoccerNet features and optionally match videos."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.downloader import download_features, download_video


def main():
    parser = argparse.ArgumentParser(description="Download SoccerNet data")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--local-dir", default="data/")
    parser.add_argument("--video", action="store_true", help="Also download match videos (requires NDA password)")
    parser.add_argument("--password", default="", help="SoccerNet NDA password for video downloads")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Would download splits {args.splits} to {args.local_dir}")
        if args.video:
            print("Would also download videos")
        return

    print(f"Downloading features for {args.splits} into {args.local_dir}...")
    download_features(local_directory=args.local_dir, splits=args.splits)
    print("Features done.")

    if args.video:
        if not args.password:
            print("ERROR: --password required for video downloads.")
            print("Get your password by submitting the NDA at https://www.soccer-net.org/data")
            sys.exit(1)
        print(f"Downloading videos for {args.splits}...")
        download_video(local_directory=args.local_dir, password=args.password, splits=args.splits)
        print("Videos done.")

    print("All downloads complete.")


if __name__ == "__main__":
    main()
