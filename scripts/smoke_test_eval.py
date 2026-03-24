#!/usr/bin/env python
"""Smoke test the evaluation harness against the NetVLAD baseline."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.seeds import set_seeds
from src.evaluation.evaluate import run_evaluation_smoke_test


def main():
    parser = argparse.ArgumentParser(description="Validate eval harness vs NetVLAD baseline")
    parser.add_argument("--soccernet-path", default="data/")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--expected-map", type=float, default=49.7)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    set_seeds(42)

    try:
        avg_map = run_evaluation_smoke_test(
            soccernet_path=args.soccernet_path,
            predictions_path=args.predictions_path,
            expected_map=args.expected_map,
            tolerance=args.tolerance,
        )
        print(f"PASS — avg-mAP = {avg_map:.2f}%")
    except AssertionError as e:
        print(f"FAIL — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
