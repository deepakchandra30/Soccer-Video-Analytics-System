"""Run match analytics and narrative generation."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.match_stats import compute_match_analytics, save_match_analytics
from src.narratives.generator import NarrativeGenerator, save_narrative
from src.narratives.factcheck import validate_narrative, compute_factcheck_score


def main():
    parser = argparse.ArgumentParser(description="Run match analytics and narratives")
    parser.add_argument("--events-json", required=True,
                        help="Path to results_spotting.json (event predictions)")
    parser.add_argument("--tracks-json", required=True,
                        help="Path to tracks.json from tracking pipeline")
    parser.add_argument("--output-dir", default="outputs/analytics")
    parser.add_argument("--generate-narrative", action="store_true",
                        help="Generate LLM narrative (requires OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model for narrative generation")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.events_json) as f:
        events_data = json.load(f)
    events = events_data.get("predictions", [])

    with open(args.tracks_json) as f:
        tracks_data = json.load(f)
    tracks = tracks_data.get("tracks", [])

    print("Computing match analytics...")
    analytics = compute_match_analytics(events, tracks, fps=args.fps)
    analytics_path = os.path.join(args.output_dir, "analytics.json")
    save_match_analytics(analytics, analytics_path)
    print(f"Analytics saved to {analytics_path}")
    print(f"  {analytics['num_events']} events, "
          f"{analytics['num_attributed']} attributed, "
          f"{analytics['num_players_tracked']} players")

    if args.generate_narrative:
        print("Generating match narrative...")
        gen = NarrativeGenerator(model=args.model)
        narrative = gen.generate(analytics)
        narr_path = os.path.join(args.output_dir, "narrative.json")
        save_narrative(narrative, narr_path)
        print(f"Narrative saved to {narr_path}")

        print("Running fact-check...")
        checks = validate_narrative(narrative, analytics)
        score = compute_factcheck_score(checks)
        print(f"  Fact-check: {score['supported']}/{score['total_claims']} "
              f"claims supported ({score['score']:.0%})")

    print("Done.")


if __name__ == "__main__":
    main()
