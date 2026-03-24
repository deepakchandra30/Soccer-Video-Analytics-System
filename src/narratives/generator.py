"""LLM-powered match narrative generation."""
import json
import os

from src.narratives.schemas import MatchNarrative
from src.narratives.prompts import MATCH_SUMMARY_SYSTEM, MATCH_SUMMARY_USER


class NarrativeGenerator:
    """Generates structured match narratives using an OpenAI LLM."""
    def __init__(self, api_key=None, model="gpt-4o-mini", base_url=None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
        )
        self.model = model

    def generate(self, analytics_data):
        """Generate a match narrative from analytics data."""
        events_json = json.dumps(
            analytics_data.get("attributed_events", [])[:20],  # limit for context
            indent=2,
        )
        stats_json = json.dumps(
            {str(k): v for k, v in
             list(analytics_data.get("player_stats", {}).items())[:15]},
            indent=2,
        )
        involvement_json = json.dumps(
            {str(k): v for k, v in
             analytics_data.get("event_involvement", {}).items()},
            indent=2,
        )

        user_msg = MATCH_SUMMARY_USER.format(
            events_json=events_json,
            player_stats_json=stats_json,
            involvement_json=involvement_json,
            num_events=analytics_data.get("num_events", 0),
            num_attributed=analytics_data.get("num_attributed", 0),
            num_players=analytics_data.get("num_players_tracked", 0),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MATCH_SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw = json.loads(response.choices[0].message.content)
        return MatchNarrative(**raw)

    def generate_from_file(self, analytics_path):
        """Generate narrative from a saved analytics JSON file."""
        with open(analytics_path) as f:
            data = json.load(f)
        return self.generate(data)


def save_narrative(narrative, output_path):
    """Save a MatchNarrative to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(narrative.model_dump(), f, indent=2)


def load_narrative(path):
    """Load a MatchNarrative from JSON."""
    with open(path) as f:
        data = json.load(f)
    return MatchNarrative(**data)
