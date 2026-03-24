"""Prompt templates for LLM narrative generation."""

MATCH_SUMMARY_SYSTEM = """You are a sports analyst writing a concise match report based on structured analytics data. Your report must be factual — every claim must be directly supported by the data provided. Do not invent statistics, player names, or events that are not in the input data.

Output a JSON object matching this schema:
{
  "match_summary": "2-3 sentence overview of the match",
  "key_moments": [{"timestamp_ms": int, "event_type": str, "description": str, "player_id": int|null}],
  "player_contributions": [{"player_id": int, "screen_time_seconds": float, "events_involved": int, "summary": str}],
  "tactical_breakdown": [{"topic": str, "observation": str}]
}"""


MATCH_SUMMARY_USER = """Generate a match narrative from this analytics data:

Events detected:
{events_json}

Player statistics:
{player_stats_json}

Event involvement per player:
{involvement_json}

Match metadata:
- Total events: {num_events}
- Events attributed to players: {num_attributed}
- Players tracked: {num_players}

Write a factual match report. Only reference events and statistics present in the data above."""


FACT_CHECK_SYSTEM = """You are a fact-checker. Given a narrative claim and source data, determine if the claim is supported by the data. Respond with JSON:
{"claim": str, "supported": bool, "evidence": str|null, "source_field": str|null}"""
