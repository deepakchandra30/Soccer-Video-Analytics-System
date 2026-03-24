"""Deterministic fact-check validator for LLM-generated match narratives."""
import re
from typing import List

from src.narratives.schemas import MatchNarrative, FactCheckResult


def validate_narrative(narrative, analytics_data):
    """Validate narrative claims against source analytics data."""
    results = []

    known_players = set()
    for k in analytics_data.get("player_stats", {}):
        known_players.add(int(k) if isinstance(k, str) else k)

    known_events = analytics_data.get("attributed_events", [])
    event_types = {e.get("label", "") for e in known_events}

    for moment in narrative.key_moments:
        # verify event type exists in data
        if moment.event_type in event_types:
            results.append(FactCheckResult(
                claim=f"Key moment: {moment.event_type} at {moment.timestamp_ms}ms",
                supported=True,
                evidence=f"Event type '{moment.event_type}' found in detected events",
                source_field="attributed_events",
            ))
        else:
            results.append(FactCheckResult(
                claim=f"Key moment: {moment.event_type} at {moment.timestamp_ms}ms",
                supported=False,
                evidence=f"Event type '{moment.event_type}' not in detected events: {event_types}",
                source_field="attributed_events",
            ))

        if moment.player_id is not None and moment.player_id not in known_players:
            results.append(FactCheckResult(
                claim=f"Player #{moment.player_id} involved in {moment.event_type}",
                supported=False,
                evidence=f"Player #{moment.player_id} not in tracked players",
                source_field="player_stats",
            ))

    for contrib in narrative.player_contributions:
        if contrib.player_id in known_players:
            stats = analytics_data.get("player_stats", {})
            player_key = contrib.player_id
            player_data = stats.get(player_key) or stats.get(str(player_key))

            if player_data:
                actual_time = player_data.get("screen_time_seconds", 0)
                claimed_time = contrib.screen_time_seconds
                if abs(claimed_time - actual_time) <= max(actual_time * 0.2, 1.0):
                    results.append(FactCheckResult(
                        claim=f"Player #{contrib.player_id} screen time: {claimed_time}s",
                        supported=True,
                        evidence=f"Actual: {actual_time}s (within tolerance)",
                        source_field="player_stats",
                    ))
                else:
                    results.append(FactCheckResult(
                        claim=f"Player #{contrib.player_id} screen time: {claimed_time}s",
                        supported=False,
                        evidence=f"Actual: {actual_time}s (outside tolerance)",
                        source_field="player_stats",
                    ))
            results.append(FactCheckResult(
                claim=f"Player #{contrib.player_id} exists",
                supported=True,
                evidence="Found in tracked players",
                source_field="player_stats",
            ))
        else:
            results.append(FactCheckResult(
                claim=f"Player #{contrib.player_id} exists",
                supported=False,
                evidence=f"Not in tracked players: {known_players}",
                source_field="player_stats",
            ))

    return results


def compute_factcheck_score(results):
    """Compute overall fact-check pass rate, returns supported/total/score."""
    total = len(results)
    supported = sum(1 for r in results if r.supported)
    return {
        "total_claims": total,
        "supported": supported,
        "unsupported": total - supported,
        "score": supported / max(total, 1),
    }
