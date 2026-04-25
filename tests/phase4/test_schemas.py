"""Tests for narrative Pydantic schemas."""
import pytest
from pydantic import ValidationError

from src.narratives.schemas import (
    MatchNarrative, KeyMoment, PlayerContribution, TacticalPoint, FactCheckResult,
)


class TestMatchNarrative:
    def test_valid_narrative(self):
        narr = MatchNarrative(
            match_summary="A tight match with two goals.",
            key_moments=[
                KeyMoment(timestamp_ms=1423000, event_type="Goal",
                          description="Opening goal", player_id=7),
            ],
            player_contributions=[
                PlayerContribution(player_id=7, screen_time_seconds=45.0,
                                   events_involved=1, summary="Scored the opener"),
            ],
            tactical_breakdown=[
                TacticalPoint(topic="Formation", observation="4-3-3 shape"),
            ],
        )
        assert narr.match_summary.startswith("A tight")
        assert len(narr.key_moments) == 1

    def test_empty_lists_valid(self):
        narr = MatchNarrative(
            match_summary="No events detected.",
            key_moments=[],
            player_contributions=[],
            tactical_breakdown=[],
        )
        assert len(narr.key_moments) == 0


class TestFactCheckResult:
    def test_supported_claim(self):
        r = FactCheckResult(claim="Goal at 23:43", supported=True,
                            evidence="Found in events", source_field="events")
        assert r.supported

    def test_unsupported_claim(self):
        r = FactCheckResult(claim="Hat trick", supported=False)
        assert not r.supported
