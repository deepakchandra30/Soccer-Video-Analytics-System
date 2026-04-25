"""Tests for fact-check validator."""
import pytest

from src.narratives.schemas import MatchNarrative, KeyMoment, PlayerContribution, TacticalPoint
from src.narratives.factcheck import validate_narrative, compute_factcheck_score


@pytest.fixture
def good_narrative():
    return MatchNarrative(
        match_summary="Match with a goal and a foul.",
        key_moments=[
            KeyMoment(timestamp_ms=1423000, event_type="Goal",
                      description="Opening goal", player_id=7),
        ],
        player_contributions=[
            PlayerContribution(player_id=7, screen_time_seconds=0.2,
                               events_involved=1, summary="Scored"),
        ],
        tactical_breakdown=[
            TacticalPoint(topic="Play", observation="Balanced"),
        ],
    )


@pytest.fixture
def bad_narrative():
    return MatchNarrative(
        match_summary="Player 99 scored a hat trick.",
        key_moments=[
            KeyMoment(timestamp_ms=0, event_type="Penalty",
                      description="Penalty kick", player_id=99),
        ],
        player_contributions=[
            PlayerContribution(player_id=99, screen_time_seconds=90.0,
                               events_involved=3, summary="Dominated"),
        ],
        tactical_breakdown=[],
    )


class TestValidateNarrative:
    def test_good_narrative_supported(self, good_narrative, sample_analytics):
        results = validate_narrative(good_narrative, sample_analytics)
        score = compute_factcheck_score(results)
        assert score["score"] > 0.5

    def test_bad_narrative_unsupported(self, bad_narrative, sample_analytics):
        results = validate_narrative(bad_narrative, sample_analytics)
        score = compute_factcheck_score(results)
        # bad narrative references non-existent player and event type
        assert score["unsupported"] > 0


class TestFactCheckScore:
    def test_perfect_score(self):
        from src.narratives.schemas import FactCheckResult
        results = [
            FactCheckResult(claim="A", supported=True),
            FactCheckResult(claim="B", supported=True),
        ]
        score = compute_factcheck_score(results)
        assert score["score"] == 1.0

    def test_zero_score(self):
        from src.narratives.schemas import FactCheckResult
        results = [
            FactCheckResult(claim="A", supported=False),
        ]
        score = compute_factcheck_score(results)
        assert score["score"] == 0.0

    def test_empty_results(self):
        score = compute_factcheck_score([])
        assert score["total_claims"] == 0
        assert score["score"] == 0.0
