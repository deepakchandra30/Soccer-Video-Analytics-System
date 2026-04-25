"""Tests for narrative quality evaluation (BLEU)."""
import pytest

from src.narratives.schemas import MatchNarrative, KeyMoment, PlayerContribution, TacticalPoint
from src.narratives.evaluate import compute_bleu, evaluate_narrative


class TestBLEU:
    def test_identical_text(self):
        score = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_different_text(self):
        score = compute_bleu("the cat sat on the mat",
                              "a dog ran through the park")
        assert score < 0.5

    def test_empty_text(self):
        assert compute_bleu("", "something") == 0.0
        assert compute_bleu("something", "") == 0.0


class TestEvaluateNarrative:
    def test_returns_metrics(self):
        narr = MatchNarrative(
            match_summary="A competitive match with two goals.",
            key_moments=[
                KeyMoment(timestamp_ms=0, event_type="Goal", description="Goal"),
            ],
            player_contributions=[],
            tactical_breakdown=[],
        )
        metrics = evaluate_narrative(narr)
        assert metrics["num_key_moments"] == 1
        assert metrics["summary_length_words"] > 0

    def test_bleu_with_reference(self):
        narr = MatchNarrative(
            match_summary="A competitive match with two goals scored.",
            key_moments=[],
            player_contributions=[],
            tactical_breakdown=[],
        )
        metrics = evaluate_narrative(narr,
                                      reference_summary="A competitive match with two goals.")
        assert "bleu_score" in metrics
        assert metrics["bleu_score"] > 0.5
