"""Tests for event-player attribution."""
import pytest

from src.analytics.attribution import (
    attribute_events_to_players,
    compute_event_involvement,
)


class TestAttribution:
    def test_returns_enriched_events(self, sample_events, sample_tracks):
        result = attribute_events_to_players(sample_events, sample_tracks)
        assert len(result) == len(sample_events)
        for e in result:
            assert "attributed_player" in e
            assert "attribution_distance" in e

    def test_preserves_original_fields(self, sample_events, sample_tracks):
        result = attribute_events_to_players(sample_events, sample_tracks)
        for orig, enriched in zip(sample_events, result):
            assert enriched["label"] == orig["label"]
            assert enriched["position"] == orig["position"]


class TestEventInvolvement:
    def test_counts_events(self, sample_analytics):
        involvement = compute_event_involvement(
            sample_analytics["attributed_events"]
        )
        # player 7 should have 1 Goal
        assert involvement[7]["Goal"] == 1
        assert involvement[7]["total"] == 1

    def test_unattributed_excluded(self, sample_analytics):
        involvement = compute_event_involvement(
            sample_analytics["attributed_events"]
        )
        # only players 7 and 11 should appear
        assert len(involvement) == 2
