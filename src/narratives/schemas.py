"""Pydantic schemas for structured narrative output."""
from typing import List, Optional
from pydantic import BaseModel


class KeyMoment(BaseModel):
    timestamp_ms: int
    event_type: str
    description: str
    player_id: Optional[int] = None


class PlayerContribution(BaseModel):
    player_id: int
    screen_time_seconds: float
    events_involved: int
    summary: str


class TacticalPoint(BaseModel):
    topic: str
    observation: str


class MatchNarrative(BaseModel):
    match_summary: str
    key_moments: List[KeyMoment]
    player_contributions: List[PlayerContribution]
    tactical_breakdown: List[TacticalPoint]


class FactCheckResult(BaseModel):
    claim: str
    supported: bool
    evidence: Optional[str] = None
    source_field: Optional[str] = None
