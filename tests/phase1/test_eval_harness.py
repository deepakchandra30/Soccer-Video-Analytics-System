"""
Tests for the evaluation harness wrapper.

Phase 1 Plan 03 — TDD RED->GREEN cycle.
These tests confirm:
  1. SoccerNet evaluate() is importable and callable.
  2. run_evaluation() is parameterized with version=2 and metric='tight'.
"""
import inspect
import pytest


def test_evaluate_import():
    """Import evaluate from SoccerNet SDK. Fails if package not installed."""
    from SoccerNet.Evaluation.ActionSpotting import evaluate
    assert callable(evaluate), "evaluate() must be callable"


def test_evaluate_params_documented():
    """
    Confirm run_evaluation() source contains version=2 and metric='tight'.
    This guards against accidental parameter drift — any change breaks this test.
    """
    from src.evaluation.evaluate import run_evaluation
    source = inspect.getsource(run_evaluation)
    assert "version=2" in source, (
        "run_evaluation() must call _sn_evaluate with version=2 "
        "(required for v3 label format, despite confusing naming)"
    )
    assert 'metric="tight"' in source, (
        "run_evaluation() must call _sn_evaluate with metric='tight' "
        "(required to match published SOTA numbers)"
    )
