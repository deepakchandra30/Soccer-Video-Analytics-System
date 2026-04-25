"""Smoke tests for analytics CLI script."""
import subprocess
import sys


def test_run_analytics_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_analytics.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--events-json" in result.stdout
    assert "--tracks-json" in result.stdout
