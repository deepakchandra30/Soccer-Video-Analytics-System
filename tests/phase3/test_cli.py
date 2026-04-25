"""Smoke tests for tracking CLI scripts."""
import subprocess
import sys


def test_run_tracking_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_tracking.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--output-dir" in result.stdout
