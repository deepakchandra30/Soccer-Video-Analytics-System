"""Smoke tests for CLI scripts."""
import subprocess
import sys


def test_run_ablation_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_ablation.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--experiments" in result.stdout


def test_generate_report_artifacts_cli_help():
    result = subprocess.run(
        [sys.executable, "scripts/generate_report_artifacts.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--results-json" in result.stdout