"""Smoke tests for SlowFast training script."""
import subprocess
import sys


def test_imports():
    """Verify train_slowfast imports without error."""
    from src.training.train_slowfast import main
    assert callable(main)


def test_cli_help():
    """Verify --help exits cleanly and mentions expected args."""
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_slowfast", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--coarse-checkpoint" in result.stdout
