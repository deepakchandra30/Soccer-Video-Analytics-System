"""
Tests for wandb experiment logging initialization.

Phase 1 Plan 03 — TDD RED->GREEN cycle.
Confirms wandb.init() runs in offline mode without error and logs seed,
pytorch_version, and feature_type as required by the plan.
"""
import os
import pytest
import wandb


def test_wandb_init_offline():
    """
    wandb.init() must succeed in offline mode (no network required).
    Logs seed=42, pytorch_version, python_version, feature_type.
    """
    os.environ["WANDB_MODE"] = "offline"
    try:
        run = wandb.init(
            project="soccer-analytics",
            name="test-run-phase1",
            config={
                "seed": 42,
                "pytorch_version": __import__("torch").__version__,
                "python_version": __import__("sys").version,
                "feature_type": "ResNET_TF2_PCA512",
            }
        )
        assert wandb.run is not None, "wandb.run must not be None after successful init()"
        assert wandb.run.config["seed"] == 42, "seed must be logged as 42"
        wandb.finish()
    except Exception as exc:
        pytest.fail(f"wandb.init() raised an exception in offline mode: {exc}")
