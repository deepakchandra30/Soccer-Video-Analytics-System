"""Wrapper around SoccerNet's official evaluation for action spotting.

Important: always use version=2 + metric="tight" to get numbers comparable
to published results. version=2 is the API flag for v3 labels (confusing but
that's how the SDK works). "tight" gives the standard avg-mAP used in papers.
"""
import json
import os
import sys
import wandb
import torch
from SoccerNet.Evaluation.ActionSpotting import evaluate as _sn_evaluate
from SoccerNet.utils import getListGames


def _ensure_empty_predictions(predictions_path, split, prediction_file):
    """Create empty prediction files for any games missing predictions.

    The SDK evaluator crashes if a prediction file is absent. Games without
    downloaded features never had predictions generated, so we stub them out
    with an empty predictions list so evaluation can continue.
    """
    for game in getListGames(split=split):
        pred_file = os.path.join(predictions_path, game, prediction_file)
        if not os.path.exists(pred_file):
            os.makedirs(os.path.join(predictions_path, game), exist_ok=True)
            with open(pred_file, "w") as f:
                json.dump({"predictions": []}, f)


def run_evaluation(soccernet_path, predictions_path, split="valid",
                   prediction_file="results_spotting.json"):
    """Run official SoccerNet eval with the correct params (version=2, tight)."""
    _ensure_empty_predictions(predictions_path, split, prediction_file)
    return _sn_evaluate(
        SoccerNet_path=soccernet_path,
        Predictions_path=predictions_path,
        split=split,
        version=2,
        prediction_file=prediction_file,
        metric="tight",
    )


def run_evaluation_smoke_test(soccernet_path, predictions_path,
                              expected_map=49.7, tolerance=5.0):
    """Validate eval harness against known NetVLAD baseline (~49.7% avg-mAP).

    If mAP > 80% you probably used metric="loose" instead of "tight".
    If mAP near 0% you probably used version=1 instead of version=2.
    """
    wandb.init(
        project="soccer-analytics",
        name="eval-smoke-test",
        config={
            "seed": 42,
            "pytorch_version": torch.__version__,
            "python_version": sys.version,
            "feature_type": "ResNET_TF2_PCA512",
            "metric": "tight",
        },
    )

    results = run_evaluation(soccernet_path, predictions_path)
    avg_map = results.get("a_mAP", results.get("average_mAP", 0.0))

    wandb.log({"eval/avg_mAP_tight": avg_map})
    wandb.finish()

    deviation = abs(avg_map - expected_map)
    assert deviation <= tolerance, (
        f"Smoke test FAILED: got {avg_map:.2f}%, expected {expected_map}±{tolerance}%"
    )
    print(f"Smoke test passed: avg-mAP = {avg_map:.2f}%")
    return avg_map
