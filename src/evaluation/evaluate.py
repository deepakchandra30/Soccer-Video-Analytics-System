"""Wrapper around SoccerNet's official evaluation for action spotting.

Important: always use version=2 + metric="tight" to get numbers comparable
to published results. version=2 is the API flag for v3 labels (confusing but
that's how the SDK works). "tight" gives the standard avg-mAP used in papers.
"""
import json
import os
import sys

import torch
import wandb
from SoccerNet.Evaluation.ActionSpotting import evaluate as _sn_evaluate


def run_evaluation(soccernet_path, predictions_path, split="valid",
                   prediction_file="results_spotting.json"):
    """Run official SoccerNet eval (version=2, tight) over games we can score.

    The upstream SDK iterates every game returned by ``getListGames(split)``
    and opens both the label file and the prediction file with a raw
    ``json.load(open(...))``.  A single missing file raises
    ``FileNotFoundError`` halfway through the loop, discarding 100% of the
    scores computed so far.  This wrapper:

      * stubs an empty prediction file for any game that has labels but no
        prediction (common when features were only downloaded for a subset),
      * restricts the SDK's game list to games that actually have
        ``Labels-v2.json`` on disk (we cannot synthesize ground truth), and
      * logs exactly how many games fell into each bucket so the caller can
        tell whether the mAP covers the full split or a subset.
    """
    import SoccerNet.Evaluation.ActionSpotting as _sn_module
    from SoccerNet.utils import getListGames

    all_games = getListGames(split=split)

    evaluable = []
    stubbed = 0
    missing_labels = []
    for game in all_games:
        label_path = os.path.join(soccernet_path, game, "Labels-v2.json")
        if not os.path.exists(label_path):
            missing_labels.append(game)
            continue

        pred_path = os.path.join(predictions_path, game, prediction_file)
        if not os.path.exists(pred_path):
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump({"UrlLocal": game, "predictions": []}, f)
            stubbed += 1

        evaluable.append(game)

    print(f"  Eval coverage: {len(evaluable)}/{len(all_games)} games scored, "
          f"{stubbed} stubbed (no features), {len(missing_labels)} skipped "
          f"(no Labels-v2.json).")
    if missing_labels:
        print("  To include them, run: "
              "soccernet.download(split='valid', files=['Labels-v2.json'])")

    if not evaluable:
        raise RuntimeError(
            f"No {split} games have both Labels-v2.json and predictions. "
            f"Cannot evaluate.")

    # The SDK reads ``getListGames`` via ``from ... import`` so we patch the
    # name bound inside its module, not ``SoccerNet.utils.getListGames``
    # directly. Restore on the way out even if the evaluator raises.
    original_get_list_games = _sn_module.getListGames
    _sn_module.getListGames = lambda *args, **kwargs: evaluable
    try:
        return _sn_evaluate(
            SoccerNet_path=soccernet_path,
            Predictions_path=predictions_path,
            split=split,
            version=2,
            prediction_file=prediction_file,
            metric="tight",
        )
    finally:
        _sn_module.getListGames = original_get_list_games


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
