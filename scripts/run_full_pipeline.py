#!/usr/bin/env python
"""Reproduce the headline tight avg-mAP with one command.

Orchestrates the three stages that produce the ensemble result documented
in ``outputs/map_results.txt``:

    1. TSM training          -> outputs/tsm_fixed/best.pt
    2. SlowFast training     -> outputs/slowfast_fixed/best.pt
    3. Ensemble inference    -> outputs/final_ensemble/ + prints mAP

Stage 3 (scripts/run_ensemble.py) is what prints the headline number;
running only stage 1 yields the TSM single-model score (~31-32%), which
is the source of the "client is seeing 31% not 42%" confusion.

Implementation notes:
    - Pure orchestrator: calls existing scripts as-is via subprocess, does
      not import or modify any code in src/ or config/.
    - Skips a training stage if its best.pt already exists, so rerunning
      after tuning ensemble knobs does not retrain from scratch.
    - Fails fast on any stage error. A submission pipeline producing no
      number is safer than one producing a misleading number.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


# Ensemble flags that produced 42.37% tight mAP on PCA-512 features.
# Source: outputs/map_results.txt (row "TSM 0.2 / SF 0.8, multi-scale, hard-NMS").
# Change these if you want to try a different operating point; the orchestrator
# always forwards them verbatim to scripts/run_ensemble.py.
HEADLINE_TSM_WEIGHT = 0.2
HEADLINE_SCALES = "40:20,80:40"
HEADLINE_NMS_MODE = "hard"
HEADLINE_CONF_THRESHOLD = 0.2


def run_stage(name, cmd, env):
    """Run one pipeline stage, failing fast with a diagnostic on error."""
    banner = "=" * 60
    print(f"\n{banner}\n[{name}]\n  $ {' '.join(cmd)}\n{banner}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"\n[run_full_pipeline] ERROR: {name} exited with code {exc.returncode}.",
              file=sys.stderr)
        print("[run_full_pipeline] The pipeline stops here. Fix the error above and "
              "rerun — stages whose best.pt already exists will be skipped, so you "
              "won't retrain from scratch.", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"\n[run_full_pipeline] ERROR: could not find python for {name}. "
              "Is your virtualenv active?", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the headline ensemble mAP")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--tsm-output-dir", default="outputs/tsm_fixed",
                        help="TSM checkpoint dir. Matches outputs/map_results.txt.")
    parser.add_argument("--slowfast-output-dir", default="outputs/slowfast_fixed",
                        help="SlowFast checkpoint dir. Matches outputs/map_results.txt.")
    parser.add_argument("--ensemble-output-dir", default="outputs/final_ensemble")
    parser.add_argument("--feature-type", default="pca512",
                        choices=["pca512", "resnet50", "baidu"],
                        help="Forwarded to all three stages. Baidu requires the "
                             "8576-dim features from scripts/download_features.py "
                             "--features baidu.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain both models even if best.pt already exists.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # Verify prerequisites before spending a minute on training.
    data_dir = (repo_root / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    if not data_dir.exists():
        print(f"[run_full_pipeline] ERROR: data directory {data_dir} not found.\n"
              f"Run: python scripts/download_features.py --local-dir {args.data_dir}",
              file=sys.stderr)
        sys.exit(1)
    ensemble_script = repo_root / "scripts" / "run_ensemble.py"
    if not ensemble_script.exists():
        print(f"[run_full_pipeline] ERROR: {ensemble_script} missing. "
              "Has the repo been modified?", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    # Default to online wandb
    # If they already set WANDB_MODE we respect it.
    env.setdefault("WANDB_MODE", "online")

    tsm_ckpt = Path(args.tsm_output_dir) / "best.pt"
    sf_ckpt = Path(args.slowfast_output_dir) / "best.pt"

    # Stage 1: TSM
    if tsm_ckpt.exists() and not args.force_retrain:
        print(f"[1/3] TSM checkpoint found at {tsm_ckpt} — skipping training.")
    else:
        run_stage("1/3  Training TSM", [
            sys.executable, "-m", "src.training.train_tsm",
            "--data-dir", str(args.data_dir),
            "--output-dir", args.tsm_output_dir,
            "--feature-type", args.feature_type,
        ], env)

    # Stage 2: SlowFast (needs TSM checkpoint as coarse-stage)
    if not tsm_ckpt.exists():
        print(f"[run_full_pipeline] ERROR: TSM training completed but {tsm_ckpt} "
              "was not produced.", file=sys.stderr)
        sys.exit(2)
    if sf_ckpt.exists() and not args.force_retrain:
        print(f"[2/3] SlowFast checkpoint found at {sf_ckpt} — skipping training.")
    else:
        run_stage("2/3  Training SlowFast", [
            sys.executable, "-m", "src.training.train_slowfast",
            "--data-dir", str(args.data_dir),
            "--output-dir", args.slowfast_output_dir,
            "--coarse-checkpoint", str(tsm_ckpt),
            "--feature-type", args.feature_type,
        ], env)

    if not sf_ckpt.exists():
        print(f"[run_full_pipeline] ERROR: SlowFast training completed but "
              f"{sf_ckpt} was not produced.", file=sys.stderr)
        sys.exit(2)

    # Stage 3: Ensemble — the step that prints the headline mAP.
    run_stage("3/3  TSM+SlowFast Ensemble (headline mAP)", [
        sys.executable, str(ensemble_script),
        "--data-dir", str(args.data_dir),
        "--tsm-checkpoint", str(tsm_ckpt),
        "--slowfast-checkpoint", str(sf_ckpt),
        "--output-dir", args.ensemble_output_dir,
        "--feature-type", args.feature_type,
        "--tsm-weight", str(HEADLINE_TSM_WEIGHT),
        "--scales", HEADLINE_SCALES,
        "--nms-mode", HEADLINE_NMS_MODE,
        "--confidence-threshold", str(HEADLINE_CONF_THRESHOLD),
    ], env)

    banner = "=" * 60
    print(f"\n{banner}")
    print("Pipeline complete. The 'avg-mAP tight' line printed just above is the")
    print("headline number for this run. Individual-model scores printed during")
    print("stages 1 and 2 are components, not the final result.")
    print(f"Predictions saved to: {args.ensemble_output_dir}/predictions/")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
