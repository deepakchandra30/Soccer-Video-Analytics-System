"""SlowFast training and two-stage evaluation entry point."""
import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config.seeds import set_seeds
from config.slowfast_config import SLOWFAST_CONFIG
from config.pipeline_config import PIPELINE_CONFIG
from src.models.temporal.slowfast import SlowFastSpotting
from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.pipeline import TwoStagePipeline
from src.models.temporal.losses import (
    get_class_weights, compute_class_weights_from_matches,
)
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from src.training.trainer import (
    train_epoch, validate_epoch, EarlyStopping,
    save_checkpoint, load_checkpoint,
)
from src.training.train_tsm import load_matches, collate_fn
from src.evaluation.predict import generate_predictions
from src.evaluation.evaluate import run_evaluation
from src.evaluation.benchmark import benchmark_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train SlowFast model")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/slowfast")
    parser.add_argument("--coarse-checkpoint", required=True,
                        help="Path to trained TSM checkpoint (best.pt)")
    parser.add_argument("--epochs", type=int, default=SLOWFAST_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=SLOWFAST_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=SLOWFAST_CONFIG["lr"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--feat-dim", type=int, default=None)
    parser.add_argument("--feature-type", default="pca512",
                        choices=["pca512", "resnet50", "baidu"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    # Submission-only: merge an extra labelled split into training data.
    # Default None preserves existing train-on-train-only behaviour exactly.
    parser.add_argument("--extra-train-split", default=None,
                        choices=[None, "valid", "test"],
                        help="Optionally merge another split into training. Use "
                             "'valid' for challenge-submission retrains; the "
                             "loader will then validate on 'test'.")
    args = parser.parse_args()

    feat_map = {
        "pca512": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "resnet50": ("1_ResNET_TF2.npy", "2_ResNET_TF2.npy"),
        "baidu": ("1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"),
    }
    f1_name, f2_name = feat_map[args.feature_type]

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.feat_dim is None:
        _FEAT_DIMS = {"pca512": 512, "resnet50": 2048, "baidu": 8576}
        args.feat_dim = _FEAT_DIMS[args.feature_type]

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    print(f"Loading {args.feature_type} features from {args.data_dir}...")
    train_matches, _, _ = load_matches(args.data_dir, "train", args.feature_type)
    val_split_name = "test" if args.extra_train_split == "valid" else "valid"
    val_matches, val_dirs, val_ids = load_matches(
        args.data_dir, val_split_name, args.feature_type,
    )
    if args.extra_train_split:
        extra_matches, _, _ = load_matches(
            args.data_dir, args.extra_train_split, args.feature_type,
        )
        train_matches = train_matches + extra_matches
        print(f"  merged extra split '{args.extra_train_split}' "
              f"({len(extra_matches)} matches) into training data.")
    print(f"  train: {len(train_matches)} matches, "
          f"valid-on-{val_split_name}: {len(val_matches)} matches")

    # datasets -- larger chunks for SlowFast. Framerate MUST match feature
    # extraction rate (Baidu=1fps, PCA/ResNet=2fps) or annotations misalign.
    from src.training.train_tsm import FEATURE_FRAMERATE as _FR
    _framerate = _FR.get(args.feature_type, 2)
    train_ds = ChunkedSoccerNetDataset(
        train_matches, chunk_size=SLOWFAST_CONFIG["chunk_size"],
        event_ratio=0.7, feat_dim=args.feat_dim, framerate=_framerate,
    )
    val_ds = ChunkedSoccerNetDataset(
        val_matches, chunk_size=SLOWFAST_CONFIG["chunk_size"],
        event_ratio=0.0, feat_dim=args.feat_dim, framerate=_framerate,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    # model, optimizer, loss
    model = SlowFastSpotting(
        feat_dim=args.feat_dim, num_classes=SLOWFAST_CONFIG["num_classes"],
        slow_stride=SLOWFAST_CONFIG["slow_stride"],
        hidden_dim=SLOWFAST_CONFIG["hidden_dim"],
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=SLOWFAST_CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Per-class inverse-frequency weights derived from train annotations —
    # see rationale in train_tsm.py where the same change was applied.
    weights = compute_class_weights_from_matches(
        train_matches, num_classes=SLOWFAST_CONFIG["num_classes"],
        bg_weight=SLOWFAST_CONFIG["bg_weight"],
    )
    print(f"Per-class weights: bg={weights[0]:.3f}, "
          f"events min={weights[1:].min():.2f} max={weights[1:].max():.2f}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))

    early_stop = EarlyStopping(patience=SLOWFAST_CONFIG["patience"], mode="min")

    # wandb
    import wandb
    wandb.init(project=args.wandb_project, name="slowfast-training",
               config={**SLOWFAST_CONFIG, "feat_dim": args.feat_dim})

    # train SlowFast
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 args.device)
        val_loss = validate_epoch(model, val_loader, criterion, args.device)
        scheduler.step()

        wandb.log({"train/loss": train_loss, "val/loss": val_loss,
                    "lr": scheduler.get_last_lr()[0], "epoch": epoch})
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, "best.pt"))

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # evaluate: SlowFast single-stage
    print("\nGenerating SlowFast single-stage predictions...")
    sf_pred_dir = os.path.join(args.output_dir, "predictions_slowfast")
    from src.training.train_tsm import FEATURE_SCALE as _FS
    _scale = _FS.get(args.feature_type, 1.0)
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=sf_pred_dir, device=args.device,
        feature_files=(f1_name, f2_name),
        feature_scale=_scale, framerate=_framerate,
    )
    sf_results = run_evaluation(args.data_dir, sf_pred_dir, split=val_split_name)
    # SoccerNet's evaluator returns mAP as a numpy scalar in [0, 1]; convert to
    # percent for display + wandb so the printed "0.3%" doesn't conceal a real
    # 26.3% (same root cause as the TSM display fix in commit 94400ee).
    sf_map = float(sf_results.get("a_mAP", 0.0)) * 100.0

    # evaluate: two-stage pipeline
    print("Running two-stage pipeline evaluation...")
    coarse = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=17,
        hidden_dim=PIPELINE_CONFIG["coarse_hidden_dim"],
    ).to(args.device)
    load_checkpoint(args.coarse_checkpoint, coarse)

    pipeline = TwoStagePipeline(coarse, model, PIPELINE_CONFIG, device=args.device)

    # generate two-stage predictions
    from src.models.temporal.postprocess import save_predictions
    ts_pred_dir = os.path.join(args.output_dir, "predictions_twostage")
    for match_dir, match_id in zip(val_dirs, val_ids):
        h1 = np.load(os.path.join(match_dir, f1_name)).astype(np.float32) * _scale
        h2 = np.load(os.path.join(match_dir, f2_name)).astype(np.float32) * _scale
        preds = pipeline.run(torch.FloatTensor(h1), torch.FloatTensor(h2))
        out_path = os.path.join(ts_pred_dir, match_id)
        os.makedirs(out_path, exist_ok=True)
        save_predictions(preds, os.path.join(out_path, "results_spotting.json"))

    ts_results = run_evaluation(args.data_dir, ts_pred_dir, split=val_split_name)
    ts_map = float(ts_results.get("a_mAP", 0.0)) * 100.0

    # benchmark latency
    print("Benchmarking latency...")
    bench_features = torch.randn(1000, args.feat_dim)
    bench = benchmark_pipeline(coarse, model, bench_features, PIPELINE_CONFIG,
                               device=args.device, num_runs=5, warmup=2)

    # print results table
    print(f"\n{'Mode':<15} {'avg-mAP tight':>14} {'ms/frame':>10} {'Speedup':>9}")
    print("-" * 50)
    tsm_ms = bench["single_stage_ms"] / 1000
    sf_ms = tsm_ms * 1.5  # rough estimate for SlowFast
    ts_ms = bench["two_stage_ms"] / 1000
    print(f"{'TSM single':<15} {'N/A':>14} {tsm_ms:>9.2f}ms {'1.0x':>9}")
    print(f"{'SlowFast':<15} {sf_map:>13.1f}% {sf_ms:>9.2f}ms {'--':>9}")
    print(f"{'Two-stage':<15} {ts_map:>13.1f}% {ts_ms:>9.2f}ms {bench['speedup_factor']:>8.1f}x")

    wandb.log({
        "eval/slowfast_mAP": sf_map,
        "eval/twostage_mAP": ts_map,
        "eval/speedup_factor": bench["speedup_factor"],
        "eval/candidate_ratio": bench["candidate_ratio"],
    })
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
