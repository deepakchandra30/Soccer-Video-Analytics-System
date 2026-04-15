"""Training entry point for SlowFast model and two-stage evaluation."""
import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config.seeds import set_seeds
from config.slowfast_config import SLOWFAST_CONFIG
from src.models.temporal.slowfast import SlowFastSpotting
from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.pipeline import TwoStagePipeline
from src.models.temporal.losses import get_class_weights
from src.data.chunked_dataset import ChunkedSoccerNetDataset
from src.training.trainer import (
    train_epoch_downsampled, validate_epoch_downsampled,
    EarlyStopping, save_checkpoint,
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
    parser.add_argument("--feat-dim", type=int, default=SLOWFAST_CONFIG["feat_dim"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    print(f"Loading features from {args.data_dir}...")
    train_matches, _, _ = load_matches(args.data_dir, "train")
    val_matches, val_dirs, val_ids = load_matches(args.data_dir, "valid")
    print(f"  train: {len(train_matches)} matches, valid: {len(val_matches)} matches")

    # datasets -- larger chunks for SlowFast
    train_ds = ChunkedSoccerNetDataset(
        train_matches, chunk_size=SLOWFAST_CONFIG["chunk_size"],
        event_ratio=0.7, feat_dim=args.feat_dim,
    )
    val_ds = ChunkedSoccerNetDataset(
        val_matches, chunk_size=SLOWFAST_CONFIG["chunk_size"],
        event_ratio=0.0, feat_dim=args.feat_dim,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

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

    weights = get_class_weights(
        SLOWFAST_CONFIG["num_classes"],
        SLOWFAST_CONFIG["bg_weight"],
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))

    early_stop = EarlyStopping(patience=SLOWFAST_CONFIG["patience"], mode="min")

    # wandb
    import wandb
    wandb.init(project=args.wandb_project, name="slowfast-training",
               config={**SLOWFAST_CONFIG, "feat_dim": args.feat_dim})

    # train SlowFast
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch_downsampled(
            model, train_loader, optimizer, criterion, args.device,
            target_stride=SLOWFAST_CONFIG["slow_stride"],
        )
        val_loss = validate_epoch_downsampled(
            model, val_loader, criterion, args.device,
            target_stride=SLOWFAST_CONFIG["slow_stride"],
        )
        scheduler.step()

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(args.output_dir, "best.pt"))

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # --------------------------
    # Two-stage evaluation setup
    # --------------------------
    print("Loading coarse TSM checkpoint for two-stage pipeline...")
    coarse = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=SLOWFAST_CONFIG["num_classes"],
        hidden_dim=256, n_shifts=2,
    ).to(args.device)

    coarse_ckpt = torch.load(args.coarse_checkpoint, map_location=args.device, weights_only=False)
    coarse.load_state_dict(coarse_ckpt["model_state_dict"])
    coarse.eval()

    model.eval()

    pipeline_cfg = {
        "coarse_window_size": 40,
        "coarse_stride": 20,
        "coarse_nms_window": 30,
        "coarse_confidence_threshold": 0.2,
        "fine_pad_frames": 20,
        "framerate": 2,
    }
    pipeline = TwoStagePipeline(coarse_model=coarse, fine_model=model,
                                config=pipeline_cfg, device=args.device)

    # benchmark two-stage latency
    try:
        bench = benchmark_pipeline(pipeline, val_dirs[:3], num_runs=1, device=args.device)
        print("\nTwo-stage benchmark:")
        print(f"  mean time/match: {bench['mean_time_per_match_s']:.2f}s")
        print(f"  candidate ratio: {bench['mean_candidate_ratio']*100:.1f}%")
        wandb.log({
            "benchmark/mean_time_per_match_s": bench["mean_time_per_match_s"],
            "benchmark/candidate_ratio": bench["mean_candidate_ratio"],
        })
    except Exception as e:
        print(f"Benchmark warning: {e}")

    # generate predictions with fine model as single-stage fallback output
    print("Generating SlowFast predictions on validation split...")
    pred_dir = os.path.join(args.output_dir, "predictions")
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=pred_dir,
        feature_files=("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        window_size=80, stride=40, nms_window=30,
        confidence_threshold=0.2, framerate=2, device=args.device,
    )

    print("Running SoccerNet evaluation...")
    results = run_evaluation(args.data_dir, pred_dir, split="valid")
    avg_map = results.get("a_mAP", 0.0)
    print(f"\nSlowFast mAP Results:\n  avg-mAP tight: {avg_map:.1f}%")
    wandb.log({"eval/avg_mAP_tight": avg_map})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()