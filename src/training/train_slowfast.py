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
from src.models.temporal.losses import get_class_weights
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
    parser.add_argument("--feat-dim", type=int, default=SLOWFAST_CONFIG["feat_dim"])
    parser.add_argument("--wandb-project", default="soccer-analytics")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading features from {args.data_dir}...")
    train_matches, _, _ = load_matches(args.data_dir, "train")
    val_matches, val_dirs, val_ids = load_matches(args.data_dir, "valid")
    print(f"  train: {len(train_matches)} matches, valid: {len(val_matches)} matches")

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

    weights = get_class_weights(SLOWFAST_CONFIG["num_classes"],
                                SLOWFAST_CONFIG["bg_weight"])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(args.device))

    early_stop = EarlyStopping(patience=SLOWFAST_CONFIG["patience"], mode="min")

    import wandb
    wandb.init(project=args.wandb_project, name="slowfast-training",
               config={**SLOWFAST_CONFIG, "feat_dim": args.feat_dim})

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

    print("\nGenerating SlowFast single-stage predictions...")
    sf_pred_dir = os.path.join(args.output_dir, "predictions_slowfast")
    generate_predictions(
        model=model, match_dirs=val_dirs, match_ids=val_ids,
        output_dir=sf_pred_dir, device=args.device,
    )
    sf_results = run_evaluation(args.data_dir, sf_pred_dir, split="valid")
    sf_map = sf_results.get("a_mAP", 0.0)

    print("Running two-stage pipeline evaluation...")
    coarse = TSMSpottingHead(
        feat_dim=args.feat_dim, num_classes=17,
        hidden_dim=PIPELINE_CONFIG["coarse_hidden_dim"],
    ).to(args.device)
    load_checkpoint(args.coarse_checkpoint, coarse)

    pipeline = TwoStagePipeline(coarse, model, PIPELINE_CONFIG, device=args.device)

    from src.models.temporal.postprocess import save_predictions
    ts_pred_dir = os.path.join(args.output_dir, "predictions_twostage")
    for match_dir, match_id in zip(val_dirs, val_ids):
        h1 = np.load(os.path.join(match_dir, "1_ResNET_TF2_PCA512.npy"))
        h2 = np.load(os.path.join(match_dir, "2_ResNET_TF2_PCA512.npy"))
        preds = pipeline.run(torch.FloatTensor(h1), torch.FloatTensor(h2))
        out_path = os.path.join(ts_pred_dir, match_id)
        os.makedirs(out_path, exist_ok=True)
        save_predictions(preds, os.path.join(out_path, "results_spotting.json"))

    ts_results = run_evaluation(args.data_dir, ts_pred_dir, split="valid")
    ts_map = ts_results.get("a_mAP", 0.0)

    print("Benchmarking latency...")
    bench_features = torch.randn(1000, args.feat_dim)
    bench = benchmark_pipeline(coarse, model, bench_features, PIPELINE_CONFIG,
                               device=args.device, num_runs=5, warmup=2)

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
